from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Request Model
# ══════════════════════════════════════════════════════════════

class SOMRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    generate: bool = False
    nAssets: int = 60
    seed: Optional[int] = None
    # SOM config
    gridX: int = 8
    gridY: int = 8
    epochs: int = 500
    sigma: float = 1.5         # neighborhood radius
    learningRate: float = 0.5
    # K-means on top of SOM
    nClusters: int = 5


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _to_native(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_native(x) for x in obj]
    return obj

def safe_float(val, default=0.0):
    try:
        if val is None: return default
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════
# Asset Universe for Data Generation
# ══════════════════════════════════════════════════════════════

SECTOR_PROFILES = {
    'Technology':    {'ret': (0.15, 0.35), 'vol': (0.22, 0.45), 'beta': (1.1, 1.6), 'pe': (20, 50), 'de': (0.1, 0.5),  'mcap': (50, 3000)},
    'Healthcare':    {'ret': (0.08, 0.20), 'vol': (0.18, 0.30), 'beta': (0.6, 1.0), 'pe': (15, 35), 'de': (0.2, 0.6),  'mcap': (30, 500)},
    'Financial':     {'ret': (0.08, 0.18), 'vol': (0.18, 0.30), 'beta': (1.0, 1.4), 'pe': (8, 15),  'de': (1.5, 5.0),  'mcap': (50, 600)},
    'Consumer':      {'ret': (0.06, 0.15), 'vol': (0.12, 0.22), 'beta': (0.5, 0.9), 'pe': (15, 30), 'de': (0.3, 0.8),  'mcap': (40, 400)},
    'Energy':        {'ret': (0.05, 0.20), 'vol': (0.25, 0.45), 'beta': (0.8, 1.3), 'pe': (5, 15),  'de': (0.5, 1.5),  'mcap': (20, 500)},
    'Industrial':    {'ret': (0.08, 0.16), 'vol': (0.15, 0.25), 'beta': (0.8, 1.2), 'pe': (12, 25), 'de': (0.4, 1.0),  'mcap': (20, 300)},
    'Utilities':     {'ret': (0.04, 0.10), 'vol': (0.10, 0.18), 'beta': (0.3, 0.6), 'pe': (12, 22), 'de': (0.8, 2.0),  'mcap': (15, 100)},
    'Real Estate':   {'ret': (0.05, 0.12), 'vol': (0.15, 0.28), 'beta': (0.6, 1.0), 'pe': (15, 40), 'de': (1.0, 3.0),  'mcap': (10, 100)},
}

RISK_FEATURES = [
    'annual_return', 'annual_vol', 'beta', 'sharpe',
    'max_drawdown', 'pe_ratio', 'debt_equity',
    'market_cap_bn', 'var_95',
]


def generate_asset_data(n_assets: int = 60, seed=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sectors = list(SECTOR_PROFILES.keys())
    rows = []

    # Ticker generation
    consonants = 'BCDFGHJKLMNPQRSTVWXYZ'
    vowels = 'AEIOU'
    used_tickers = set()

    for i in range(n_assets):
        sector = sectors[i % len(sectors)]
        sp = SECTOR_PROFILES[sector]

        # Generate unique ticker
        while True:
            length = rng.choice([3, 4])
            ticker = ''
            for j in range(length):
                ticker += rng.choice(list(consonants)) if j % 2 == 0 else rng.choice(list(vowels))
            if ticker not in used_tickers:
                used_tickers.add(ticker)
                break

        # Risk characteristics
        ret = rng.uniform(*sp['ret'])
        vol = rng.uniform(*sp['vol'])
        beta = rng.uniform(*sp['beta'])
        sharpe = (ret - 0.04) / vol if vol > 0 else 0
        # Max drawdown correlated with vol
        mdd = -(vol * rng.uniform(1.2, 2.5))
        pe = rng.uniform(*sp['pe'])
        de = rng.uniform(*sp['de'])
        mcap = rng.uniform(*sp['mcap'])
        # VaR 95 (parametric)
        var95 = -(ret / 252 - 1.645 * vol / np.sqrt(252)) * 100

        rows.append({
            'ticker': ticker,
            'sector': sector,
            'annual_return': round(ret * 100, 2),      # %
            'annual_vol': round(vol * 100, 2),          # %
            'beta': round(beta, 3),
            'sharpe': round(sharpe, 3),
            'max_drawdown': round(mdd * 100, 2),        # %
            'pe_ratio': round(pe, 1),
            'debt_equity': round(de, 3),
            'market_cap_bn': round(mcap, 1),
            'var_95': round(var95, 3),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# SOM Training (MiniSom)
# ══════════════════════════════════════════════════════════════

def train_som(
    X: np.ndarray,
    grid_x: int = 8, grid_y: int = 8,
    epochs: int = 500,
    sigma: float = 1.5,
    learning_rate: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    from minisom import MiniSom

    som = MiniSom(
        grid_x, grid_y, X.shape[1],
        sigma=sigma,
        learning_rate=learning_rate,
        random_seed=seed,
        neighborhood_function='gaussian',
    )
    som.random_weights_init(X)
    som.train(X, epochs, verbose=False)

    # Quantization error over training (sample)
    q_errors = []
    check_points = list(range(0, epochs, max(1, epochs // 20))) + [epochs - 1]
    for ep in check_points:
        # Re-train to this point for error tracking (approximate)
        q_err = som.quantization_error(X)
        q_errors.append({'epoch': ep, 'q_error': safe_float(q_err)})

    # BMU for each sample
    bmus = np.array([som.winner(x) for x in X])

    # U-matrix (distance between neighboring weight vectors)
    weights = som.get_weights()  # (grid_x, grid_y, n_features)
    u_matrix = np.zeros((grid_x, grid_y))
    for i in range(grid_x):
        for j in range(grid_y):
            neighbors = []
            if i > 0: neighbors.append(weights[i - 1, j])
            if i < grid_x - 1: neighbors.append(weights[i + 1, j])
            if j > 0: neighbors.append(weights[i, j - 1])
            if j < grid_y - 1: neighbors.append(weights[i, j + 1])
            u_matrix[i, j] = np.mean([np.linalg.norm(weights[i, j] - n) for n in neighbors])

    # Hit map (number of samples per node)
    hit_map = np.zeros((grid_x, grid_y), dtype=int)
    for bx, by in bmus:
        hit_map[bx, by] += 1

    # Topographic error
    topo_error = som.topographic_error(X)

    return {
        'bmus': bmus,
        'weights': weights,
        'u_matrix': u_matrix,
        'hit_map': hit_map,
        'q_error': safe_float(som.quantization_error(X)),
        'topo_error': safe_float(topo_error),
        'q_errors': q_errors,
    }


# ══════════════════════════════════════════════════════════════
# K-Means on SOM nodes → Cluster labels
# ══════════════════════════════════════════════════════════════

def cluster_som_nodes(
    weights: np.ndarray,
    bmus: np.ndarray,
    n_clusters: int = 5,
) -> Dict[str, Any]:
    from sklearn.cluster import KMeans

    grid_x, grid_y, n_features = weights.shape
    flat_weights = weights.reshape(-1, n_features)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    node_labels = km.fit_predict(flat_weights)
    node_labels_2d = node_labels.reshape(grid_x, grid_y)

    # Map sample → cluster via BMU
    sample_labels = np.array([node_labels_2d[bx, by] for bx, by in bmus])

    # Cluster centroids in original feature space
    centroids = km.cluster_centers_

    return {
        'node_labels': node_labels_2d,
        'sample_labels': sample_labels,
        'centroids': centroids,
        'inertia': safe_float(km.inertia_),
    }


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/som-cluster")
async def som_cluster(request: SOMRequest):
    try:
        from sklearn.preprocessing import StandardScaler

        # ── 1. Data ──
        if request.generate or not request.data:
            df = generate_asset_data(request.nAssets, request.seed)
            features = RISK_FEATURES
        else:
            df = pd.DataFrame(request.data)
            features = [c for c in df.columns if c not in ('ticker', 'sector', 'name', 'company') and not c.startswith('_')]
            for f in features:
                df[f] = pd.to_numeric(df[f], errors='coerce')
            df = df.dropna(subset=features)

        n = len(df)
        n_feat = len(features)
        if n < 10:
            raise HTTPException(status_code=400, detail=f"Need >=10 assets, got {n}")

        X = df[features].values.astype(np.float64)

        # ── 2. Scale ──
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ── 3. SOM ──
        grid_x = min(request.gridX, max(3, int(np.sqrt(n))))
        grid_y = min(request.gridY, max(3, int(np.sqrt(n))))

        som_result = train_som(
            X_scaled, grid_x, grid_y,
            epochs=request.epochs,
            sigma=request.sigma,
            learning_rate=request.learningRate,
        )

        # ── 4. Cluster ──
        n_clusters = min(request.nClusters, n // 2, grid_x * grid_y // 2)
        cluster_result = cluster_som_nodes(
            som_result['weights'], som_result['bmus'], n_clusters,
        )

        df['_cluster'] = cluster_result['sample_labels']
        df['_bmu_x'] = som_result['bmus'][:, 0]
        df['_bmu_y'] = som_result['bmus'][:, 1]

        # ── 5. Cluster Profiles ──
        cluster_profiles = []
        for c in range(n_clusters):
            mask = df['_cluster'] == c
            subset = df[mask]
            profile = {
                'cluster': int(c),
                'count': int(mask.sum()),
            }
            if 'sector' in df.columns:
                sector_dist = subset['sector'].value_counts().to_dict()
                profile['sectors'] = {k: int(v) for k, v in sector_dist.items()}
                profile['top_sector'] = subset['sector'].mode().iloc[0] if len(subset) > 0 else ''

            for f in features:
                profile[f'{f}_mean'] = safe_float(subset[f].mean())
                profile[f'{f}_std'] = safe_float(subset[f].std())

            # Risk characterization
            if 'annual_vol' in features and 'annual_return' in features:
                avg_vol = subset['annual_vol'].mean() if 'annual_vol' in subset else 0
                avg_ret = subset['annual_return'].mean() if 'annual_return' in subset else 0
                if avg_vol > 30 and avg_ret > 20:
                    profile['risk_label'] = 'High Growth / High Risk'
                elif avg_vol > 30:
                    profile['risk_label'] = 'High Risk / Low Return'
                elif avg_ret > 15:
                    profile['risk_label'] = 'Growth'
                elif avg_vol < 18:
                    profile['risk_label'] = 'Defensive / Low Vol'
                else:
                    profile['risk_label'] = 'Balanced'
            else:
                profile['risk_label'] = f'Cluster {c}'

            cluster_profiles.append(profile)

        # ── 6. Chart Data ──

        # Asset scatter (BMU positions + jitter)
        asset_scatter = []
        for _, row in df.iterrows():
            entry = {
                'ticker': str(row.get('ticker', '')),
                'sector': str(row.get('sector', '')),
                'cluster': int(row['_cluster']),
                'bmu_x': int(row['_bmu_x']) + np.random.uniform(-0.3, 0.3),
                'bmu_y': int(row['_bmu_y']) + np.random.uniform(-0.3, 0.3),
            }
            for f in features:
                entry[f] = safe_float(row[f])
            asset_scatter.append(entry)

        # U-matrix heatmap
        u_matrix = som_result['u_matrix']
        u_heatmap = []
        for i in range(grid_x):
            for j in range(grid_y):
                u_heatmap.append({
                    'x': i, 'y': j,
                    'distance': safe_float(u_matrix[i, j]),
                    'cluster': int(cluster_result['node_labels'][i, j]),
                    'hits': int(som_result['hit_map'][i, j]),
                })

        # Hit map data
        hit_data = []
        for i in range(grid_x):
            for j in range(grid_y):
                hit_data.append({'x': i, 'y': j, 'hits': int(som_result['hit_map'][i, j])})

        # Feature comparison by cluster (radar-like)
        feat_comparison = []
        for f in features:
            entry: Dict[str, Any] = {'feature': f}
            for c in range(n_clusters):
                mask = df['_cluster'] == c
                entry[f'cluster_{c}'] = safe_float(df.loc[mask, f].mean())
            feat_comparison.append(entry)

        # Sector distribution per cluster
        sector_chart = []
        if 'sector' in df.columns:
            for c in range(n_clusters):
                mask = df['_cluster'] == c
                for sector, count in df.loc[mask, 'sector'].value_counts().items():
                    sector_chart.append({'cluster': int(c), 'sector': str(sector), 'count': int(count)})

        # Concentration analysis
        concentration = []
        if 'sector' in df.columns:
            for c in range(n_clusters):
                mask = df['_cluster'] == c
                n_in = int(mask.sum())
                if n_in > 0:
                    sector_shares = df.loc[mask, 'sector'].value_counts(normalize=True)
                    hhi = float((sector_shares ** 2).sum())
                    concentration.append({
                        'cluster': int(c),
                        'n_assets': n_in,
                        'hhi': safe_float(hhi),
                        'top_sector_pct': safe_float(sector_shares.iloc[0] * 100),
                        'n_sectors': int(sector_shares.shape[0]),
                    })

        # ── 7. Response ──
        results = {
            'n_assets': n,
            'n_features': n_feat,
            'features': features,
            'grid': {'x': grid_x, 'y': grid_y},
            'n_clusters': n_clusters,
            'som': {
                'q_error': som_result['q_error'],
                'topo_error': som_result['topo_error'],
                'epochs': request.epochs,
                'sigma': request.sigma,
                'learning_rate': request.learningRate,
            },
            'cluster_profiles': cluster_profiles,
            'concentration': concentration,
            'charts': {
                'asset_scatter': asset_scatter,
                'u_heatmap': u_heatmap,
                'hit_map': hit_data,
                'feature_comparison': feat_comparison,
                'sector_distribution': sector_chart,
                'training_error': som_result['q_errors'],
            },
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
