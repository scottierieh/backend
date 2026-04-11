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


class SegmentationRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    generate: bool = False
    nCustomers: int = 500
    seed: Optional[int] = None
    maxK: int = 8
    method: str = 'kmeans'   # kmeans | dbscan
    # Column mapping
    colCustomerId: Optional[str] = None
    colRecency: Optional[str] = None
    colFrequency: Optional[str] = None
    colMonetary: Optional[str] = None
    # Optional extra columns
    colAge: Optional[str] = None
    colChannel: Optional[str] = None
    colCategory: Optional[str] = None


def _to_native(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    elif isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    elif isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [_to_native(x) for x in obj]
    return obj

def safe_float(val, default=0.0):
    try:
        if val is None: return default
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except Exception: return default


# ══════════════════════════════════════════════════════════════
# Data Generation
# ══════════════════════════════════════════════════════════════

CHANNELS = ['Online', 'Mobile App', 'In-Store', 'Email', 'Social']
CATEGORIES = ['Electronics', 'Fashion', 'Food', 'Health', 'Home', 'Sports']
SEGMENT_ARCHETYPES = [
    # (name, recency_range, freq_range, monetary_range, pct)
    ('Champions',    (1, 15),   (20, 60),  (800, 3000),  0.10),
    ('Loyal',        (10, 40),  (10, 30),  (400, 1200),  0.18),
    ('Promising',    (5, 30),   (5, 15),   (200, 600),   0.15),
    ('New',          (1, 20),   (1, 3),    (50, 300),    0.12),
    ('At Risk',      (60, 150), (8, 25),   (300, 1000),  0.15),
    ('Needs Attention', (40, 90), (3, 10), (100, 500),   0.13),
    ('Hibernating',  (120, 365),(1, 5),    (30, 200),    0.10),
    ('Lost',         (200, 500),(1, 3),    (10, 100),    0.07),
]


def generate_customers(n: int, seed=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for archetype, r_range, f_range, m_range, pct in SEGMENT_ARCHETYPES:
        n_seg = max(1, int(n * pct))
        for _ in range(n_seg):
            recency = rng.integers(r_range[0], r_range[1] + 1)
            frequency = rng.integers(f_range[0], f_range[1] + 1)
            monetary = round(rng.uniform(m_range[0], m_range[1]), 2)
            age = rng.integers(18, 70)
            channel = rng.choice(CHANNELS)
            category = rng.choice(CATEGORIES)
            avg_order = round(monetary / max(frequency, 1), 2)
            tenure_days = rng.integers(30, 1200)
            rows.append({
                'customer_id': f'C{len(rows)+1:05d}',
                'recency': int(recency),
                'frequency': int(frequency),
                'monetary': float(monetary),
                'avg_order_value': float(avg_order),
                'tenure_days': int(tenure_days),
                'age': int(age),
                'channel': channel,
                'top_category': category,
            })

    # Pad or trim to exact n
    while len(rows) < n:
        rows.append(rows[rng.integers(0, len(rows))].copy())
        rows[-1]['customer_id'] = f'C{len(rows):05d}'
    rows = rows[:n]
    rng.shuffle(rows)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# Segmentation Engine
# ══════════════════════════════════════════════════════════════

SEGMENT_LABELS = {
    'high_r_high_f_high_m': 'Champions',
    'mid_r_high_f_high_m': 'Loyal',
    'high_r_mid_f_mid_m': 'Promising',
    'high_r_low_f_low_m': 'New Customers',
    'low_r_high_f_high_m': 'At Risk',
    'low_r_mid_f_mid_m': 'Needs Attention',
    'low_r_low_f_low_m': 'Hibernating',
    'very_low_r': 'Lost',
}

SEGMENT_STRATEGIES = {
    'Champions': {'action': 'Reward & Advocate', 'tactics': 'Loyalty programs, early access, referral incentives, VIP treatment', 'priority': 1},
    'Loyal': {'action': 'Upsell & Cross-sell', 'tactics': 'Personalized recommendations, bundle offers, exclusive content', 'priority': 2},
    'Promising': {'action': 'Nurture & Grow', 'tactics': 'Onboarding series, category expansion, engagement programs', 'priority': 3},
    'New Customers': {'action': 'Onboard & Engage', 'tactics': 'Welcome series, first-purchase discount, product education', 'priority': 4},
    'At Risk': {'action': 'Reactivate & Retain', 'tactics': 'Win-back campaigns, special offers, feedback surveys', 'priority': 2},
    'Needs Attention': {'action': 'Re-engage', 'tactics': 'Targeted promotions, reminder emails, limited-time offers', 'priority': 3},
    'Hibernating': {'action': 'Win-back', 'tactics': 'Deep discount, "we miss you" campaigns, new product alerts', 'priority': 4},
    'Lost': {'action': 'Last Resort', 'tactics': 'Final reactivation attempt, survey for churn reasons, sunset policy', 'priority': 5},
}


def auto_label_segment(centroid_r, centroid_f, centroid_m, all_centroids):
    """Label segment based on RFM centroid position relative to median."""
    med_r = np.median([c[0] for c in all_centroids])
    med_f = np.median([c[1] for c in all_centroids])
    med_m = np.median([c[2] for c in all_centroids])

    # Recency: lower is better (more recent)
    r_level = 'high' if centroid_r < med_r * 0.6 else ('mid' if centroid_r < med_r * 1.4 else 'low')
    f_level = 'high' if centroid_f > med_f * 1.4 else ('mid' if centroid_f > med_f * 0.6 else 'low')
    m_level = 'high' if centroid_m > med_m * 1.4 else ('mid' if centroid_m > med_m * 0.6 else 'low')

    if centroid_r > med_r * 2.5:
        return 'Lost'
    key = f'{r_level}_r_{f_level}_f_{m_level}_m'
    return SEGMENT_LABELS.get(key, f'Segment ({r_level}R/{f_level}F/{m_level}M)')


def estimate_clv(recency, frequency, monetary, tenure_days, discount_rate=0.1):
    """
    Simple CLV estimate:
    CLV = avg_order × purchase_rate × expected_lifetime × margin
    """
    tenure_years = max(tenure_days / 365.0, 0.1)
    purchase_rate = frequency / tenure_years  # purchases per year
    avg_order = monetary / max(frequency, 1)
    margin = 0.3  # assumed 30% margin

    # Retention probability (higher freq + lower recency = higher retention)
    recency_factor = max(0, 1 - recency / 365)
    freq_factor = min(1, purchase_rate / 12)
    retention = 0.3 + 0.4 * recency_factor + 0.3 * freq_factor
    retention = np.clip(retention, 0.05, 0.98)

    # Expected lifetime in years
    expected_lifetime = 1 / (1 - retention) if retention < 1 else 10

    clv = avg_order * purchase_rate * expected_lifetime * margin
    return safe_float(clv), safe_float(retention), safe_float(purchase_rate)


def estimate_churn_probability(recency, frequency, monetary, tenure_days):
    """
    Logistic-style churn probability based on RFM signals.
    High recency (days since last purchase) → high churn risk.
    High frequency → low churn risk.
    """
    r_score = min(recency / 365, 1.0)   # normalized recency (higher = worse)
    f_score = min(frequency / 30, 1.0)  # normalized frequency (higher = better)
    m_score = min(monetary / 2000, 1.0) # normalized monetary (higher = better)

    # Logistic combination
    z = 2.0 * r_score - 1.5 * f_score - 0.5 * m_score + 0.3
    churn_prob = 1 / (1 + np.exp(-z * 3))

    return safe_float(churn_prob)


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/audience-segmentation")
async def audience_segmentation(request: SegmentationRequest):
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.metrics import silhouette_score, calinski_harabasz_score

        rng = np.random.default_rng(request.seed)

        # ── 1. Data ──
        if request.generate or not request.data:
            df = generate_customers(request.nCustomers, request.seed)
            col_r, col_f, col_m = 'recency', 'frequency', 'monetary'
            col_id = 'customer_id'
        else:
            df = pd.DataFrame(request.data)
            # Column mapping
            col_id = request.colCustomerId or next((c for c in df.columns if 'customer' in c.lower() or 'id' in c.lower()), df.columns[0])
            col_r = request.colRecency or next((c for c in df.columns if 'recen' in c.lower()), None)
            col_f = request.colFrequency or next((c for c in df.columns if 'freq' in c.lower()), None)
            col_m = request.colMonetary or next((c for c in df.columns if 'monet' in c.lower() or 'revenue' in c.lower() or 'spend' in c.lower() or 'amount' in c.lower()), None)
            if not all([col_r, col_f, col_m]):
                raise HTTPException(status_code=400, detail=f"Cannot auto-detect RFM columns. Found: R={col_r}, F={col_f}, M={col_m}")

        for c in [col_r, col_f, col_m]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=[col_r, col_f, col_m])
        n = len(df)
        if n < 20:
            raise HTTPException(status_code=400, detail=f"Need >=20 customers. Got {n}.")

        R = df[col_r].values
        F = df[col_f].values
        M = df[col_m].values

        # ── 2. Feature Matrix ──
        rfm = np.column_stack([R, F, M])
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm)

        # ── 3. Optimal K (Elbow + Silhouette) ──
        max_k = min(request.maxK, n // 5, 10)
        k_range = range(2, max_k + 1)
        inertias = []
        silhouettes = []
        ch_scores = []

        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(rfm_scaled)
            inertias.append(safe_float(km.inertia_))
            sil = silhouette_score(rfm_scaled, labels) if len(set(labels)) > 1 else 0
            silhouettes.append(safe_float(sil))
            ch = calinski_harabasz_score(rfm_scaled, labels) if len(set(labels)) > 1 else 0
            ch_scores.append(safe_float(ch))

        optimal_k = list(k_range)[np.argmax(silhouettes)]

        elbow_chart = [{'k': int(k), 'inertia': iner, 'silhouette': sil, 'calinski_harabasz': ch}
                       for k, iner, sil, ch in zip(k_range, inertias, silhouettes, ch_scores)]

        # ── 4. Final Clustering ──
        if request.method == 'dbscan':
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=5).fit(rfm_scaled)
            distances, _ = nn.kneighbors(rfm_scaled)
            eps = float(np.percentile(distances[:, -1], 90))
            clusterer = DBSCAN(eps=eps, min_samples=5)
            labels = clusterer.fit_predict(rfm_scaled)
            # Remap noise (-1) to its own cluster
            if -1 in labels:
                labels[labels == -1] = labels.max() + 1
            final_k = len(set(labels))
            final_silhouette = safe_float(silhouette_score(rfm_scaled, labels)) if final_k > 1 else 0
        else:
            km = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
            labels = km.fit_predict(rfm_scaled)
            final_k = optimal_k
            final_silhouette = safe_float(silhouette_score(rfm_scaled, labels))

        df['segment_id'] = labels

        # ── 5. Segment Profiling ──
        centroids_raw = []
        for seg in range(final_k):
            mask = labels == seg
            centroids_raw.append([R[mask].mean(), F[mask].mean(), M[mask].mean()])

        segment_profiles = []
        for seg in range(final_k):
            mask = df['segment_id'] == seg
            seg_df = df[mask]
            r_mean = seg_df[col_r].mean()
            f_mean = seg_df[col_f].mean()
            m_mean = seg_df[col_m].mean()

            label = auto_label_segment(r_mean, f_mean, m_mean, centroids_raw)
            strategy = SEGMENT_STRATEGIES.get(label, SEGMENT_STRATEGIES['Needs Attention'])

            # CLV & churn per customer
            clvs, retentions, churn_probs = [], [], []
            for _, row in seg_df.iterrows():
                tenure = row.get('tenure_days', 365)
                clv, ret, _ = estimate_clv(row[col_r], row[col_f], row[col_m], tenure)
                churn = estimate_churn_probability(row[col_r], row[col_f], row[col_m], tenure)
                clvs.append(clv)
                retentions.append(ret)
                churn_probs.append(churn)

            # Channel distribution
            channel_dist = {}
            if 'channel' in seg_df.columns:
                vc = seg_df['channel'].value_counts(normalize=True)
                channel_dist = {k: safe_float(v * 100) for k, v in vc.items()}

            # Category distribution
            category_dist = {}
            if 'top_category' in seg_df.columns:
                vc = seg_df['top_category'].value_counts(normalize=True)
                category_dist = {k: safe_float(v * 100) for k, v in vc.items()}

            segment_profiles.append({
                'segment_id': int(seg),
                'label': label,
                'size': int(mask.sum()),
                'pct': safe_float(mask.sum() / n * 100),
                'rfm': {
                    'recency_mean': safe_float(r_mean),
                    'recency_median': safe_float(seg_df[col_r].median()),
                    'frequency_mean': safe_float(f_mean),
                    'frequency_median': safe_float(seg_df[col_f].median()),
                    'monetary_mean': safe_float(m_mean),
                    'monetary_median': safe_float(seg_df[col_m].median()),
                    'monetary_total': safe_float(seg_df[col_m].sum()),
                },
                'clv': {
                    'mean': safe_float(np.mean(clvs)),
                    'median': safe_float(np.median(clvs)),
                    'total': safe_float(np.sum(clvs)),
                },
                'churn': {
                    'mean_prob': safe_float(np.mean(churn_probs)),
                    'high_risk_pct': safe_float((np.array(churn_probs) > 0.6).mean() * 100),
                },
                'retention_mean': safe_float(np.mean(retentions)),
                'channel_distribution': channel_dist,
                'category_distribution': category_dist,
                'strategy': strategy,
            })

        # Sort by CLV total descending
        segment_profiles.sort(key=lambda x: x['clv']['total'], reverse=True)

        # ── 6. Customer-Level Data (sample) ──
        customer_sample = []
        step = max(1, n // 300)
        for i in range(0, n, step):
            row = df.iloc[i]
            tenure = row.get('tenure_days', 365)
            clv, ret, pr = estimate_clv(row[col_r], row[col_f], row[col_m], tenure)
            churn = estimate_churn_probability(row[col_r], row[col_f], row[col_m], tenure)
            seg_id = int(row['segment_id'])
            seg_label = next((s['label'] for s in segment_profiles if s['segment_id'] == seg_id), f'Seg {seg_id}')

            customer_sample.append({
                'customer_id': str(row.get(col_id, f'C{i}')),
                'recency': safe_float(row[col_r]),
                'frequency': safe_float(row[col_f]),
                'monetary': safe_float(row[col_m]),
                'segment': seg_label,
                'clv': clv,
                'churn_prob': churn,
            })

        # ── 7. Charts ──

        # Radar data (normalized RFM per segment)
        r_max = max(s['rfm']['recency_mean'] for s in segment_profiles) or 1
        f_max = max(s['rfm']['frequency_mean'] for s in segment_profiles) or 1
        m_max = max(s['rfm']['monetary_mean'] for s in segment_profiles) or 1

        radar_data = []
        for s in segment_profiles:
            radar_data.append({
                'label': s['label'],
                'recency': safe_float(1 - s['rfm']['recency_mean'] / r_max),  # invert: low recency = good
                'frequency': safe_float(s['rfm']['frequency_mean'] / f_max),
                'monetary': safe_float(s['rfm']['monetary_mean'] / m_max),
                'clv': safe_float(s['clv']['mean'] / max(sp['clv']['mean'] for sp in segment_profiles)) if max(sp['clv']['mean'] for sp in segment_profiles) > 0 else 0,
                'retention': safe_float(s['retention_mean']),
            })

        # CLV distribution by segment
        clv_dist = []
        for s in segment_profiles:
            clv_dist.append({
                'segment': s['label'],
                'clv_mean': s['clv']['mean'],
                'clv_total': s['clv']['total'],
                'size': s['size'],
                'churn_mean': s['churn']['mean_prob'],
            })

        # Segment size chart
        size_chart = [{'segment': s['label'], 'count': s['size'], 'pct': s['pct'], 'monetary_total': s['rfm']['monetary_total']}
                      for s in segment_profiles]

        # RFM scatter (sampled)
        scatter = []
        step_sc = max(1, n // 500)
        for i in range(0, n, step_sc):
            seg_id = int(labels[i])
            seg_label = next((s['label'] for s in segment_profiles if s['segment_id'] == seg_id), f'Seg {seg_id}')
            scatter.append({
                'recency': safe_float(R[i]),
                'frequency': safe_float(F[i]),
                'monetary': safe_float(M[i]),
                'segment': seg_label,
            })

        # Churn risk by segment
        churn_chart = [{'segment': s['label'], 'churn_prob': s['churn']['mean_prob'], 'high_risk_pct': s['churn']['high_risk_pct']}
                       for s in segment_profiles]

        # Strategy table
        strategy_table = [{
            'segment': s['label'],
            'size': s['size'],
            'action': s['strategy']['action'],
            'tactics': s['strategy']['tactics'],
            'priority': s['strategy']['priority'],
            'clv_mean': s['clv']['mean'],
            'churn_mean': s['churn']['mean_prob'],
        } for s in segment_profiles]
        strategy_table.sort(key=lambda x: x['priority'])

        # ── 8. Summary ──
        total_clv = sum(s['clv']['total'] for s in segment_profiles)
        top_seg = segment_profiles[0] if segment_profiles else None

        results = {
            'n_customers': n,
            'n_segments': final_k,
            'optimal_k': optimal_k,
            'method': request.method,
            'silhouette_score': final_silhouette,
            'columns_used': {'recency': col_r, 'frequency': col_f, 'monetary': col_m},
            'summary': {
                'total_clv': safe_float(total_clv),
                'avg_clv': safe_float(total_clv / n),
                'avg_churn': safe_float(np.mean([s['churn']['mean_prob'] for s in segment_profiles])),
                'top_segment': top_seg['label'] if top_seg else '',
                'top_segment_clv_pct': safe_float(top_seg['clv']['total'] / total_clv * 100) if top_seg and total_clv > 0 else 0,
                'at_risk_customers': sum(s['size'] for s in segment_profiles if s['churn']['mean_prob'] > 0.5),
            },
            'segments': segment_profiles,
            'charts': {
                'elbow': elbow_chart,
                'scatter': scatter,
                'size': size_chart,
                'clv_distribution': clv_dist,
                'churn_risk': churn_chart,
                'radar': radar_data,
            },
            'strategy_table': strategy_table,
            'customer_sample': customer_sample[:200],
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
