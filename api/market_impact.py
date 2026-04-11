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

class MarketImpactRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    generate: bool = False
    ticker: str = 'AAPL'
    nTrades: int = 2000
    seed: Optional[int] = None
    # GMM config
    nComponents: int = 3
    # Almgren-Chriss params
    totalShares: int = 100_000     # order size
    timeHorizon: float = 1.0       # days
    riskAversion: float = 1e-6
    # Column mapping
    sizeCol: Optional[str] = None
    priceChangeCol: Optional[str] = None
    volumeCol: Optional[str] = None
    directionCol: Optional[str] = None


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
# Asset Profiles
# ══════════════════════════════════════════════════════════════

ASSET_PROFILES = {
    'AAPL':  {'name': 'Apple Inc.',       'price': 185, 'adv': 65_000_000, 'spread_bp': 2.0,  'daily_vol': 0.018, 'kyle_lambda': 2.5e-7},
    'MSFT':  {'name': 'Microsoft Corp.',  'price': 420, 'adv': 25_000_000, 'spread_bp': 1.8,  'daily_vol': 0.016, 'kyle_lambda': 5.0e-7},
    'GOOGL': {'name': 'Alphabet Inc.',    'price': 170, 'adv': 22_000_000, 'spread_bp': 2.5,  'daily_vol': 0.019, 'kyle_lambda': 4.5e-7},
    'TSLA':  {'name': 'Tesla Inc.',       'price': 245, 'adv': 95_000_000, 'spread_bp': 5.0,  'daily_vol': 0.035, 'kyle_lambda': 1.8e-7},
    'JPM':   {'name': 'JPMorgan Chase',   'price': 195, 'adv': 10_000_000, 'spread_bp': 2.0,  'daily_vol': 0.015, 'kyle_lambda': 8.0e-7},
    'NVDA':  {'name': 'NVIDIA Corp.',     'price': 880, 'adv': 40_000_000, 'spread_bp': 3.5,  'daily_vol': 0.030, 'kyle_lambda': 3.0e-7},
    'SPY':   {'name': 'S&P 500 ETF',     'price': 530, 'adv': 80_000_000, 'spread_bp': 0.5,  'daily_vol': 0.011, 'kyle_lambda': 0.8e-7},
    'GME':   {'name': 'GameStop Corp.',   'price': 25,  'adv': 5_000_000,  'spread_bp': 15.0, 'daily_vol': 0.055, 'kyle_lambda': 25e-7},
    'META':  {'name': 'Meta Platforms',   'price': 510, 'adv': 18_000_000, 'spread_bp': 2.8,  'daily_vol': 0.022, 'kyle_lambda': 6.0e-7},
    'AMZN':  {'name': 'Amazon.com Inc.',  'price': 185, 'adv': 45_000_000, 'spread_bp': 2.2,  'daily_vol': 0.020, 'kyle_lambda': 3.5e-7},
}

AVAILABLE_TICKERS = list(ASSET_PROFILES.keys())


# ══════════════════════════════════════════════════════════════
# Trade Data Generator
# ══════════════════════════════════════════════════════════════

def generate_trade_data(ticker: str, n_trades: int = 2000, seed=None) -> pd.DataFrame:
    """
    Generate realistic trade-level data with size-dependent price impact.
    Three regimes: small (retail), medium (institutional), large (block).
    """
    rng = np.random.default_rng(seed)
    p = ASSET_PROFILES.get(ticker.upper(), ASSET_PROFILES['AAPL'])

    price = p['price']
    adv = p['adv']
    lam = p['kyle_lambda']
    spread_bp = p['spread_bp']

    rows = []
    base_time = pd.Timestamp('2025-04-30 09:30:00')

    for i in range(n_trades):
        # Regime selection: 70% small, 22% medium, 8% large
        regime_r = rng.random()
        if regime_r < 0.70:
            regime = 'small'
            trade_size = int(rng.lognormal(np.log(200), 0.8))
            trade_size = max(10, min(trade_size, 5000))
        elif regime_r < 0.92:
            regime = 'medium'
            trade_size = int(rng.lognormal(np.log(8000), 0.6))
            trade_size = max(2000, min(trade_size, 50000))
        else:
            regime = 'large'
            trade_size = int(rng.lognormal(np.log(60000), 0.5))
            trade_size = max(20000, min(trade_size, 500000))

        direction = rng.choice([1, -1])  # buy/sell
        participation_rate = trade_size / adv

        # Price impact model:
        # Temporary: spread cost + sqrt impact
        # Permanent: Kyle lambda * signed volume
        spread_cost = price * spread_bp / 10000 / 2
        temp_impact = spread_cost + price * 0.1 * np.sqrt(participation_rate) * (1 + rng.normal(0, 0.3))
        perm_impact = lam * price * direction * trade_size * (1 + rng.normal(0, 0.2))

        total_impact_bp = (temp_impact + abs(perm_impact)) / price * 10000
        price_change_bp = direction * total_impact_bp * (1 + rng.normal(0, 0.15))

        # VWAP slippage
        vwap_slippage = temp_impact + rng.normal(0, spread_cost * 0.3)

        # Timestamp (random within trading hours)
        offset_sec = rng.integers(0, 23400)  # 6.5 hours
        ts = base_time + pd.Timedelta(seconds=int(offset_sec) - i * 5)

        rows.append({
            'timestamp': ts.strftime('%H:%M:%S'),
            'trade_size': trade_size,
            'direction': direction,
            'direction_label': 'BUY' if direction == 1 else 'SELL',
            'participation_rate': round(participation_rate * 100, 4),  # %
            'price_change_bp': round(price_change_bp, 2),
            'temp_impact_bp': round(temp_impact / price * 10000, 2),
            'perm_impact_bp': round(abs(perm_impact) / price * 10000, 2),
            'vwap_slippage_bp': round(vwap_slippage / price * 10000, 2),
            'regime': regime,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# GMM Regime Classification
# ══════════════════════════════════════════════════════════════

def fit_gmm_regimes(
    trade_sizes: np.ndarray,
    price_impacts: np.ndarray,
    n_components: int = 3,
) -> Dict[str, Any]:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    X = np.column_stack([np.log1p(trade_sizes), np.abs(price_impacts)])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=42,
        n_init=5,
    )
    gmm.fit(X_scaled)

    labels = gmm.predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)

    # Sort components by mean trade size (ascending)
    means_original = scaler.inverse_transform(gmm.means_)
    order = np.argsort(means_original[:, 0])
    label_map = {old: new for new, old in enumerate(order)}
    labels = np.array([label_map[l] for l in labels])

    regime_names = ['Small/Retail', 'Medium/Institutional', 'Large/Block']
    if n_components > 3:
        regime_names += [f'Regime {i}' for i in range(3, n_components)]
    regime_names = regime_names[:n_components]

    # Component stats
    components = []
    for c in range(n_components):
        mapped_c = order[c]
        mask = labels == c
        components.append({
            'regime': regime_names[c],
            'count': int(mask.sum()),
            'pct': safe_float(mask.mean() * 100),
            'mean_size': safe_float(np.exp(means_original[mapped_c, 0]) - 1),
            'mean_impact': safe_float(means_original[mapped_c, 1]),
            'avg_trade_size': safe_float(trade_sizes[mask].mean()),
            'avg_impact_bp': safe_float(np.abs(price_impacts[mask]).mean()),
            'median_impact_bp': safe_float(np.median(np.abs(price_impacts[mask]))),
        })

    return {
        'labels': labels,
        'probs': probs,
        'components': components,
        'bic': safe_float(gmm.bic(X_scaled)),
        'aic': safe_float(gmm.aic(X_scaled)),
        'n_components': n_components,
    }


# ══════════════════════════════════════════════════════════════
# Kyle Lambda Estimation
# ══════════════════════════════════════════════════════════════

def estimate_kyle_lambda(
    signed_volume: np.ndarray,
    price_changes: np.ndarray,
) -> Dict[str, Any]:
    """
    Kyle (1985) lambda: ΔP = λ · SignedVolume + ε
    Lambda measures permanent price impact per unit of order flow.
    """
    import statsmodels.api as sm

    X = sm.add_constant(signed_volume)
    model = sm.OLS(price_changes, X)
    result = model.fit(cov_type='HC1')  # heteroskedasticity-robust

    return {
        'lambda': safe_float(result.params[1]),
        'lambda_se': safe_float(result.bse[1]),
        'lambda_t': safe_float(result.tvalues[1]),
        'lambda_p': safe_float(result.pvalues[1]),
        'r_squared': safe_float(result.rsquared),
        'intercept': safe_float(result.params[0]),
    }


# ══════════════════════════════════════════════════════════════
# Almgren-Chriss Optimal Execution
# ══════════════════════════════════════════════════════════════

def almgren_chriss_schedule(
    total_shares: int,
    time_horizon: float,
    daily_vol: float,
    price: float,
    adv: int,
    kyle_lambda: float,
    risk_aversion: float = 1e-6,
    n_steps: int = 20,
) -> Dict[str, Any]:
    """
    Almgren-Chriss (2000) optimal execution schedule.
    Minimizes: E[cost] + risk_aversion · Var[cost]

    Temporary impact: η · (n_j / τ)
    Permanent impact: γ · n_j
    """
    tau = time_horizon / n_steps  # time per slice
    sigma = daily_vol * price     # daily vol in $

    # Impact parameters (calibrated from Kyle lambda)
    gamma = kyle_lambda * price   # permanent impact
    eta = gamma * 0.5             # temporary ≈ 0.5× permanent

    # Almgren-Chriss solution
    kappa_sq = risk_aversion * sigma ** 2 / (eta / tau)
    kappa = np.sqrt(max(kappa_sq, 1e-12))

    schedule = []
    remaining = total_shares
    cumulative_cost = 0

    for j in range(n_steps):
        t = j * tau
        # Optimal trajectory: x_j = X * sinh(κ(T-t)) / sinh(κT)
        T = time_horizon
        t_j = j * tau
        if kappa * T > 20:  # numerical guard
            x_j = total_shares * np.exp(-kappa * t_j)
        else:
            x_j = total_shares * np.sinh(kappa * (T - t_j)) / np.sinh(kappa * T) if np.sinh(kappa * T) != 0 else total_shares * (1 - t_j / T)

        n_j = remaining - max(x_j, 0)
        n_j = max(0, min(n_j, remaining))

        temp_cost = eta * (n_j / tau) * n_j if tau > 0 else 0
        perm_cost = gamma * n_j * remaining
        step_cost = temp_cost + perm_cost

        remaining -= n_j
        cumulative_cost += step_cost

        schedule.append({
            'step': j + 1,
            'time': round(t_j, 3),
            'shares_traded': int(n_j),
            'remaining': int(max(remaining, 0)),
            'participation_pct': safe_float(n_j / (adv * tau) * 100) if adv * tau > 0 else 0,
            'step_cost_bp': safe_float(step_cost / (price * total_shares) * 10000),
            'cumulative_cost_bp': safe_float(cumulative_cost / (price * total_shares) * 10000),
        })

    # Compare strategies
    # TWAP baseline
    twap_n = total_shares / n_steps
    twap_cost = sum(eta * (twap_n / tau) * twap_n + gamma * twap_n * (total_shares - j * twap_n) for j in range(n_steps))

    # Immediate execution
    imm_cost = eta * (total_shares / tau) * total_shares + gamma * total_shares * total_shares

    return {
        'schedule': schedule,
        'total_cost_bp': safe_float(cumulative_cost / (price * total_shares) * 10000),
        'twap_cost_bp': safe_float(twap_cost / (price * total_shares) * 10000),
        'immediate_cost_bp': safe_float(imm_cost / (price * total_shares) * 10000),
        'kappa': safe_float(kappa),
        'gamma': safe_float(gamma),
        'eta': safe_float(eta),
        'total_shares': total_shares,
        'time_horizon': time_horizon,
    }


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/market-impact")
async def market_impact(request: MarketImpactRequest):
    try:
        # ── 1. Data ──
        if request.generate or not request.data:
            df = generate_trade_data(request.ticker, request.nTrades, request.seed)
            size_col = 'trade_size'
            impact_col = 'price_change_bp'
            dir_col = 'direction'
        else:
            df = pd.DataFrame(request.data)

            def find_col(candidates, override):
                if override and override in df.columns: return override
                for c in candidates:
                    for col in df.columns:
                        if col.lower() == c.lower(): return col
                return None

            size_col = find_col(['trade_size', 'size', 'quantity', 'shares', 'volume'], request.sizeCol)
            impact_col = find_col(['price_change_bp', 'impact', 'price_change', 'return', 'slippage'], request.priceChangeCol)
            dir_col = find_col(['direction', 'side', 'dir'], request.directionCol)

            if not size_col or not impact_col:
                raise HTTPException(status_code=400, detail="Need trade_size and price_change columns.")

            df[size_col] = pd.to_numeric(df[size_col], errors='coerce')
            df[impact_col] = pd.to_numeric(df[impact_col], errors='coerce')
            df = df.dropna(subset=[size_col, impact_col])

        n = len(df)
        if n < 30:
            raise HTTPException(status_code=400, detail=f"Need >=30 trades, got {n}")

        trade_sizes = df[size_col].values.astype(float)
        price_changes = df[impact_col].values.astype(float)
        directions = df[dir_col].values.astype(float) if dir_col and dir_col in df.columns else np.sign(price_changes)
        signed_volume = directions * trade_sizes

        profile = ASSET_PROFILES.get(request.ticker.upper(), ASSET_PROFILES['AAPL'])

        # ── 2. GMM ──
        n_comp = min(request.nComponents, n // 10)
        gmm_results = fit_gmm_regimes(trade_sizes, price_changes, n_comp)
        df['_gmm_regime'] = gmm_results['labels']

        # ── 3. Kyle Lambda ──
        kyle = estimate_kyle_lambda(signed_volume, price_changes)

        # ── 4. Square-Root Impact Fit ──
        # |ΔP| = α + β·√(Size/ADV)
        participation = trade_sizes / profile['adv']
        sqrt_part = np.sqrt(participation)
        abs_impact = np.abs(price_changes)

        import statsmodels.api as sm
        X_sqrt = sm.add_constant(sqrt_part)
        sqrt_model = sm.OLS(abs_impact, X_sqrt).fit()
        sqrt_alpha = safe_float(sqrt_model.params[0])
        sqrt_beta = safe_float(sqrt_model.params[1])
        sqrt_r2 = safe_float(sqrt_model.rsquared)

        # ── 5. Almgren-Chriss ──
        ac = almgren_chriss_schedule(
            total_shares=request.totalShares,
            time_horizon=request.timeHorizon,
            daily_vol=profile['daily_vol'],
            price=profile['price'],
            adv=profile['adv'],
            kyle_lambda=kyle['lambda'] / 10000 if kyle['lambda'] != 0 else profile['kyle_lambda'],
            risk_aversion=request.riskAversion,
        )

        # ── 6. Charts ──

        # Size vs Impact scatter
        step_sc = max(1, n // 500)
        scatter = []
        for i in range(0, n, step_sc):
            scatter.append({
                'trade_size': safe_float(trade_sizes[i]),
                'impact_bp': safe_float(abs_impact[i]),
                'signed_impact': safe_float(price_changes[i]),
                'regime': int(gmm_results['labels'][i]),
                'direction': 'BUY' if directions[i] > 0 else 'SELL',
            })

        # Square-root fit line
        size_range = np.linspace(0, np.percentile(trade_sizes, 99), 50)
        sqrt_fit_line = []
        for s in size_range:
            part = np.sqrt(s / profile['adv'])
            predicted = sqrt_alpha + sqrt_beta * part
            sqrt_fit_line.append({
                'trade_size': safe_float(s),
                'predicted_bp': safe_float(max(predicted, 0)),
            })

        # Impact by regime
        regime_chart = gmm_results['components']

        # Participation rate buckets
        pct_buckets = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        bucket_chart = []
        for j in range(len(pct_buckets) - 1):
            lo, hi = pct_buckets[j], pct_buckets[j + 1]
            mask = (participation >= lo) & (participation < hi)
            if mask.sum() > 0:
                bucket_chart.append({
                    'bucket': f'{lo*100:.1f}-{hi*100:.1f}%',
                    'avg_impact_bp': safe_float(abs_impact[mask].mean()),
                    'median_impact_bp': safe_float(np.median(abs_impact[mask])),
                    'count': int(mask.sum()),
                    'avg_size': safe_float(trade_sizes[mask].mean()),
                })

        # Execution schedule
        schedule_chart = ac['schedule']

        # Strategy comparison
        strategy_comparison = [
            {'strategy': 'Optimal (AC)', 'cost_bp': ac['total_cost_bp']},
            {'strategy': 'TWAP', 'cost_bp': ac['twap_cost_bp']},
            {'strategy': 'Immediate', 'cost_bp': ac['immediate_cost_bp']},
        ]

        # Impact distribution
        impact_bins = np.linspace(0, np.percentile(abs_impact, 98), 35)
        impact_hist = []
        for j in range(len(impact_bins) - 1):
            lo, hi = impact_bins[j], impact_bins[j + 1]
            count = int(((abs_impact >= lo) & (abs_impact < hi)).sum())
            impact_hist.append({'range': f'{(lo+hi)/2:.1f}', 'count': count})

        # Buy vs Sell asymmetry
        buy_mask = directions > 0
        sell_mask = directions < 0
        asymmetry = {
            'buy_avg_bp': safe_float(abs_impact[buy_mask].mean()) if buy_mask.sum() > 0 else 0,
            'sell_avg_bp': safe_float(abs_impact[sell_mask].mean()) if sell_mask.sum() > 0 else 0,
            'buy_count': int(buy_mask.sum()),
            'sell_count': int(sell_mask.sum()),
        }

        # ── 7. Summary ──
        results = {
            'ticker': request.ticker.upper(),
            'asset_name': profile['name'],
            'price': profile['price'],
            'adv': profile['adv'],
            'n_trades': n,
            'summary': {
                'avg_impact_bp': safe_float(abs_impact.mean()),
                'median_impact_bp': safe_float(np.median(abs_impact)),
                'p95_impact_bp': safe_float(np.percentile(abs_impact, 95)),
                'avg_trade_size': safe_float(trade_sizes.mean()),
                'median_trade_size': safe_float(np.median(trade_sizes)),
                'avg_participation_pct': safe_float(participation.mean() * 100),
            },
            'kyle': kyle,
            'sqrt_model': {
                'alpha': sqrt_alpha,
                'beta': sqrt_beta,
                'r_squared': sqrt_r2,
            },
            'gmm': {
                'n_components': gmm_results['n_components'],
                'bic': gmm_results['bic'],
                'aic': gmm_results['aic'],
                'components': gmm_results['components'],
            },
            'almgren_chriss': {
                'total_cost_bp': ac['total_cost_bp'],
                'twap_cost_bp': ac['twap_cost_bp'],
                'immediate_cost_bp': ac['immediate_cost_bp'],
                'savings_vs_twap_bp': safe_float(ac['twap_cost_bp'] - ac['total_cost_bp']),
                'savings_vs_immediate_bp': safe_float(ac['immediate_cost_bp'] - ac['total_cost_bp']),
                'total_shares': ac['total_shares'],
                'time_horizon': ac['time_horizon'],
                'kappa': ac['kappa'],
            },
            'asymmetry': asymmetry,
            'charts': {
                'scatter': scatter,
                'sqrt_fit': sqrt_fit_line,
                'regime_stats': regime_chart,
                'participation_buckets': bucket_chart,
                'execution_schedule': schedule_chart,
                'strategy_comparison': strategy_comparison,
                'impact_distribution': impact_hist,
            },
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
