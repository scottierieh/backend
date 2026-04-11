from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


class AttributionRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    generate: bool = False
    nCustomers: int = 500
    seed: Optional[int] = None
    # Column mapping
    colCustomerId: Optional[str] = None
    colTimestamp: Optional[str] = None
    colChannel: Optional[str] = None
    colConversion: Optional[str] = None
    colRevenue: Optional[str] = None
    # Config
    timeDecayHalfLife: float = 7.0  # days
    topPaths: int = 15


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

CHANNELS = ['Organic Search', 'Paid Search', 'Social Media', 'Email', 'Display Ads', 'Direct', 'Referral', 'Video Ads']

# Channel profiles: (avg touches, conversion boost, avg revenue)
CHANNEL_PROFILES = {
    'Organic Search':  {'weight': 0.20, 'conv_boost': 1.0, 'avg_rev': 85},
    'Paid Search':     {'weight': 0.18, 'conv_boost': 1.3, 'avg_rev': 110},
    'Social Media':    {'weight': 0.16, 'conv_boost': 0.7, 'avg_rev': 60},
    'Email':           {'weight': 0.14, 'conv_boost': 1.5, 'avg_rev': 95},
    'Display Ads':     {'weight': 0.10, 'conv_boost': 0.5, 'avg_rev': 45},
    'Direct':          {'weight': 0.10, 'conv_boost': 1.8, 'avg_rev': 120},
    'Referral':        {'weight': 0.07, 'conv_boost': 1.2, 'avg_rev': 90},
    'Video Ads':       {'weight': 0.05, 'conv_boost': 0.6, 'avg_rev': 55},
}


def generate_journeys(n_customers: int, seed=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base_date = pd.Timestamp('2025-01-01')
    channels = list(CHANNEL_PROFILES.keys())
    ch_weights = [CHANNEL_PROFILES[c]['weight'] for c in channels]
    ch_weights = np.array(ch_weights) / sum(ch_weights)

    for cust in range(n_customers):
        cid = f'U{cust+1:05d}'
        n_touches = rng.integers(1, 8)
        journey_channels = rng.choice(channels, size=n_touches, p=ch_weights)

        # Conversion probability based on journey
        conv_prob = 0.15
        for ch in journey_channels:
            conv_prob *= CHANNEL_PROFILES[ch]['conv_boost'] ** 0.3
        conv_prob = min(conv_prob, 0.85)
        converted = rng.random() < conv_prob

        start_day = rng.integers(0, 90)
        for t, ch in enumerate(journey_channels):
            ts = base_date + pd.Timedelta(days=start_day + t * rng.integers(1, 5))
            is_last = (t == n_touches - 1)
            is_conv = converted and is_last

            revenue = 0.0
            if is_conv:
                revenue = float(rng.uniform(20, 300))

            rows.append({
                'customer_id': cid,
                'timestamp': ts.strftime('%Y-%m-%d %H:%M'),
                'channel': ch,
                'conversion': int(is_conv),
                'revenue': round(revenue, 2),
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# Attribution Models
# ══════════════════════════════════════════════════════════════

def build_journeys(df, col_cust, col_ts, col_channel, col_conv, col_rev):
    """Group touchpoints into customer journeys."""
    df = df.sort_values([col_cust, col_ts])
    journeys = []

    for cid, grp in df.groupby(col_cust):
        touches = grp[col_channel].tolist()
        converted = grp[col_conv].max() > 0
        revenue = grp[col_rev].sum() if col_rev and col_rev in df.columns else (1.0 if converted else 0.0)
        timestamps = pd.to_datetime(grp[col_ts]).tolist() if col_ts in df.columns else None

        journeys.append({
            'customer_id': str(cid),
            'touches': touches,
            'converted': converted,
            'revenue': float(revenue),
            'timestamps': timestamps,
        })

    return journeys


def last_touch(journeys):
    credit = defaultdict(float)
    for j in journeys:
        if j['converted'] and j['touches']:
            credit[j['touches'][-1]] += j['revenue']
    return dict(credit)


def first_touch(journeys):
    credit = defaultdict(float)
    for j in journeys:
        if j['converted'] and j['touches']:
            credit[j['touches'][0]] += j['revenue']
    return dict(credit)


def linear_attr(journeys):
    credit = defaultdict(float)
    for j in journeys:
        if j['converted'] and j['touches']:
            share = j['revenue'] / len(j['touches'])
            for ch in j['touches']:
                credit[ch] += share
    return dict(credit)


def time_decay(journeys, half_life_days=7.0):
    credit = defaultdict(float)
    for j in journeys:
        if j['converted'] and j['touches'] and j['timestamps']:
            conv_time = j['timestamps'][-1]
            weights = []
            for ts in j['timestamps']:
                days_before = (conv_time - ts).total_seconds() / 86400
                w = 2 ** (-days_before / half_life_days)
                weights.append(w)

            total_w = sum(weights)
            if total_w > 0:
                for ch, w in zip(j['touches'], weights):
                    credit[ch] += j['revenue'] * w / total_w
    return dict(credit)


def position_based(journeys, first_pct=0.4, last_pct=0.4):
    """40% first, 40% last, 20% distributed among middle."""
    credit = defaultdict(float)
    mid_pct = 1.0 - first_pct - last_pct

    for j in journeys:
        if j['converted'] and j['touches']:
            n = len(j['touches'])
            rev = j['revenue']
            if n == 1:
                credit[j['touches'][0]] += rev
            elif n == 2:
                credit[j['touches'][0]] += rev * 0.5
                credit[j['touches'][1]] += rev * 0.5
            else:
                credit[j['touches'][0]] += rev * first_pct
                credit[j['touches'][-1]] += rev * last_pct
                mid_share = rev * mid_pct / (n - 2)
                for ch in j['touches'][1:-1]:
                    credit[ch] += mid_share
    return dict(credit)


def markov_chain(journeys):
    """
    Markov Chain attribution via Removal Effect.
    1. Build transition probability matrix from journeys
    2. Calculate baseline conversion probability
    3. Remove each channel and recalculate
    4. Removal effect = baseline - P(conv without channel)
    5. Normalize removal effects to distribute total revenue
    """
    # Build transitions
    transitions = defaultdict(lambda: defaultdict(int))

    for j in journeys:
        path = ['Start'] + j['touches']
        if j['converted']:
            path.append('Conversion')
        else:
            path.append('Null')

        for i in range(len(path) - 1):
            transitions[path[i]][path[i + 1]] += 1

    # Get all states
    all_channels = set()
    for j in journeys:
        all_channels.update(j['touches'])
    all_channels = sorted(all_channels)

    states = ['Start'] + all_channels + ['Conversion', 'Null']

    # Transition probability matrix
    def get_trans_probs(transitions_dict):
        probs = {}
        for state, nexts in transitions_dict.items():
            total = sum(nexts.values())
            if total > 0:
                probs[state] = {n: c / total for n, c in nexts.items()}
            else:
                probs[state] = {}
        return probs

    trans_probs = get_trans_probs(transitions)

    # Calculate conversion probability via simulation
    def calc_conv_prob(trans_probs, removed_channel=None, n_sim=5000):
        rng = np.random.default_rng(42)
        conversions = 0

        for _ in range(n_sim):
            state = 'Start'
            for step in range(20):  # max path length
                if state in ('Conversion', 'Null'):
                    break
                if state not in trans_probs or not trans_probs[state]:
                    break

                # Filter out removed channel
                nexts = trans_probs[state]
                if removed_channel and removed_channel in nexts:
                    filtered = {k: v for k, v in nexts.items() if k != removed_channel}
                    if not filtered:
                        break
                    total = sum(filtered.values())
                    filtered = {k: v / total for k, v in filtered.items()}
                    nexts = filtered

                next_states = list(nexts.keys())
                probs = list(nexts.values())
                state = rng.choice(next_states, p=probs)

            if state == 'Conversion':
                conversions += 1

        return conversions / n_sim

    # Baseline
    baseline = calc_conv_prob(trans_probs)

    # Removal effect per channel
    removal_effects = {}
    for ch in all_channels:
        prob_without = calc_conv_prob(trans_probs, removed_channel=ch)
        effect = max(0, baseline - prob_without)
        removal_effects[ch] = effect

    # Normalize and distribute revenue
    total_effect = sum(removal_effects.values())
    total_revenue = sum(j['revenue'] for j in journeys if j['converted'])

    credit = {}
    if total_effect > 0:
        for ch, effect in removal_effects.items():
            credit[ch] = total_revenue * effect / total_effect
    else:
        # Fallback to linear
        credit = linear_attr(journeys)

    return credit, removal_effects, baseline


# ══════════════════════════════════════════════════════════════
# Path Analysis
# ══════════════════════════════════════════════════════════════

def analyze_paths(journeys, top_n=15):
    """Find most common conversion paths."""
    path_counts = defaultdict(lambda: {'count': 0, 'revenue': 0.0})

    for j in journeys:
        if j['converted']:
            path_key = ' → '.join(j['touches'])
            path_counts[path_key]['count'] += 1
            path_counts[path_key]['revenue'] += j['revenue']

    sorted_paths = sorted(path_counts.items(), key=lambda x: x[1]['count'], reverse=True)[:top_n]
    return [{'path': p, 'count': d['count'], 'revenue': safe_float(d['revenue']),
             'length': len(p.split(' → '))} for p, d in sorted_paths]


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/attribution")
async def attribution_modeling(request: AttributionRequest):
    try:
        rng = np.random.default_rng(request.seed)

        # ── 1. Data ──
        if request.generate or not request.data:
            df = generate_journeys(request.nCustomers, request.seed)
            col_cust, col_ts, col_ch, col_conv, col_rev = 'customer_id', 'timestamp', 'channel', 'conversion', 'revenue'
        else:
            df = pd.DataFrame(request.data)
            col_cust = request.colCustomerId or next((c for c in df.columns if 'customer' in c.lower() or 'user' in c.lower() or 'id' in c.lower()), df.columns[0])
            col_ts = request.colTimestamp or next((c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()), None)
            col_ch = request.colChannel or next((c for c in df.columns if 'channel' in c.lower() or 'source' in c.lower() or 'medium' in c.lower() or 'touchpoint' in c.lower()), None)
            col_conv = request.colConversion or next((c for c in df.columns if 'conv' in c.lower()), None)
            col_rev = request.colRevenue or next((c for c in df.columns if 'revenue' in c.lower() or 'value' in c.lower() or 'amount' in c.lower()), None)

            if not col_ch:
                raise HTTPException(status_code=400, detail="Cannot find channel column.")
            if not col_conv:
                raise HTTPException(status_code=400, detail="Cannot find conversion column.")

        if col_conv:
            df[col_conv] = pd.to_numeric(df[col_conv], errors='coerce').fillna(0)
        if col_rev and col_rev in df.columns:
            df[col_rev] = pd.to_numeric(df[col_rev], errors='coerce').fillna(0)

        n_touchpoints = len(df)
        n_customers = df[col_cust].nunique()
        channels = sorted(df[col_ch].unique().tolist())
        n_conversions = int(df[col_conv].sum()) if col_conv else 0
        total_revenue = float(df[col_rev].sum()) if col_rev and col_rev in df.columns else float(n_conversions)

        if n_touchpoints < 10:
            raise HTTPException(status_code=400, detail=f"Need >=10 touchpoints. Got {n_touchpoints}.")

        # ── 2. Build Journeys ──
        journeys = build_journeys(df, col_cust, col_ts, col_ch, col_conv, col_rev)
        converting_journeys = [j for j in journeys if j['converted']]

        # ── 3. Run All Models ──
        lt = last_touch(journeys)
        ft = first_touch(journeys)
        lin = linear_attr(journeys)
        td = time_decay(journeys, request.timeDecayHalfLife) if col_ts else linear_attr(journeys)
        pb = position_based(journeys)
        mk, removal_effects, markov_baseline = markov_chain(journeys)

        all_models = {
            'Last Touch': lt,
            'First Touch': ft,
            'Linear': lin,
            'Time Decay': td,
            'Position Based': pb,
            'Markov Chain': mk,
        }

        # ── 4. Channel Summary Table ──
        channel_table = []
        for ch in channels:
            touch_count = int((df[col_ch] == ch).sum())
            conv_count = int(df[df[col_ch] == ch][col_conv].sum()) if col_conv else 0

            row = {
                'channel': ch,
                'touchpoints': touch_count,
                'conversions': conv_count,
                'touch_pct': safe_float(touch_count / n_touchpoints * 100),
            }
            for model_name, credits in all_models.items():
                key = model_name.lower().replace(' ', '_')
                row[f'{key}_revenue'] = safe_float(credits.get(ch, 0))
                row[f'{key}_pct'] = safe_float(credits.get(ch, 0) / total_revenue * 100) if total_revenue > 0 else 0

            if ch in removal_effects:
                row['removal_effect'] = safe_float(removal_effects[ch])

            channel_table.append(row)

        # Sort by Markov credit
        channel_table.sort(key=lambda x: x.get('markov_chain_revenue', 0), reverse=True)

        # ── 5. Model Comparison Chart ──
        model_comparison = []
        for ch in channels:
            entry = {'channel': ch}
            for model_name, credits in all_models.items():
                key = model_name.replace(' ', '_')
                entry[key] = safe_float(credits.get(ch, 0) / total_revenue * 100) if total_revenue > 0 else 0
            model_comparison.append(entry)

        # ── 6. Path Analysis ──
        top_paths = analyze_paths(journeys, request.topPaths)

        # ── 7. Journey Length Distribution ──
        lengths = [len(j['touches']) for j in converting_journeys]
        length_dist = []
        if lengths:
            for l in range(1, max(lengths) + 1):
                c = lengths.count(l)
                if c > 0:
                    length_dist.append({'length': l, 'count': c, 'pct': safe_float(c / len(lengths) * 100)})

        # ── 8. Channel Position Analysis ──
        position_chart = []
        for ch in channels:
            first_count = sum(1 for j in converting_journeys if j['touches'] and j['touches'][0] == ch)
            last_count = sum(1 for j in converting_journeys if j['touches'] and j['touches'][-1] == ch)
            mid_count = sum(1 for j in converting_journeys if ch in j['touches'][1:-1]) if len(converting_journeys) > 0 else 0
            total_ch = first_count + last_count + mid_count
            if total_ch > 0:
                position_chart.append({
                    'channel': ch,
                    'first': first_count,
                    'middle': mid_count,
                    'last': last_count,
                    'first_pct': safe_float(first_count / total_ch * 100),
                    'last_pct': safe_float(last_count / total_ch * 100),
                })

        # ── 9. Removal Effect Chart ──
        removal_chart = [{'channel': ch, 'removal_effect': safe_float(removal_effects.get(ch, 0))}
                         for ch in channels if ch in removal_effects]
        removal_chart.sort(key=lambda x: x['removal_effect'], reverse=True)

        # ── Response ──
        results = {
            'n_touchpoints': n_touchpoints,
            'n_customers': n_customers,
            'n_conversions': n_conversions,
            'total_revenue': safe_float(total_revenue),
            'conversion_rate': safe_float(n_conversions / n_customers * 100) if n_customers > 0 else 0,
            'avg_journey_length': safe_float(np.mean([len(j['touches']) for j in journeys])),
            'channels': channels,
            'markov_baseline': safe_float(markov_baseline),
            'columns_used': {
                'customer_id': col_cust, 'timestamp': col_ts,
                'channel': col_ch, 'conversion': col_conv, 'revenue': col_rev,
            },
            'channel_table': channel_table,
            'charts': {
                'model_comparison': model_comparison,
                'top_paths': top_paths,
                'journey_length': length_dist,
                'channel_position': position_chart,
                'removal_effect': removal_chart,
            },
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
