from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from collections import Counter
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


class FulfillmentRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    generate: bool = False
    nOrders: int = 2000
    seed: Optional[int] = None
    # Column mapping
    colOrderId: Optional[str] = None
    colOrderDate: Optional[str] = None
    colShipDate: Optional[str] = None
    colDeliveryDate: Optional[str] = None
    colStatus: Optional[str] = None
    colCarrier: Optional[str] = None
    colWarehouse: Optional[str] = None
    colRegion: Optional[str] = None
    colPromisedDays: Optional[str] = None
    colActualDays: Optional[str] = None
    colOrderValue: Optional[str] = None
    # Config
    slaTarget: float = 3.0  # days promised


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
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════
# Data Generation
# ══════════════════════════════════════════════════════════════

CARRIERS = ['FedEx', 'UPS', 'DHL', 'USPS', 'Amazon Logistics']
WAREHOUSES = ['East Coast DC', 'West Coast DC', 'Central DC', 'South DC']
REGIONS = ['Northeast', 'Southeast', 'Midwest', 'West', 'Southwest']
STATUSES = ['delivered', 'delivered', 'delivered', 'delivered', 'delivered',
            'delivered', 'delivered', 'delivered', 'delayed', 'returned']

CARRIER_PROFILES = {
    'FedEx':            {'avg_ship': 0.8, 'avg_transit': 2.2, 'std': 0.6, 'delay_rate': 0.08},
    'UPS':              {'avg_ship': 0.9, 'avg_transit': 2.4, 'std': 0.7, 'delay_rate': 0.10},
    'DHL':              {'avg_ship': 1.0, 'avg_transit': 3.0, 'std': 1.0, 'delay_rate': 0.15},
    'USPS':             {'avg_ship': 1.2, 'avg_transit': 3.5, 'std': 1.2, 'delay_rate': 0.18},
    'Amazon Logistics': {'avg_ship': 0.5, 'avg_transit': 1.8, 'std': 0.4, 'delay_rate': 0.05},
}

WAREHOUSE_PROFILES = {
    'East Coast DC': {'processing_add': 0.0, 'capacity_factor': 1.0},
    'West Coast DC': {'processing_add': 0.2, 'capacity_factor': 0.9},
    'Central DC':    {'processing_add': 0.1, 'capacity_factor': 0.95},
    'South DC':      {'processing_add': 0.3, 'capacity_factor': 0.85},
}


def generate_fulfillment_data(n_orders: int, seed=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base_date = pd.Timestamp('2024-01-01')

    for i in range(n_orders):
        # Random order date spread over 6 months
        order_offset = rng.integers(0, 180)
        order_date = base_date + pd.Timedelta(days=int(order_offset))

        # Carrier & warehouse
        carrier = rng.choice(CARRIERS)
        warehouse = rng.choice(WAREHOUSES)
        region = rng.choice(REGIONS)

        cp = CARRIER_PROFILES[carrier]
        wp = WAREHOUSE_PROFILES[warehouse]

        # Processing time (order → ship)
        processing_days = max(0, rng.normal(cp['avg_ship'] + wp['processing_add'], 0.3))

        # Holiday/weekend surge
        dow = order_date.dayofweek
        month = order_date.month
        if dow >= 4:  # Fri-Sun orders take longer
            processing_days += rng.uniform(0.2, 0.8)
        if month in [11, 12]:  # Holiday season
            processing_days += rng.uniform(0.3, 1.5)

        ship_date = order_date + pd.Timedelta(days=max(0, round(processing_days)))

        # Transit time (ship → deliver)
        transit_days = max(1, rng.normal(cp['avg_transit'], cp['std']))

        # Regional distance factor
        if region in ['West', 'Southwest'] and warehouse == 'East Coast DC':
            transit_days += rng.uniform(1, 2)
        elif region in ['Northeast'] and warehouse == 'West Coast DC':
            transit_days += rng.uniform(1, 2)

        delivery_date = ship_date + pd.Timedelta(days=max(1, round(transit_days)))

        # Promised days (SLA)
        promised_days = rng.choice([2, 3, 3, 3, 5, 5, 7])
        actual_days = (delivery_date - order_date).days

        # Status
        if actual_days > promised_days + 2:
            status = 'delayed'
        elif rng.random() < 0.03:
            status = 'returned'
        else:
            status = 'delivered'

        # If delayed, extend delivery
        if rng.random() < cp['delay_rate']:
            extra = rng.integers(1, 4)
            delivery_date += pd.Timedelta(days=int(extra))
            actual_days += extra
            if actual_days > promised_days:
                status = 'delayed'

        order_value = round(rng.lognormal(3.5, 0.8), 2)

        rows.append({
            'order_id': f'ORD-{i+1:06d}',
            'order_date': order_date.strftime('%Y-%m-%d'),
            'ship_date': ship_date.strftime('%Y-%m-%d'),
            'delivery_date': delivery_date.strftime('%Y-%m-%d'),
            'status': status,
            'carrier': carrier,
            'warehouse': warehouse,
            'region': region,
            'promised_days': int(promised_days),
            'actual_days': int(actual_days),
            'order_value': order_value,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# Analysis Functions
# ══════════════════════════════════════════════════════════════

def calc_lead_times(df, col_order, col_ship, col_deliver):
    """Calculate processing and transit times."""
    df['_order_dt'] = pd.to_datetime(df[col_order], errors='coerce')
    df['_ship_dt'] = pd.to_datetime(df[col_ship], errors='coerce')
    df['_deliver_dt'] = pd.to_datetime(df[col_deliver], errors='coerce')

    df['_processing_days'] = (df['_ship_dt'] - df['_order_dt']).dt.total_seconds() / 86400
    df['_transit_days'] = (df['_deliver_dt'] - df['_ship_dt']).dt.total_seconds() / 86400
    df['_total_days'] = (df['_deliver_dt'] - df['_order_dt']).dt.total_seconds() / 86400

    # Clean negatives
    for c in ['_processing_days', '_transit_days', '_total_days']:
        df[c] = df[c].clip(lower=0)

    return df


def performance_by_group(df, group_col, col_promised, col_actual, col_value=None):
    """Compute performance metrics by grouping column."""
    results = []
    for grp, gdf in df.groupby(group_col):
        n = len(gdf)
        avg_processing = safe_float(gdf['_processing_days'].mean())
        avg_transit = safe_float(gdf['_transit_days'].mean())
        avg_total = safe_float(gdf['_total_days'].mean())
        p95_total = safe_float(gdf['_total_days'].quantile(0.95))

        if col_promised and col_promised in gdf.columns:
            on_time = (gdf[col_actual] <= gdf[col_promised]).sum()
            sla_pct = safe_float(on_time / n * 100) if n > 0 else 0
        else:
            sla_pct = 0

        delayed = (gdf.get('status', pd.Series(['delivered'] * n)) == 'delayed').sum() if 'status' in gdf.columns else 0
        returned = (gdf.get('status', pd.Series(['delivered'] * n)) == 'returned').sum() if 'status' in gdf.columns else 0

        row = {
            'group': str(grp),
            'orders': n,
            'avg_processing': avg_processing,
            'avg_transit': avg_transit,
            'avg_total': avg_total,
            'p95_total': p95_total,
            'sla_pct': sla_pct,
            'delayed_count': int(delayed),
            'delayed_pct': safe_float(delayed / n * 100) if n > 0 else 0,
            'returned_count': int(returned),
            'returned_pct': safe_float(returned / n * 100) if n > 0 else 0,
        }

        if col_value and col_value in gdf.columns:
            row['total_value'] = safe_float(gdf[col_value].sum())
            row['avg_value'] = safe_float(gdf[col_value].mean())

        results.append(row)

    return sorted(results, key=lambda x: x['avg_total'])


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/fulfillment-delivery")
async def fulfillment_delivery(request: FulfillmentRequest):
    try:
        # ── 1. Data ──
        if request.generate or not request.data:
            df = generate_fulfillment_data(request.nOrders, request.seed)
            col_oid = 'order_id'
            col_odate = 'order_date'
            col_sdate = 'ship_date'
            col_ddate = 'delivery_date'
            col_status = 'status'
            col_carrier = 'carrier'
            col_warehouse = 'warehouse'
            col_region = 'region'
            col_promised = 'promised_days'
            col_actual = 'actual_days'
            col_value = 'order_value'
        else:
            df = pd.DataFrame(request.data)
            lc = {c: c.lower() for c in df.columns}
            rlc = {v: k for k, v in lc.items()}

            def find_col(candidates, override=None):
                if override and override in df.columns:
                    return override
                for c in candidates:
                    if c in rlc:
                        return rlc[c]
                    for col_lc, col_orig in zip(lc.values(), lc.keys()):
                        if c in col_lc:
                            return col_orig
                return None

            col_oid = find_col(['order_id', 'id', 'order_number', 'order_no'], request.colOrderId)
            col_odate = find_col(['order_date', 'ordered_at', 'created_at', 'order_time'], request.colOrderDate)
            col_sdate = find_col(['ship_date', 'shipped_at', 'shipped_date', 'dispatch_date', 'shipment_date'], request.colShipDate)
            col_ddate = find_col(['delivery_date', 'delivered_at', 'delivered_date', 'received_date', 'arrival_date'], request.colDeliveryDate)
            col_status = find_col(['status', 'order_status', 'delivery_status'], request.colStatus)
            col_carrier = find_col(['carrier', 'shipping_carrier', 'courier', 'logistics_provider'], request.colCarrier)
            col_warehouse = find_col(['warehouse', 'fulfillment_center', 'dc', 'distribution_center', 'origin'], request.colWarehouse)
            col_region = find_col(['region', 'destination_region', 'zone', 'area', 'state', 'country'], request.colRegion)
            col_promised = find_col(['promised_days', 'sla_days', 'promised', 'target_days', 'eta_days'], request.colPromisedDays)
            col_actual = find_col(['actual_days', 'delivery_days', 'lead_time', 'total_days'], request.colActualDays)
            col_value = find_col(['order_value', 'total', 'amount', 'revenue', 'value', 'order_total'], request.colOrderValue)

            if not col_odate:
                raise HTTPException(status_code=400, detail=f"Cannot find order_date column. Columns: {list(df.columns)}")

        n_orders = len(df)
        if n_orders < 10:
            raise HTTPException(status_code=400, detail=f"Need >= 10 orders. Got {n_orders}.")

        # ── 2. Compute lead times ──
        has_dates = col_odate and col_sdate and col_ddate
        if has_dates:
            df = calc_lead_times(df, col_odate, col_sdate, col_ddate)

        # If actual_days not in data, derive from dates
        if col_actual and col_actual in df.columns:
            df[col_actual] = pd.to_numeric(df[col_actual], errors='coerce').fillna(0)
        elif has_dates:
            col_actual = '_total_days'

        if col_promised and col_promised in df.columns:
            df[col_promised] = pd.to_numeric(df[col_promised], errors='coerce').fillna(request.slaTarget)
        elif has_dates:
            df['_promised'] = request.slaTarget
            col_promised = '_promised'

        if col_value and col_value in df.columns:
            df[col_value] = pd.to_numeric(df[col_value], errors='coerce').fillna(0)

        # ── 3. Overall Metrics ──
        total_orders = n_orders
        status_counts = {}
        if col_status and col_status in df.columns:
            status_counts = df[col_status].value_counts().to_dict()

        delivered_count = int(status_counts.get('delivered', 0))
        delayed_count = int(status_counts.get('delayed', 0))
        returned_count = int(status_counts.get('returned', 0))
        other_count = total_orders - delivered_count - delayed_count - returned_count

        # SLA compliance
        sla_on_time = 0
        if col_actual and col_promised:
            mask = df[col_actual] <= df[col_promised]
            sla_on_time = int(mask.sum())
        sla_pct = safe_float(sla_on_time / total_orders * 100) if total_orders > 0 else 0

        # Lead time stats
        avg_processing = safe_float(df['_processing_days'].mean()) if '_processing_days' in df.columns else 0
        avg_transit = safe_float(df['_transit_days'].mean()) if '_transit_days' in df.columns else 0
        avg_total = safe_float(df['_total_days'].mean()) if '_total_days' in df.columns else safe_float(df[col_actual].mean()) if col_actual else 0
        median_total = safe_float(df['_total_days'].median()) if '_total_days' in df.columns else 0
        p95_total = safe_float(df['_total_days'].quantile(0.95)) if '_total_days' in df.columns else 0
        std_total = safe_float(df['_total_days'].std()) if '_total_days' in df.columns else 0

        total_value = safe_float(df[col_value].sum()) if col_value and col_value in df.columns else 0
        avg_value = safe_float(df[col_value].mean()) if col_value and col_value in df.columns else 0

        # ── 4. Performance by Carrier ──
        carrier_perf = []
        if col_carrier and col_carrier in df.columns:
            carrier_perf = performance_by_group(df, col_carrier, col_promised, col_actual, col_value)

        # ── 5. Performance by Warehouse ──
        warehouse_perf = []
        if col_warehouse and col_warehouse in df.columns:
            warehouse_perf = performance_by_group(df, col_warehouse, col_promised, col_actual, col_value)

        # ── 6. Performance by Region ──
        region_perf = []
        if col_region and col_region in df.columns:
            region_perf = performance_by_group(df, col_region, col_promised, col_actual, col_value)

        # ── 7. Lead Time Distribution ──
        lt_dist = []
        if '_total_days' in df.columns:
            bins = range(0, int(df['_total_days'].max()) + 2)
            for b in bins:
                count = int(((df['_total_days'] >= b) & (df['_total_days'] < b + 1)).sum())
                if count > 0:
                    lt_dist.append({'days': b, 'count': count, 'pct': safe_float(count / total_orders * 100)})

        # ── 8. Bottleneck Analysis ──
        bottleneck = []
        if '_processing_days' in df.columns and '_transit_days' in df.columns:
            bottleneck = [
                {'stage': 'Processing (Order→Ship)', 'avg_days': avg_processing, 'pct_of_total': safe_float(avg_processing / avg_total * 100) if avg_total > 0 else 0},
                {'stage': 'Transit (Ship→Deliver)', 'avg_days': avg_transit, 'pct_of_total': safe_float(avg_transit / avg_total * 100) if avg_total > 0 else 0},
            ]

        # ── 9. Trend Over Time ──
        trend = []
        if col_odate:
            df['_order_dt'] = pd.to_datetime(df[col_odate], errors='coerce')
            df['_week'] = df['_order_dt'].dt.to_period('W').astype(str)
            for wk, wdf in df.groupby('_week'):
                if len(wdf) < 2:
                    continue
                row = {
                    'period': str(wk),
                    'orders': len(wdf),
                    'avg_total_days': safe_float(wdf['_total_days'].mean()) if '_total_days' in wdf.columns else 0,
                    'sla_pct': 0,
                    'delayed_pct': 0,
                }
                if col_actual and col_promised:
                    on_time = (wdf[col_actual] <= wdf[col_promised]).sum()
                    row['sla_pct'] = safe_float(on_time / len(wdf) * 100)
                if col_status and col_status in wdf.columns:
                    row['delayed_pct'] = safe_float((wdf[col_status] == 'delayed').sum() / len(wdf) * 100)
                trend.append(row)

        # ── 10. Day-of-Week Pattern ──
        dow_pattern = []
        if '_order_dt' in df.columns and '_total_days' in df.columns:
            df['_dow'] = df['_order_dt'].dt.dayofweek
            dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            for d in range(7):
                ddf = df[df['_dow'] == d]
                if len(ddf) > 0:
                    dow_pattern.append({
                        'day': dow_names[d],
                        'day_num': d,
                        'orders': len(ddf),
                        'avg_total_days': safe_float(ddf['_total_days'].mean()),
                        'avg_processing': safe_float(ddf['_processing_days'].mean()) if '_processing_days' in ddf.columns else 0,
                    })

        # ── 11. Status Distribution ──
        status_chart = []
        if col_status and col_status in df.columns:
            for s, c in df[col_status].value_counts().items():
                status_chart.append({
                    'status': str(s),
                    'count': int(c),
                    'pct': safe_float(c / total_orders * 100),
                })

        # ── 12. SLA Band Analysis ──
        sla_bands = []
        if col_actual and col_promised:
            df['_sla_diff'] = df[col_actual] - df[col_promised]
            bands = [
                ('Early (2+ days)', df['_sla_diff'] <= -2),
                ('Early (1 day)', (df['_sla_diff'] > -2) & (df['_sla_diff'] <= -1)),
                ('On Time', (df['_sla_diff'] > -1) & (df['_sla_diff'] <= 0)),
                ('Late (1 day)', (df['_sla_diff'] > 0) & (df['_sla_diff'] <= 1)),
                ('Late (2+ days)', df['_sla_diff'] > 1),
            ]
            for label, mask in bands:
                c = int(mask.sum())
                if c > 0:
                    sla_bands.append({
                        'band': label,
                        'count': c,
                        'pct': safe_float(c / total_orders * 100),
                        'color': '#10B981' if 'Early' in label or 'On Time' in label else '#EF4444',
                    })

        # ── Columns used ──
        columns_used = {
            'order_id': col_oid, 'order_date': col_odate, 'ship_date': col_sdate,
            'delivery_date': col_ddate, 'status': col_status, 'carrier': col_carrier,
            'warehouse': col_warehouse, 'region': col_region, 'promised_days': col_promised,
            'actual_days': col_actual, 'order_value': col_value,
        }

        # ── Response ──
        results = {
            'n_orders': total_orders,
            'columns_used': columns_used,
            'summary': {
                'total_orders': total_orders,
                'delivered': delivered_count,
                'delayed': delayed_count,
                'returned': returned_count,
                'sla_on_time': sla_on_time,
                'sla_pct': sla_pct,
                'avg_processing_days': avg_processing,
                'avg_transit_days': avg_transit,
                'avg_total_days': avg_total,
                'median_total_days': median_total,
                'p95_total_days': p95_total,
                'std_total_days': std_total,
                'total_value': total_value,
                'avg_value': avg_value,
            },
            'carrier_performance': carrier_perf,
            'warehouse_performance': warehouse_perf,
            'region_performance': region_perf,
            'bottleneck': bottleneck,
            'charts': {
                'lead_time_dist': lt_dist,
                'status': status_chart,
                'sla_bands': sla_bands,
                'trend': trend,
                'dow_pattern': dow_pattern,
            },
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
