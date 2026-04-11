from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize, minimize_scalar
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


class InventoryRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    generate: bool = False
    nSKUs: int = 30
    nPeriods: int = 52
    seed: Optional[int] = None
    # Column mapping
    colSKU: Optional[str] = None
    colDemand: Optional[str] = None
    colDate: Optional[str] = None
    colLeadTime: Optional[str] = None
    colUnitCost: Optional[str] = None
    colOrderCost: Optional[str] = None
    colHoldingCostPct: Optional[str] = None
    colPrice: Optional[str] = None
    # Global defaults (used if columns not provided)
    defaultLeadTime: float = 2.0       # weeks
    defaultOrderCost: float = 50.0     # $ per order
    defaultHoldingPct: float = 0.25    # 25% of unit cost per year
    targetServiceLevel: float = 0.95   # 95%
    reviewPeriod: float = 1.0          # weeks between reviews
    periodsPerYear: int = 52           # 52 for weekly, 12 for monthly


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

SKU_CATALOG = [
    {'sku': 'SKU-001', 'name': 'Widget A', 'category': 'Components', 'unit_cost': 12.50, 'price': 25.00, 'base_demand': 80, 'cv': 0.25, 'lead_time': 2},
    {'sku': 'SKU-002', 'name': 'Widget B', 'category': 'Components', 'unit_cost': 8.00, 'price': 18.00, 'base_demand': 120, 'cv': 0.30, 'lead_time': 1},
    {'sku': 'SKU-003', 'name': 'Gadget X', 'category': 'Assemblies', 'unit_cost': 45.00, 'price': 95.00, 'base_demand': 25, 'cv': 0.40, 'lead_time': 3},
    {'sku': 'SKU-004', 'name': 'Gadget Y', 'category': 'Assemblies', 'unit_cost': 30.00, 'price': 65.00, 'base_demand': 40, 'cv': 0.35, 'lead_time': 2},
    {'sku': 'SKU-005', 'name': 'Fastener M5', 'category': 'Hardware', 'unit_cost': 0.50, 'price': 1.20, 'base_demand': 500, 'cv': 0.20, 'lead_time': 1},
    {'sku': 'SKU-006', 'name': 'Fastener M8', 'category': 'Hardware', 'unit_cost': 0.80, 'price': 1.80, 'base_demand': 350, 'cv': 0.22, 'lead_time': 1},
    {'sku': 'SKU-007', 'name': 'Sensor Alpha', 'category': 'Electronics', 'unit_cost': 85.00, 'price': 180.00, 'base_demand': 10, 'cv': 0.50, 'lead_time': 4},
    {'sku': 'SKU-008', 'name': 'Sensor Beta', 'category': 'Electronics', 'unit_cost': 55.00, 'price': 120.00, 'base_demand': 18, 'cv': 0.45, 'lead_time': 3},
    {'sku': 'SKU-009', 'name': 'Cable 1m', 'category': 'Accessories', 'unit_cost': 3.00, 'price': 8.00, 'base_demand': 200, 'cv': 0.18, 'lead_time': 1},
    {'sku': 'SKU-010', 'name': 'Cable 3m', 'category': 'Accessories', 'unit_cost': 5.00, 'price': 12.00, 'base_demand': 90, 'cv': 0.20, 'lead_time': 1},
    {'sku': 'SKU-011', 'name': 'Housing Unit', 'category': 'Assemblies', 'unit_cost': 65.00, 'price': 140.00, 'base_demand': 15, 'cv': 0.55, 'lead_time': 4},
    {'sku': 'SKU-012', 'name': 'Power Supply', 'category': 'Electronics', 'unit_cost': 35.00, 'price': 75.00, 'base_demand': 30, 'cv': 0.30, 'lead_time': 2},
    {'sku': 'SKU-013', 'name': 'Rubber Seal', 'category': 'Hardware', 'unit_cost': 1.20, 'price': 3.00, 'base_demand': 250, 'cv': 0.15, 'lead_time': 1},
    {'sku': 'SKU-014', 'name': 'Filter Cartridge', 'category': 'Consumables', 'unit_cost': 7.00, 'price': 15.00, 'base_demand': 60, 'cv': 0.25, 'lead_time': 2},
    {'sku': 'SKU-015', 'name': 'Lubricant 500ml', 'category': 'Consumables', 'unit_cost': 10.00, 'price': 22.00, 'base_demand': 45, 'cv': 0.20, 'lead_time': 1},
    {'sku': 'SKU-016', 'name': 'Motor Unit', 'category': 'Assemblies', 'unit_cost': 120.00, 'price': 250.00, 'base_demand': 8, 'cv': 0.60, 'lead_time': 5},
    {'sku': 'SKU-017', 'name': 'Display Panel', 'category': 'Electronics', 'unit_cost': 95.00, 'price': 200.00, 'base_demand': 12, 'cv': 0.50, 'lead_time': 4},
    {'sku': 'SKU-018', 'name': 'Bracket Steel', 'category': 'Hardware', 'unit_cost': 2.50, 'price': 6.00, 'base_demand': 180, 'cv': 0.18, 'lead_time': 1},
    {'sku': 'SKU-019', 'name': 'Adhesive Tube', 'category': 'Consumables', 'unit_cost': 4.00, 'price': 9.00, 'base_demand': 70, 'cv': 0.22, 'lead_time': 1},
    {'sku': 'SKU-020', 'name': 'PCB Board', 'category': 'Electronics', 'unit_cost': 25.00, 'price': 55.00, 'base_demand': 22, 'cv': 0.35, 'lead_time': 3},
    {'sku': 'SKU-021', 'name': 'Bearing 6201', 'category': 'Hardware', 'unit_cost': 6.00, 'price': 14.00, 'base_demand': 100, 'cv': 0.20, 'lead_time': 2},
    {'sku': 'SKU-022', 'name': 'O-Ring Set', 'category': 'Hardware', 'unit_cost': 1.50, 'price': 4.00, 'base_demand': 150, 'cv': 0.15, 'lead_time': 1},
    {'sku': 'SKU-023', 'name': 'Valve Assembly', 'category': 'Assemblies', 'unit_cost': 50.00, 'price': 110.00, 'base_demand': 20, 'cv': 0.40, 'lead_time': 3},
    {'sku': 'SKU-024', 'name': 'Thermocouple', 'category': 'Electronics', 'unit_cost': 18.00, 'price': 40.00, 'base_demand': 35, 'cv': 0.30, 'lead_time': 2},
    {'sku': 'SKU-025', 'name': 'Cleaning Solvent', 'category': 'Consumables', 'unit_cost': 15.00, 'price': 30.00, 'base_demand': 25, 'cv': 0.25, 'lead_time': 1},
    {'sku': 'SKU-026', 'name': 'Relay Module', 'category': 'Electronics', 'unit_cost': 12.00, 'price': 28.00, 'base_demand': 50, 'cv': 0.28, 'lead_time': 2},
    {'sku': 'SKU-027', 'name': 'Hose 2m', 'category': 'Accessories', 'unit_cost': 8.00, 'price': 18.00, 'base_demand': 55, 'cv': 0.22, 'lead_time': 1},
    {'sku': 'SKU-028', 'name': 'Pump Mini', 'category': 'Assemblies', 'unit_cost': 75.00, 'price': 160.00, 'base_demand': 10, 'cv': 0.55, 'lead_time': 4},
    {'sku': 'SKU-029', 'name': 'Label Roll', 'category': 'Consumables', 'unit_cost': 3.00, 'price': 7.00, 'base_demand': 80, 'cv': 0.15, 'lead_time': 1},
    {'sku': 'SKU-030', 'name': 'Spring Set', 'category': 'Hardware', 'unit_cost': 2.00, 'price': 5.00, 'base_demand': 130, 'cv': 0.18, 'lead_time': 1},
]


def generate_inventory_data(n_skus: int, n_periods: int, seed=None) -> pd.DataFrame:
    """Generate weekly demand history for multiple SKUs."""
    rng = np.random.default_rng(seed)
    catalog = SKU_CATALOG[:min(n_skus, len(SKU_CATALOG))]
    rows = []
    base_date = pd.Timestamp('2023-01-02')

    for item in catalog:
        for w in range(n_periods):
            date = base_date + pd.Timedelta(weeks=w)
            # Demand with trend + seasonality + noise
            trend = 1.0 + 0.001 * w  # slight upward trend
            seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * w / 52)
            base = item['base_demand'] * trend * seasonal
            demand = max(0, int(rng.normal(base, base * item['cv'])))

            rows.append({
                'date': date.strftime('%Y-%m-%d'),
                'sku': item['sku'],
                'sku_name': item['name'],
                'category': item['category'],
                'demand': demand,
                'unit_cost': item['unit_cost'],
                'price': item['price'],
                'lead_time_weeks': item['lead_time'],
                'order_cost': 50.0,
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# Inventory Calculations
# ══════════════════════════════════════════════════════════════

def calc_eoq(annual_demand: float, order_cost: float, holding_cost_per_unit: float) -> float:
    """Economic Order Quantity."""
    if annual_demand <= 0 or order_cost <= 0 or holding_cost_per_unit <= 0:
        return 0.0
    return float(np.sqrt(2 * annual_demand * order_cost / holding_cost_per_unit))


def calc_rop(demand_during_lt: float, safety_stock: float) -> float:
    """Reorder Point = expected demand during lead time + safety stock."""
    return demand_during_lt + safety_stock


def calc_safety_stock(demand_std_per_period: float, lead_time_periods: float, service_level: float) -> float:
    """Safety stock = z * sigma_d * sqrt(LT)."""
    z = sp_stats.norm.ppf(service_level)
    return float(z * demand_std_per_period * np.sqrt(lead_time_periods))


def calc_total_cost(annual_demand: float, order_qty: float, unit_cost: float,
                    order_cost: float, holding_pct: float, safety_stock: float) -> dict:
    """Total annual inventory cost breakdown."""
    if order_qty <= 0:
        return {'purchasing': 0, 'ordering': 0, 'holding': 0, 'total': 0}

    purchasing = annual_demand * unit_cost
    n_orders = annual_demand / order_qty
    ordering = n_orders * order_cost
    avg_inventory = order_qty / 2 + safety_stock
    holding = avg_inventory * unit_cost * holding_pct
    total = purchasing + ordering + holding

    return {
        'purchasing': safe_float(purchasing),
        'ordering': safe_float(ordering),
        'holding': safe_float(holding),
        'total': safe_float(total),
        'n_orders_per_year': safe_float(n_orders),
        'avg_inventory': safe_float(avg_inventory),
        'inventory_turns': safe_float(annual_demand / avg_inventory) if avg_inventory > 0 else 0,
    }


def classify_abc(skus: list, key='annual_revenue') -> dict:
    """ABC classification by cumulative value."""
    sorted_skus = sorted(skus, key=lambda x: x.get(key, 0), reverse=True)
    total = sum(s.get(key, 0) for s in sorted_skus)
    if total <= 0:
        return {s.get('sku', ''): 'C' for s in sorted_skus}

    result = {}
    cumsum = 0
    for s in sorted_skus:
        cumsum += s.get(key, 0)
        pct = cumsum / total
        if pct <= 0.80:
            result[s.get('sku', '')] = 'A'
        elif pct <= 0.95:
            result[s.get('sku', '')] = 'B'
        else:
            result[s.get('sku', '')] = 'C'
    return result


def classify_xyz(skus: list) -> dict:
    """XYZ classification by demand variability (CV)."""
    result = {}
    for s in skus:
        cv = s.get('cv', 0)
        if cv <= 0.25:
            result[s.get('sku', '')] = 'X'
        elif cv <= 0.50:
            result[s.get('sku', '')] = 'Y'
        else:
            result[s.get('sku', '')] = 'Z'
    return result


def optimize_service_levels(skus_data: list, total_budget: float, holding_pct: float) -> list:
    """
    Optimize service levels across SKUs to minimize total stockout cost
    subject to total safety stock investment budget.
    Uses marginal analysis: allocate safety stock $ where it reduces stockout cost most.
    """
    results = []
    for s in skus_data:
        demand_std = s.get('demand_std', 0)
        lt = s.get('lead_time', 1)
        unit_cost = s.get('unit_cost', 1)
        annual_demand = s.get('annual_demand', 0)
        price = s.get('price', unit_cost * 2)
        stockout_cost_per_unit = price - unit_cost  # margin as proxy for stockout cost

        # Try service levels from 0.80 to 0.99
        best_sl = 0.90
        for sl in np.arange(0.80, 0.995, 0.01):
            z = sp_stats.norm.ppf(sl)
            ss = z * demand_std * np.sqrt(lt)
            ss_cost = ss * unit_cost * holding_pct
            expected_shortage = demand_std * np.sqrt(lt) * sp_stats.norm.pdf(z) * annual_demand / (demand_std * np.sqrt(lt) + 1e-9)
            stockout_penalty = (1 - sl) * annual_demand * stockout_cost_per_unit * 0.1  # simplified
            total = ss_cost + stockout_penalty
            if sl == 0.80 or total < best_total:
                best_total = total
                best_sl = sl

        results.append({
            'sku': s.get('sku', ''),
            'optimal_service_level': safe_float(best_sl),
        })
    return results


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/inventory-optimization")
async def inventory_optimization(request: InventoryRequest):
    try:
        # ── 1. Data ──
        if request.generate or not request.data:
            df = generate_inventory_data(request.nSKUs, request.nPeriods, request.seed)
            col_sku = 'sku'
            col_demand = 'demand'
            col_date = 'date'
            col_lt = 'lead_time_weeks'
            col_unit_cost = 'unit_cost'
            col_order_cost = 'order_cost'
            col_price = 'price'
            col_holding_pct = None
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

            col_sku = find_col(['sku', 'item', 'product', 'sku_id', 'product_id', 'item_id', 'part'], request.colSKU)
            col_demand = find_col(['demand', 'quantity', 'qty', 'sales', 'units', 'consumption', 'usage'], request.colDemand)
            col_date = find_col(['date', 'week', 'period', 'month'], request.colDate)
            col_lt = find_col(['lead_time', 'lead_time_weeks', 'lt', 'leadtime', 'delivery_time'], request.colLeadTime)
            col_unit_cost = find_col(['unit_cost', 'cost', 'cogs', 'cost_per_unit', 'unit_price'], request.colUnitCost)
            col_order_cost = find_col(['order_cost', 'ordering_cost', 'setup_cost', 'fixed_cost'], request.colOrderCost)
            col_price = find_col(['price', 'selling_price', 'revenue_per_unit', 'sell_price'], request.colPrice)
            col_holding_pct = find_col(['holding_cost_pct', 'holding_pct', 'carrying_cost_pct'], request.colHoldingCostPct)

            if not col_sku or not col_demand:
                raise HTTPException(status_code=400,
                    detail=f"Cannot find SKU or demand column. Found: sku={col_sku}, demand={col_demand}. Columns: {list(df.columns)}")

        # Parse demand as numeric
        df[col_demand] = pd.to_numeric(df[col_demand], errors='coerce').fillna(0)

        periods_per_year = request.periodsPerYear
        holding_pct = request.defaultHoldingPct
        default_order_cost = request.defaultOrderCost
        default_lt = request.defaultLeadTime
        service_level = request.targetServiceLevel

        skus = sorted(df[col_sku].unique().tolist())
        n_skus = len(skus)

        if n_skus < 1:
            raise HTTPException(status_code=400, detail="No SKUs found in data.")

        # ── 2. Per-SKU Analysis ──
        sku_results = []
        for sku in skus:
            sdf = df[df[col_sku] == sku].copy()
            demands = sdf[col_demand].values

            # Basic stats
            mean_demand = float(np.mean(demands))
            std_demand = float(np.std(demands, ddof=1)) if len(demands) > 1 else mean_demand * 0.3
            cv = std_demand / mean_demand if mean_demand > 0 else 0
            total_demand = float(np.sum(demands))
            n_periods = len(demands)
            annual_demand = mean_demand * periods_per_year

            # Get per-SKU params or use defaults
            lead_time = float(sdf[col_lt].iloc[0]) if col_lt and col_lt in sdf.columns else default_lt
            unit_cost = float(sdf[col_unit_cost].iloc[0]) if col_unit_cost and col_unit_cost in sdf.columns else 10.0
            order_cost = float(sdf[col_order_cost].iloc[0]) if col_order_cost and col_order_cost in sdf.columns else default_order_cost
            price = float(sdf[col_price].iloc[0]) if col_price and col_price in sdf.columns else unit_cost * 2.0
            h_pct = float(sdf[col_holding_pct].iloc[0]) if col_holding_pct and col_holding_pct in sdf.columns else holding_pct

            holding_cost_per_unit = unit_cost * h_pct
            annual_revenue = annual_demand * price

            # Category (if available)
            category = ''
            cat_candidates = ['category', 'product_type', 'type', 'group']
            for cc in cat_candidates:
                if cc in sdf.columns:
                    category = str(sdf[cc].iloc[0])
                    break

            # SKU name
            sku_name = sku
            name_candidates = ['sku_name', 'name', 'product_name', 'item_name', 'description']
            for nc in name_candidates:
                if nc in sdf.columns:
                    sku_name = str(sdf[nc].iloc[0])
                    break

            # EOQ
            eoq = calc_eoq(annual_demand, order_cost, holding_cost_per_unit)

            # Safety Stock
            safety_stock = calc_safety_stock(std_demand, lead_time, service_level)

            # Reorder Point
            demand_during_lt = mean_demand * lead_time
            rop = calc_rop(demand_during_lt, safety_stock)

            # Total Cost
            costs = calc_total_cost(annual_demand, eoq, unit_cost, order_cost, h_pct, safety_stock)

            # Cost with naive policy (order monthly, no safety stock)
            naive_qty = annual_demand / 12
            costs_naive = calc_total_cost(annual_demand, max(naive_qty, 1), unit_cost, order_cost, h_pct, 0)

            savings = costs_naive['total'] - costs['total']
            savings_pct = (savings / costs_naive['total'] * 100) if costs_naive['total'] > 0 else 0

            # Demand timeline for this SKU
            demand_timeline = demands.tolist()

            sku_results.append({
                'sku': sku,
                'sku_name': sku_name,
                'category': category,
                'n_periods': n_periods,
                'mean_demand': safe_float(mean_demand),
                'std_demand': safe_float(std_demand),
                'cv': safe_float(cv),
                'total_demand': safe_float(total_demand),
                'annual_demand': safe_float(annual_demand),
                'annual_revenue': safe_float(annual_revenue),
                'unit_cost': safe_float(unit_cost),
                'price': safe_float(price),
                'order_cost': safe_float(order_cost),
                'lead_time': safe_float(lead_time),
                'holding_pct': safe_float(h_pct),
                # Optimization results
                'eoq': safe_float(eoq),
                'safety_stock': safe_float(safety_stock),
                'rop': safe_float(rop),
                'service_level': safe_float(service_level),
                'demand_during_lt': safe_float(demand_during_lt),
                # Costs
                'cost_purchasing': costs['purchasing'],
                'cost_ordering': costs['ordering'],
                'cost_holding': costs['holding'],
                'cost_total': costs['total'],
                'n_orders_per_year': costs['n_orders_per_year'],
                'avg_inventory': costs['avg_inventory'],
                'inventory_turns': costs['inventory_turns'],
                'cost_naive_total': costs_naive['total'],
                'savings': safe_float(savings),
                'savings_pct': safe_float(savings_pct),
                'demand_timeline': demand_timeline,
            })

        # ── 3. ABC / XYZ Classification ──
        abc_map = classify_abc(sku_results, key='annual_revenue')
        xyz_map = classify_xyz(sku_results)

        for s in sku_results:
            s['abc_class'] = abc_map.get(s['sku'], 'C')
            s['xyz_class'] = xyz_map.get(s['sku'], 'Y')
            s['abc_xyz'] = s['abc_class'] + s['xyz_class']

        # ── 4. Service Level Sensitivity ──
        # For the top 5 SKUs by revenue, show cost at different service levels
        top_skus = sorted(sku_results, key=lambda x: x['annual_revenue'], reverse=True)[:5]
        service_sensitivity = []
        for sl in [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]:
            row = {'service_level': sl, 'service_level_pct': f"{sl*100:.0f}%"}
            total_ss_cost = 0
            for s in top_skus:
                ss = calc_safety_stock(s['std_demand'], s['lead_time'], sl)
                ss_cost = ss * s['unit_cost'] * s['holding_pct']
                row[s['sku']] = safe_float(ss)
                total_ss_cost += ss_cost
            row['total_safety_stock_cost'] = safe_float(total_ss_cost)
            service_sensitivity.append(row)

        # ── 5. Aggregated Stats ──
        total_annual_demand_value = sum(s['annual_demand'] * s['unit_cost'] for s in sku_results)
        total_annual_revenue = sum(s['annual_revenue'] for s in sku_results)
        total_safety_stock_value = sum(s['safety_stock'] * s['unit_cost'] for s in sku_results)
        total_eoq_cost = sum(s['cost_total'] for s in sku_results)
        total_naive_cost = sum(s['cost_naive_total'] for s in sku_results)
        total_savings = total_naive_cost - total_eoq_cost
        avg_turns = np.mean([s['inventory_turns'] for s in sku_results if s['inventory_turns'] > 0])
        avg_service_level = service_level

        # ABC summary
        abc_summary = []
        for cls in ['A', 'B', 'C']:
            items = [s for s in sku_results if s['abc_class'] == cls]
            if items:
                abc_summary.append({
                    'class': cls,
                    'count': len(items),
                    'pct_skus': safe_float(len(items) / n_skus * 100),
                    'total_revenue': safe_float(sum(s['annual_revenue'] for s in items)),
                    'pct_revenue': safe_float(sum(s['annual_revenue'] for s in items) / total_annual_revenue * 100) if total_annual_revenue > 0 else 0,
                    'avg_cv': safe_float(np.mean([s['cv'] for s in items])),
                })

        # XYZ summary
        xyz_summary = []
        for cls in ['X', 'Y', 'Z']:
            items = [s for s in sku_results if s['xyz_class'] == cls]
            if items:
                xyz_summary.append({
                    'class': cls,
                    'count': len(items),
                    'pct_skus': safe_float(len(items) / n_skus * 100),
                    'avg_cv': safe_float(np.mean([s['cv'] for s in items])),
                    'total_revenue': safe_float(sum(s['annual_revenue'] for s in items)),
                })

        # ABC-XYZ matrix
        abc_xyz_matrix = []
        for a in ['A', 'B', 'C']:
            for x in ['X', 'Y', 'Z']:
                items = [s for s in sku_results if s['abc_class'] == a and s['xyz_class'] == x]
                if items:
                    abc_xyz_matrix.append({
                        'class': f"{a}{x}",
                        'abc': a,
                        'xyz': x,
                        'count': len(items),
                        'total_revenue': safe_float(sum(s['annual_revenue'] for s in items)),
                        'avg_cv': safe_float(np.mean([s['cv'] for s in items])),
                        'policy': _get_policy_recommendation(a, x),
                    })

        # ── 6. Charts ──
        # EOQ vs Current bar chart
        eoq_comparison = [{
            'sku': s['sku'],
            'sku_name': s['sku_name'],
            'eoq': safe_float(s['eoq']),
            'annual_demand': safe_float(s['annual_demand']),
            'safety_stock': safe_float(s['safety_stock']),
            'rop': safe_float(s['rop']),
        } for s in sorted(sku_results, key=lambda x: x['annual_revenue'], reverse=True)[:15]]

        # Cost breakdown chart (top SKUs)
        cost_breakdown = [{
            'sku': s['sku'],
            'sku_name': s['sku_name'],
            'ordering': safe_float(s['cost_ordering']),
            'holding': safe_float(s['cost_holding']),
            'total': safe_float(s['cost_ordering'] + s['cost_holding']),
        } for s in sorted(sku_results, key=lambda x: x['cost_total'], reverse=True)[:15]]

        # Savings chart
        savings_chart = [{
            'sku': s['sku'],
            'sku_name': s['sku_name'],
            'savings': safe_float(s['savings']),
            'savings_pct': safe_float(s['savings_pct']),
            'naive_cost': safe_float(s['cost_naive_total']),
            'optimal_cost': safe_float(s['cost_total']),
        } for s in sorted(sku_results, key=lambda x: x['savings'], reverse=True)[:15] if s['savings'] > 0]

        # Inventory turns chart
        turns_chart = [{
            'sku': s['sku'],
            'sku_name': s['sku_name'],
            'turns': safe_float(s['inventory_turns']),
            'abc_class': s['abc_class'],
        } for s in sorted(sku_results, key=lambda x: x['inventory_turns'])[:15]]

        # ABC Pareto chart
        sorted_by_rev = sorted(sku_results, key=lambda x: x['annual_revenue'], reverse=True)
        cumsum = 0
        pareto_chart = []
        for i, s in enumerate(sorted_by_rev):
            cumsum += s['annual_revenue']
            pareto_chart.append({
                'sku': s['sku'],
                'sku_name': s['sku_name'],
                'revenue': safe_float(s['annual_revenue']),
                'cum_pct': safe_float(cumsum / total_annual_revenue * 100) if total_annual_revenue > 0 else 0,
                'abc_class': s['abc_class'],
            })

        # Scatter: CV vs Annual Revenue (colored by ABC)
        scatter_data = [{
            'sku': s['sku'],
            'sku_name': s['sku_name'],
            'cv': safe_float(s['cv']),
            'annual_revenue': safe_float(s['annual_revenue']),
            'abc_class': s['abc_class'],
            'xyz_class': s['xyz_class'],
            'safety_stock': safe_float(s['safety_stock']),
        } for s in sku_results]

        # Category summary
        category_chart = []
        categories = set(s['category'] for s in sku_results if s['category'])
        for cat in sorted(categories):
            items = [s for s in sku_results if s['category'] == cat]
            category_chart.append({
                'category': cat,
                'sku_count': len(items),
                'total_revenue': safe_float(sum(s['annual_revenue'] for s in items)),
                'avg_turns': safe_float(np.mean([s['inventory_turns'] for s in items if s['inventory_turns'] > 0])),
                'total_safety_stock_value': safe_float(sum(s['safety_stock'] * s['unit_cost'] for s in items)),
            })

        # ── 7. Columns used ──
        columns_used = {
            'sku': col_sku,
            'demand': col_demand,
            'date': col_date,
            'lead_time': col_lt,
            'unit_cost': col_unit_cost,
            'order_cost': col_order_cost,
            'price': col_price,
            'holding_pct': col_holding_pct,
        }

        # ── Response ──
        results = {
            'n_skus': n_skus,
            'n_periods': int(df.groupby(col_sku).size().median()) if col_sku else 0,
            'n_rows': len(df),
            'target_service_level': service_level,
            'periods_per_year': periods_per_year,
            'columns_used': columns_used,
            'summary': {
                'total_annual_revenue': safe_float(total_annual_revenue),
                'total_annual_demand_value': safe_float(total_annual_demand_value),
                'total_safety_stock_value': safe_float(total_safety_stock_value),
                'total_optimal_cost': safe_float(total_eoq_cost),
                'total_naive_cost': safe_float(total_naive_cost),
                'total_savings': safe_float(total_savings),
                'total_savings_pct': safe_float(total_savings / total_naive_cost * 100) if total_naive_cost > 0 else 0,
                'avg_inventory_turns': safe_float(avg_turns),
                'avg_service_level': safe_float(avg_service_level),
                'n_a_class': len([s for s in sku_results if s['abc_class'] == 'A']),
                'n_b_class': len([s for s in sku_results if s['abc_class'] == 'B']),
                'n_c_class': len([s for s in sku_results if s['abc_class'] == 'C']),
            },
            'sku_details': sku_results,
            'abc_summary': abc_summary,
            'xyz_summary': xyz_summary,
            'abc_xyz_matrix': abc_xyz_matrix,
            'service_sensitivity': service_sensitivity,
            'charts': {
                'eoq_comparison': eoq_comparison,
                'cost_breakdown': cost_breakdown,
                'savings': savings_chart,
                'turns': turns_chart,
                'pareto': pareto_chart,
                'scatter': scatter_data,
                'category': category_chart,
            },
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


def _get_policy_recommendation(abc: str, xyz: str) -> str:
    policies = {
        'AX': 'JIT / Kanban — high value, predictable. Minimize stock, frequent small orders.',
        'AY': 'MRP with safety stock — high value, moderate variability. EOQ + reviewed safety stock.',
        'AZ': 'Strategic buffer — high value, unpredictable. Hold safety stock, dual sourcing recommended.',
        'BX': 'Automated reorder — moderate value, predictable. Standard EOQ, periodic review.',
        'BY': 'EOQ + moderate safety stock — standard inventory management.',
        'BZ': 'Safety stock focus — moderate value, volatile. Higher safety stock, flexible suppliers.',
        'CX': 'Bulk ordering — low value, predictable. Large infrequent orders to minimize ordering cost.',
        'CY': 'Periodic review — low value, some variability. Simple min-max policy.',
        'CZ': 'Simplify — low value, unpredictable. Consider consignment, VMI, or drop from catalog.',
    }
    return policies.get(f"{abc}{xyz}", 'Standard EOQ policy.')
