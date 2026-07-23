import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate, stats
import io
import base64
import warnings

warnings.filterwarnings('ignore')


def _native(obj):
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return _native(obj.tolist())
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    return obj


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')


def find_intersection(x, y1, y2):
    """Real crossing-point solve: first sign-change of (y1-y2) with linear interpolation."""
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    diff = y1 - y2
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) == 0:
        idx = int(np.argmin(np.abs(diff)))
        return float(x[idx])
    idx = sign_changes[0]
    x1, x2 = x[idx], x[idx + 1]
    d1, d2 = diff[idx], diff[idx + 1]
    if d2 == d1:
        return float(x1)
    t = d1 / (d1 - d2)
    return float(x1 + t * (x2 - x1))


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        too_cheap_col = payload.get('too_cheap_col')
        cheap_col = payload.get('cheap_col')
        expensive_col = payload.get('expensive_col')
        too_expensive_col = payload.get('too_expensive_col')
        segment_col = payload.get('segment_col') or None
        intention_cols = payload.get('intention_cols') or None  # list of {price: number, col: str}
        demand_points = payload.get('demand_points') or None    # list of {price, units}
        competitors = payload.get('competitors') or None        # list of {name, price, value_tier}

        if not data:
            raise ValueError('Missing required data.')
        price_cols = [too_cheap_col, cheap_col, expensive_col, too_expensive_col]
        if not all(price_cols):
            raise ValueError('Missing one or more of the four Van Westendorp price columns.')

        df = pd.DataFrame(data)
        for col in price_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found.")
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df_clean = df[price_cols + ([segment_col] if segment_col and segment_col in df.columns else [])].dropna(subset=price_cols)
        if len(df_clean) < 10:
            raise ValueError(f'Need at least 10 complete responses, found {len(df_clean)}.')
        n = len(df_clean)

        # ---- ② Question setup: median/mean per question ----
        question_setup = []
        labels = [('too_cheap', 'Too Cheap'), ('cheap', 'Cheap'), ('expensive', 'Expensive'), ('too_expensive', 'Too Expensive')]
        for (key, label), col in zip(labels, price_cols):
            s = df_clean[col]
            question_setup.append({
                'question': label, 'column': col,
                'median': float(s.median()), 'mean': float(s.mean()),
                'std': float(s.std()) if n > 1 else 0.0,
                'min': float(s.min()), 'max': float(s.max()),
            })

        # ---- ③ Cumulative curves (real CDFs) ----
        price_min = df_clean[price_cols].min().min()
        price_max = df_clean[price_cols].max().max()
        price_range = np.linspace(price_min, price_max, 400)

        too_cheap_cum = np.array([(df_clean[too_cheap_col] >= p).sum() / n * 100 for p in price_range])       # "too cheap" share falls as price rises
        cheap_cum = np.array([(df_clean[cheap_col] >= p).sum() / n * 100 for p in price_range])
        expensive_cum = np.array([(df_clean[expensive_col] <= p).sum() / n * 100 for p in price_range])       # "expensive" share rises as price rises
        too_expensive_cum = np.array([(df_clean[too_expensive_col] <= p).sum() / n * 100 for p in price_range])

        curves = {
            'too_cheap': too_cheap_cum, 'cheap': cheap_cum,
            'expensive': expensive_cum, 'too_expensive': too_expensive_cum,
        }

        # ---- ④ Key price points (standard VW crossing definitions) ----
        pmc = find_intersection(price_range, too_cheap_cum, expensive_cum)       # Too Cheap ∩ Expensive
        ipp = find_intersection(price_range, cheap_cum, expensive_cum)          # Cheap ∩ Expensive
        opp = find_intersection(price_range, too_cheap_cum, too_expensive_cum)  # Too Cheap ∩ Too Expensive
        pme = find_intersection(price_range, cheap_cum, too_expensive_cum)      # Cheap ∩ Too Expensive

        # sanity clamp: PMC should be <= PME; sort defensively in case of thin/odd data
        acceptable_low, acceptable_high = (pmc, pme) if pmc <= pme else (pme, pmc)

        key_price_points = [
            {'code': 'PMC', 'name': 'Point of Marginal Cheapness', 'value': pmc,
             'definition': 'Too Cheap curve crosses Expensive curve — below this, too many buyers begin to doubt quality.'},
            {'code': 'IPP', 'name': 'Indifference Price Point', 'value': ipp,
             'definition': 'Cheap curve crosses Expensive curve — the price the median respondent treats as normal.'},
            {'code': 'OPP', 'name': 'Optimal Price Point', 'value': opp,
             'definition': 'Too Cheap curve crosses Too Expensive curve — equal resistance from both directions.'},
            {'code': 'PME', 'name': 'Point of Marginal Expensiveness', 'value': pme,
             'definition': 'Cheap curve crosses Too Expensive curve — above this, too many buyers reject the price as too high.'},
        ]

        # ---- Chart: PSM curves (centerpiece) ----
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.plot(price_range, curves['too_cheap'], label='Too Cheap', color='#2E86AB', linewidth=2.2)
        ax.plot(price_range, curves['cheap'], label='Cheap', color='#7CB518', linewidth=2.2)
        ax.plot(price_range, curves['expensive'], label='Expensive', color='#F18F01', linewidth=2.2)
        ax.plot(price_range, curves['too_expensive'], label='Too Expensive', color='#C73E1D', linewidth=2.2)
        for val, name, color in [(pmc, 'PMC', '#2E86AB'), (ipp, 'IPP', '#7CB518'), (opp, 'OPP', 'green'), (pme, 'PME', '#C73E1D')]:
            ax.axvline(val, color=color, linestyle=':', alpha=0.8, linewidth=1.8)
            ax.text(val, 96, f'{name}\n{val:.1f}', ha='center', fontsize=8, bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85))
        ax.axvspan(acceptable_low, acceptable_high, color='green', alpha=0.08)
        ax.set_xlabel('Price'); ax.set_ylabel('Cumulative % of respondents')
        ax.set_title('Van Westendorp Price Sensitivity Meter')
        ax.legend(loc='center right', fontsize=9); ax.grid(alpha=0.3)
        ax.set_ylim(0, 100); ax.set_xlim(price_range.min(), price_range.max())
        plt.tight_layout()
        chart_psm = fig_to_b64(fig)

        # ---- ⑤ Acceptable price range bar (separate visual) ----
        fig, ax = plt.subplots(figsize=(9, 2.2))
        ax.barh([0], [acceptable_high - acceptable_low], left=[acceptable_low], color='#7CB518', alpha=0.35, height=0.5)
        for val, name in [(pmc, 'PMC'), (ipp, 'IPP'), (pme, 'PME')]:
            ax.axvline(val, color='#333333', linewidth=1.5)
            ax.text(val, 0.42, f'{name}\n{val:.1f}', ha='center', fontsize=9)
        ax.set_xlim(price_range.min(), price_range.max())
        ax.set_ylim(-0.5, 0.6); ax.set_yticks([])
        ax.set_xlabel('Price'); ax.set_title('Acceptable Price Range (PMC – IPP – PME)')
        plt.tight_layout()
        chart_range = fig_to_b64(fig)

        # ---- ⑧ Distribution of per-respondent acceptable price (midpoint of cheap-expensive) ----
        per_resp_mid = (df_clean[cheap_col] + df_clean[expensive_col]) / 2.0
        dist_stats = {
            'mean': float(per_resp_mid.mean()), 'median': float(per_resp_mid.median()),
            'q1': float(per_resp_mid.quantile(0.25)), 'q3': float(per_resp_mid.quantile(0.75)),
            'std': float(per_resp_mid.std()) if n > 1 else 0.0,
        }
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(per_resp_mid, bins=25, color='#2E86AB', alpha=0.75, edgecolor='white')
        for val, name, color in [(dist_stats['mean'], 'Mean', 'red'), (dist_stats['median'], 'Median', 'green')]:
            ax.axvline(val, color=color, linestyle='--', linewidth=1.6, label=f'{name}: {val:.1f}')
        ax.set_xlabel('Per-respondent acceptable price (midpoint of cheap–expensive)')
        ax.set_ylabel('Respondents'); ax.set_title('Price Sensitivity Distribution')
        ax.legend(); plt.tight_layout()
        chart_distribution = fig_to_b64(fig)

        results = {
            'status': 'ok',
            'n_respondents': n,
            'opp': opp, 'ipp': ipp, 'pmc': pmc, 'pme': pme,
            'acceptable_low': acceptable_low, 'acceptable_high': acceptable_high,
            'median_expensive': float(df_clean[expensive_col].median()),
            'question_setup': question_setup,
            'key_price_points': key_price_points,
            'distribution': dist_stats,
            'curves': [
                {'price': float(p), 'too_cheap': float(a), 'cheap': float(b), 'expensive': float(c), 'too_expensive': float(d)}
                for p, a, b, c, d in zip(price_range, curves['too_cheap'], curves['cheap'], curves['expensive'], curves['too_expensive'])
            ][::8],  # thin for payload size
            'interpretation': (
                f'Optimal price point (OPP) is {opp:.2f}; acceptable range runs {acceptable_low:.2f}-{acceptable_high:.2f} '
                f'(PMC {pmc:.2f} to PME {pme:.2f}); indifference price (IPP) is {ipp:.2f}.'
            ),
            'charts': {
                'psm_curves': chart_psm,
                'acceptable_range': chart_range,
                'distribution': chart_distribution,
            },
        }

        # ---- ⑥ Purchase intention by price (CONDITIONAL) ----
        if intention_cols and isinstance(intention_cols, list) and len(intention_cols) >= 2:
            pts = []
            for item in intention_cols:
                col = item.get('col'); price = item.get('price')
                if col and col in df.columns and price is not None:
                    vals = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(vals) > 0:
                        pts.append({'price': float(price), 'mean_intention': float(vals.mean()), 'n': int(len(vals))})
            pts.sort(key=lambda x: x['price'])
            if len(pts) >= 2:
                fig, ax = plt.subplots(figsize=(9, 5))
                xs = [p['price'] for p in pts]; ys = [p['mean_intention'] for p in pts]
                ax.plot(xs, ys, marker='o', color='#7A28CB', linewidth=2.2)
                ax.set_xlabel('Price'); ax.set_ylabel('Mean purchase intention (0-100)')
                ax.set_title('Purchase Intention by Price'); ax.grid(alpha=0.3)
                plt.tight_layout()
                results['charts']['purchase_intention'] = fig_to_b64(fig)
                results['purchase_intention'] = {'status': 'ok', 'points': pts}
            else:
                results['purchase_intention'] = {'status': 'skipped', 'note': 'Not enough intention-by-price columns provided.'}
        else:
            results['purchase_intention'] = {'status': 'skipped', 'note': 'No purchase-intention-by-price data provided; skipping section 6.'}

        # ---- ⑦ Price acceptance by segment (CONDITIONAL) ----
        if segment_col and segment_col in df_clean.columns:
            seg_rows = []
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.tab10(np.linspace(0, 1, df_clean[segment_col].nunique()))
            for color, (seg_name, sub) in zip(colors, df_clean.groupby(segment_col)):
                if len(sub) < 5:
                    continue
                ns = len(sub)
                tc = np.array([(sub[too_cheap_col] >= p).sum() / ns * 100 for p in price_range])
                ch = np.array([(sub[cheap_col] >= p).sum() / ns * 100 for p in price_range])
                ex = np.array([(sub[expensive_col] <= p).sum() / ns * 100 for p in price_range])
                te = np.array([(sub[too_expensive_col] <= p).sum() / ns * 100 for p in price_range])
                seg_ipp = find_intersection(price_range, ch, ex)
                seg_opp = find_intersection(price_range, tc, te)
                seg_pmc = find_intersection(price_range, tc, ex)
                seg_pme = find_intersection(price_range, ch, te)
                lo, hi = (seg_pmc, seg_pme) if seg_pmc <= seg_pme else (seg_pme, seg_pmc)
                seg_rows.append({'segment': str(seg_name), 'n': ns, 'ipp': seg_ipp, 'opp': seg_opp,
                                  'acceptable_low': lo, 'acceptable_high': hi})
                ax.plot(price_range, ex, color=color, linewidth=2, label=f'{seg_name} (Expensive)')
                ax.plot(price_range, te, color=color, linewidth=2, linestyle='--', label=f'{seg_name} (Too Expensive)')
            ax.set_xlabel('Price'); ax.set_ylabel('Cumulative %'); ax.set_title('Price Acceptance by Segment')
            ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.tight_layout()
            if seg_rows:
                results['charts']['segments'] = fig_to_b64(fig)
                results['segments'] = {'status': 'ok', 'rows': seg_rows}
            else:
                plt.close(fig)
                results['segments'] = {'status': 'skipped', 'note': 'Segment column present but no segment had enough respondents (n>=5).'}
        else:
            results['segments'] = {'status': 'skipped', 'note': 'No segment column provided; skipping section 7.'}

        # ---- ⑨ Price elasticity, ⑩ revenue simulation, ⑪ scenario analysis (CONDITIONAL on demand_points) ----
        if demand_points and isinstance(demand_points, list) and len(demand_points) >= 3:
            dp = pd.DataFrame(demand_points)
            dp = dp.dropna(subset=['price', 'units'])
            dp = dp.sort_values('price').reset_index(drop=True)
            if len(dp) >= 3:
                # point elasticities between consecutive levels
                point_elasticities = []
                for i in range(len(dp) - 1):
                    p1, p2 = dp.loc[i, 'price'], dp.loc[i + 1, 'price']
                    q1, q2 = dp.loc[i, 'units'], dp.loc[i + 1, 'units']
                    pct_q = (q2 - q1) / q1
                    pct_p = (p2 - p1) / p1
                    e = pct_q / pct_p if pct_p != 0 else None
                    point_elasticities.append({'price_from': float(p1), 'price_to': float(p2), 'elasticity': None if e is None else float(e)})

                # overall log-log regression slope = elasticity
                log_p = np.log(dp['price'].values.astype(float))
                log_q = np.log(dp['units'].values.astype(float))
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_p, log_q)
                overall_elasticity = float(slope)

                results['elasticity'] = {
                    'status': 'ok', 'overall_elasticity': overall_elasticity, 'r_squared': float(r_value ** 2),
                    'point_elasticities': point_elasticities,
                    'interpretation': f'Overall log-log elasticity is {overall_elasticity:.2f}: a 1% price increase is associated with a {overall_elasticity:.2f}% change in demand.'
                }

                # revenue simulation across observed price range (interpolated)
                f_demand = interpolate.interp1d(dp['price'], dp['units'], kind='linear', fill_value='extrapolate', bounds_error=False)
                sim_prices = np.linspace(dp['price'].min(), dp['price'].max(), 60)
                sim_demand = f_demand(sim_prices)
                sim_revenue = sim_prices * sim_demand
                best_idx = int(np.argmax(sim_revenue))
                revenue_max_price = float(sim_prices[best_idx])
                revenue_max_value = float(sim_revenue[best_idx])

                fig, ax = plt.subplots(figsize=(9, 5))
                ax.plot(sim_prices, sim_revenue, color='#2E86AB', linewidth=2.2)
                ax.scatter(dp['price'], dp['price'] * dp['units'], color='#C73E1D', zorder=5, label='Observed price points')
                ax.axvline(revenue_max_price, color='green', linestyle='--', linewidth=1.8, label=f'Revenue-max price {revenue_max_price:.1f}')
                ax.set_xlabel('Price'); ax.set_ylabel('Expected Revenue'); ax.set_title('Revenue Simulation')
                ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
                results['charts']['revenue_simulation'] = fig_to_b64(fig)
                results['revenue_simulation'] = {
                    'status': 'ok', 'revenue_max_price': revenue_max_price, 'revenue_max_value': revenue_max_value,
                    'curve': [{'price': float(p), 'demand': float(d), 'revenue': float(r)} for p, d, r in zip(sim_prices, sim_demand, sim_revenue)][::5],
                }

                # scenario analysis: current price = observed price closest to OPP (or median demand price)
                current_price = float(dp.loc[(dp['price'] - opp).abs().idxmin(), 'price'])
                scenarios = []
                for label, mult in [('-10%', 0.9), ('-5%', 0.95), ('current', 1.0), ('+5%', 1.05), ('+10%', 1.10)]:
                    p = current_price * mult
                    q = float(f_demand(p))
                    rev = p * q
                    scenarios.append({'scenario': label, 'price': float(p), 'demand': q, 'revenue': float(rev)})
                current_rev = next(s['revenue'] for s in scenarios if s['scenario'] == 'current')
                best_scenario = max(scenarios, key=lambda s: s['revenue'])
                if best_scenario['scenario'] == 'current':
                    conclusion = f"Current price ({current_price:.2f}) already yields the highest revenue among the tested scenarios ({current_rev:.0f}); price appears near-optimal."
                else:
                    delta_pct = (best_scenario['revenue'] - current_rev) / current_rev * 100 if current_rev else 0
                    conclusion = (
                        f"Moving price to {best_scenario['scenario']} ({best_scenario['price']:.2f}) yields "
                        f"{'a limited' if abs(delta_pct) < 5 else 'a notable'} revenue change of {delta_pct:+.1f}% versus the current price."
                    )
                results['scenario_analysis'] = {'status': 'ok', 'current_price': current_price, 'scenarios': scenarios, 'conclusion': conclusion}
            else:
                results['elasticity'] = {'status': 'skipped', 'note': 'Not enough valid price-demand rows.'}
                results['revenue_simulation'] = {'status': 'skipped', 'note': 'Not enough valid price-demand rows.'}
                results['scenario_analysis'] = {'status': 'skipped', 'note': 'Not enough valid price-demand rows.'}
        else:
            results['elasticity'] = {'status': 'skipped', 'note': 'No real price-vs-demand data provided; skipping section 9 (elasticity requires actual sales/demand data, not PSM survey responses alone).'}
            results['revenue_simulation'] = {'status': 'skipped', 'note': 'No real price-vs-demand data provided; skipping section 10.'}
            results['scenario_analysis'] = {'status': 'skipped', 'note': 'No real price-vs-demand data provided; skipping section 11.'}

        # ---- ⑫ Competitive price perception (CONDITIONAL) ----
        if competitors and isinstance(competitors, list) and len(competitors) >= 1:
            comp_rows = [{'name': c.get('name'), 'price': float(c.get('price')), 'value_tier': c.get('value_tier', '—')} for c in competitors if c.get('price') is not None]
            if comp_rows:
                fig, ax = plt.subplots(figsize=(9, 5))
                names = [c['name'] for c in comp_rows] + ['This product (OPP)']
                prices = [c['price'] for c in comp_rows] + [opp]
                colors_bar = ['#94a3b8'] * len(comp_rows) + ['#2E86AB']
                ax.bar(names, prices, color=colors_bar)
                ax.set_ylabel('Price'); ax.set_title('Competitive Price Perception')
                plt.xticks(rotation=25, ha='right'); plt.tight_layout()
                results['charts']['competitive'] = fig_to_b64(fig)
                results['competitive'] = {'status': 'ok', 'rows': comp_rows}
            else:
                results['competitive'] = {'status': 'skipped', 'note': 'Competitor list provided but no valid prices.'}
        else:
            results['competitive'] = {'status': 'skipped', 'note': 'No competitor price/perceived-value data provided; skipping section 12.'}

        response = {
            'results': _native(results),
            'plot': results['charts']['psm_curves'],
        }
        print(json.dumps(response))

    except Exception as e:
        print(json.dumps({'error': str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
