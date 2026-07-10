
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import io
import base64
import warnings

warnings.filterwarnings('ignore')

def _to_native_type(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

def find_intersection(x, y1, y2):
    """Find intersection point of two curves"""
    try:
        # Create interpolation functions
        f1 = interpolate.interp1d(x, y1, kind='linear', fill_value='extrapolate', bounds_error=False)
        f2 = interpolate.interp1d(x, y2, kind='linear', fill_value='extrapolate', bounds_error=False)
        
        # Find where curves cross
        diff = f1(x) - f2(x)
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        
        if len(sign_changes) == 0:
            # No intersection found, return point of minimum difference
            idx = np.argmin(np.abs(diff))
            return x[idx]
        
        # Use first intersection point
        idx = sign_changes[0]
        
        # Linear interpolation for precise intersection
        x1, x2 = x[idx], x[idx + 1]
        y1_1, y1_2 = f1(x1), f1(x2)
        y2_1, y2_2 = f2(x1), f2(x2)
        
        # Solve for intersection
        denominator = (y1_2 - y1_1) - (y2_2 - y2_1)
        if denominator == 0:
            return x1
        
        t = (y2_1 - y1_1) / denominator
        return x1 + t * (x2 - x1)
        
    except Exception:
        # Fallback: find minimum difference
        diff = np.abs(np.array(y1) - np.array(y2))
        idx = np.argmin(diff)
        return x[idx]

def calculate_vw_curves(df, price_cols, price_range):
    """Calculate Van Westendorp cumulative curves"""
    too_cheap_col, cheap_col, expensive_col, too_expensive_col = price_cols
    n = len(df)
    if n == 0:
        return None

    # Cumulative "Cheaper than" percentages
    too_cheap_cum = np.array([(df[too_cheap_col] <= p).sum() / n * 100 for p in price_range])
    cheap_cum = np.array([(df[cheap_col] <= p).sum() / n * 100 for p in price_range])

    # Cumulative "More expensive than" percentages
    expensive_cum_inv = np.array([(df[expensive_col] > p).sum() / n * 100 for p in price_range])
    too_expensive_cum_inv = np.array([(df[too_expensive_col] > p).sum() / n * 100 for p in price_range])
    
    return {
        'too_cheap': too_cheap_cum,
        'cheap': cheap_cum,
        'expensive': expensive_cum_inv,
        'too_expensive': too_expensive_cum_inv
    }


def calculate_price_points(price_range, curves):
    """Calculate Van Westendorp price points from curves"""
    # Point of Marginal Cheapness (PMC): "Too Cheap" vs "Expensive"
    pmc = find_intersection(price_range, curves['too_cheap'], curves['expensive'])
    
    # Point of Marginal Expensiveness (PME): "Cheap" vs "Too Expensive"
    pme = find_intersection(price_range, curves['cheap'], curves['too_expensive'])
    
    # Indifference Price Point (IDP): "Cheap" vs "Expensive"
    idp = find_intersection(price_range, curves['cheap'], curves['expensive'])
    
    # Optimal Price Point (OPP): "Too Cheap" vs "Too Expensive"
    opp = find_intersection(price_range, curves['too_cheap'], curves['too_expensive'])

    return {
        'pmc': float(pmc),
        'pme': float(pme),
        'opp': float(opp),
        'idp': float(idp)
    }

def create_psm_plot(price_range, curves, results):
    """Create Van Westendorp PSM plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the four main curves
    ax.plot(price_range, curves['too_cheap'], label='Too Cheap', color='#2E86AB', linewidth=2.5, linestyle='--')
    ax.plot(price_range, curves['cheap'], label='Cheap', color='#A23B72', linewidth=2.5, linestyle='--')
    ax.plot(price_range, curves['expensive'], label='Expensive', color='#F18F01', linewidth=2.5, linestyle='-')
    ax.plot(price_range, curves['too_expensive'], label='Too Expensive', color='#C73E1D', linewidth=2.5, linestyle='-')
    
    # Add vertical lines for price points
    if results['pmc']:
        ax.axvline(results['pmc'], color='#2E86AB', linestyle=':', alpha=0.7, linewidth=2)
        ax.text(results['pmc'], 95, f'PMC\n${results["pmc"]:.2f}', ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    if results['pme']:
        ax.axvline(results['pme'], color='#C73E1D', linestyle=':', alpha=0.7, linewidth=2)
        ax.text(results['pme'], 95, f'PME\n${results["pme"]:.2f}', ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    if results['opp']:
        ax.axvline(results['opp'], color='green', linestyle='-', alpha=0.9, linewidth=3)
        ax.text(results['opp'], 85, f'OPP\n${results["opp"]:.2f}', ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.9))
    
    if results['idp']:
        ax.axvline(results['idp'], color='purple', linestyle=':', alpha=0.7, linewidth=2)
        ax.text(results['idp'], 75, f'IDP\n${results["idp"]:.2f}', ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Formatting
    ax.set_xlabel('Price ($)', fontsize=12)
    ax.set_ylabel('Percentage of Respondents (%)', fontsize=12)
    ax.set_title('Van Westendorp Price Sensitivity Meter', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim(0, 100)
    ax.set_xlim(price_range.min(), price_range.max())
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_base64

def create_acceptance_plot(price_range, curves):
    """Create price acceptance curve plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    not_too_expensive = 100 - curves['too_expensive']
    not_cheap = curves['cheap'] # This should be "Not Too Cheap" - let's calculate it
    
    # Calculate 'not too cheap' which is 100 - 'too cheap'
    not_too_cheap = 100 - curves['too_cheap']

    acceptance = np.minimum(not_too_cheap, not_too_expensive)
    
    max_idx = np.argmax(acceptance)
    max_price = price_range[max_idx]
    max_acceptance = acceptance[max_idx]
    
    # Plot curves
    ax.plot(price_range, not_too_cheap, label='Not Too Cheap', color='#2E86AB', linewidth=2, linestyle='--', alpha=0.7)
    ax.plot(price_range, not_too_expensive, label='Not Too Expensive', color='#C73E1D', linewidth=2, linestyle='--', alpha=0.7)
    ax.plot(price_range, acceptance, label='Acceptable Price Range', color='green', linewidth=3)
    ax.fill_between(price_range, 0, acceptance, alpha=0.2, color='green')
    
    ax.scatter([max_price], [max_acceptance], s=200, c='darkgreen', marker='*', zorder=5, edgecolors='black', linewidth=1.5)
    ax.annotate(f'Maximum Acceptance\n${max_price:.2f} ({max_acceptance:.1f}%)', (max_price, max_acceptance),
                xytext=(max_price + (price_range.max() - price_range.min()) * 0.05, max_acceptance - 10),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))
    
    ax.set_xlabel('Price ($)', fontsize=12)
    ax.set_ylabel('Percentage of Respondents (%)', fontsize=12)
    ax.set_title('Price Acceptance Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim(0, 100)
    ax.set_xlim(price_range.min(), price_range.max())
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_base64

def generate_interpretation(results):
    pmc, pme, opp, idp = results.get('pmc'), results.get('pme'), results.get('opp'), results.get('idp')

    if any(val is None for val in [pmc, pme, opp, idp]):
        return "Could not determine all price points. Check data quality and distribution."

    interpretation = (
        f"The analysis identifies a recommended price range between **${pmc:.2f}** (Point of Marginal Cheapness) "
        f"and **${pme:.2f}** (Point of Marginal Expensiveness).\n\n"
        f"- **Optimal Price Point (OPP): ${opp:.2f}**\n"
        f"  This minimizes the number of customers who find the product too cheap or too expensive.\n\n"
        f"- **Indifference Price Point (IDP): ${idp:.2f}**\n"
        f"  At this price, equal numbers consider the product 'cheap' vs 'expensive'.\n\n"
        f"**Strategic Recommendations:**\n"
        f"- **Premium positioning**: Price between ${idp:.2f} and ${pme:.2f}\n"
        f"- **Value positioning**: Price between ${pmc:.2f} and ${idp:.2f}\n"
        f"- **Avoid**: Pricing below ${pmc:.2f} (quality concerns) or above ${pme:.2f} (price resistance)"
    )
    return interpretation.strip()

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        too_cheap_col = payload.get('too_cheap_col', 'Too Cheap')
        cheap_col = payload.get('cheap_col', 'Cheap')
        expensive_col = payload.get('expensive_col', 'Expensive')
        too_expensive_col = payload.get('too_expensive_col', 'Too Expensive')

        if not data: raise ValueError("Missing required data.")

        df = pd.DataFrame(data)
        price_cols = [too_cheap_col, cheap_col, expensive_col, too_expensive_col]
        
        for col in price_cols:
            if col not in df.columns: raise ValueError(f"Required column '{col}' not found.")
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df_clean = df[price_cols].dropna()
        if len(df_clean) < 10: raise ValueError(f"Need at least 10 complete responses, found {len(df_clean)}.")
        
        price_min = df_clean[price_cols].min().min()
        price_max = df_clean[price_cols].max().max()
        price_range = np.linspace(price_min, price_max, 500)
            
        curves = calculate_vw_curves(df_clean, price_cols, price_range)
        if curves is None: raise ValueError("Could not calculate VW curves.")

        results = calculate_price_points(price_range, curves)
        results['interpretation'] = generate_interpretation(results)

        plots = {
            'psm_plot': create_psm_plot(price_range, curves, results),
            'acceptance_plot': create_acceptance_plot(price_range, curves)
        }
        
        response = {
            'results': results,
            'plots': plots,
            'summary': {
                'total_responses': len(df_clean),
                'price_range': {
                    'min': float(price_min),
                    'max': float(price_max)
                }
            }
        }
        
        print(json.dumps(response, default=_to_native_type))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
