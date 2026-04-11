from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

router = APIRouter()


class VariabilityRequest(BaseModel):
    data: List[Dict[str, Any]]
    variables: List[str]


def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
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


@router.post("/variability")
async def variability_analysis(request: VariabilityRequest):
    try:
        df = pd.DataFrame(request.data)
        variables = request.variables
        
        if not variables or len(variables) < 2:
            raise HTTPException(status_code=400, detail="At least 2 variables required")
        
        # Check for missing variables
        missing = [v for v in variables if v not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Variables not found: {missing}")
        
        results = []
        
        for var in variables:
            series = pd.to_numeric(df[var], errors='coerce').dropna()
            
            if len(series) == 0:
                results.append({
                    'variable': var,
                    'range': 0,
                    'iqr': 0,
                    'cv': 0
                })
                continue
            
            # Calculate metrics
            range_val = series.max() - series.min()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            mean = series.mean()
            std = series.std()
            raw_cv = (std / abs(mean) * 100) if (mean != 0 and np.isfinite(mean)) else 0.0

            results.append(_to_native({
                'variable': var,
                'range': range_val,
                'iqr': iqr,
                'cv': raw_cv
            }))
        
        # Generate interpretation
        interpretation_parts = []
        
        # Sort by CV to find most/least variable
        sorted_by_cv = sorted(results, key=lambda x: x['cv'])
        most_consistent = sorted_by_cv[0]
        most_variable = sorted_by_cv[-1]
        
        interpretation_parts.append(
            f"**Most consistent variable:** {most_consistent['variable']} (CV = {most_consistent['cv']:.1f}%)"
        )
        interpretation_parts.append(
            f"**Most variable:** {most_variable['variable']} (CV = {most_variable['cv']:.1f}%)"
        )
        
        # Check for high variability warnings
        high_cv_vars = [r for r in results if r['cv'] > 50]
        if high_cv_vars:
            var_names = ', '.join([r['variable'] for r in high_cv_vars])
            interpretation_parts.append(
                f"⚠ **High variability detected** in: {var_names} (CV > 50%). These variables show considerable inconsistency relative to their means."
            )
        
        # Check for low variability
        low_cv_vars = [r for r in results if r['cv'] < 10]
        if low_cv_vars:
            var_names = ', '.join([r['variable'] for r in low_cv_vars])
            interpretation_parts.append(
                f"✓ **Low variability** in: {var_names} (CV < 10%). These variables are highly consistent."
            )
        
        # IQR vs Range comparison for outlier detection
        potential_outliers = []
        for r in results:
            if r['range'] > 0 and r['iqr'] > 0:
                ratio = r['range'] / r['iqr']
                if ratio > 4:  # Range much larger than IQR suggests outliers
                    potential_outliers.append(r['variable'])
        
        if potential_outliers:
            interpretation_parts.append(
                f"⚠ **Potential outliers** suspected in: {', '.join(potential_outliers)} (Range >> IQR). Consider examining extreme values."
            )
        
        interpretation = '\n'.join(interpretation_parts)
        
        return _to_native({
            'results': results,
            'interpretation': interpretation
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
