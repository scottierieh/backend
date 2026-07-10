

import sys
import json
import pandas as pd
import numpy as np
import math
from scipy.stats import iqr
import pingouin as pg

def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def calculate_cvr(series, threshold):
    """Calculate Content Validity Ratio."""
    Ne = (series >= threshold).sum()
    N = len(series)
    if N == 0:
        return 0
    return (Ne - (N / 2)) / (N / 2)

def calculate_consensus(series):
    """Calculate consensus based on interquartile range."""
    if len(series) < 2:
        return np.nan
    
    series_range = series.max() - series.min()
    if series_range == 0:
        return 1.0 # Perfect consensus if all values are the same

    return 1 - (iqr(series) / series_range)


def calculate_cv(series):
    """Calculate Coefficient of Variation."""
    mean = series.mean()
    std = series.std()
    if mean == 0:
        return np.inf
    return std / mean

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        rounds = payload.get('rounds', []) # Expects [{ 'name': 'Round 1', 'items': ['item1', 'item2'] }]
        scale_max = int(payload.get('scaleMax', 5))
        cvr_threshold = float(payload.get('cvrThreshold', 4))

        if not data or not rounds:
            raise ValueError("Missing 'data' or 'rounds' configuration.")

        df = pd.DataFrame(data)
        
        all_results = {}
        
        # Calculate stats for each round
        for round_config in rounds:
            round_name = round_config['name']
            items = round_config['items']
            
            round_results = {}
            if not items:
                continue

            # Convert all relevant columns to numeric and then drop rows with any NaN in those columns
            round_df_numeric = df[items].apply(pd.to_numeric, errors='coerce')
            round_df_clean = round_df_numeric.dropna()
            
            for item_col in items:
                if item_col not in round_df_clean.columns:
                    continue
                    
                series = round_df_clean[item_col]
                
                if series.empty:
                    continue

                q1 = series.quantile(0.25)
                median_val = series.median()

                round_results[item_col] = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'median': median_val,
                    'q1': q1,
                    'q3': series.quantile(0.75),
                    'cvr': calculate_cvr(series, cvr_threshold),
                    'consensus': calculate_consensus(series),
                    'convergence': median_val - q1,
                    'cv': calculate_cv(series),
                    'positive_responses': (series >= cvr_threshold).sum(),
                    'stability': np.nan, # Initialize stability
                }
            
            # Calculate Cronbach's Alpha for the round, using the cleaned dataframe
            cronbach_alpha = np.nan
            if not round_df_clean.empty and len(round_df_clean.columns) > 1:
                cronbach_alpha = pg.cronbach_alpha(data=round_df_clean)[0]
            
            all_results[round_name] = {
                "items": round_results,
                "cronbach_alpha": cronbach_alpha
            }
        
        # Calculate Stability between rounds if more than one round exists
        if len(rounds) > 1:
            for i in range(1, len(rounds)):
                prev_round_config = rounds[i-1]
                curr_round_config = rounds[i]
                
                prev_items = prev_round_config['items']
                curr_items = curr_round_config['items']
                
                # Create item mapping based on base name (e.g., 'item1_r1' -> 'item1')
                item_map = {}
                for prev_item in prev_items:
                    base_name = prev_item.split('_r')[0]
                    for curr_item in curr_items:
                        if curr_item.split('_r')[0] == base_name:
                            item_map[curr_item] = prev_item
                            break
                            
                for curr_item, prev_item in item_map.items():
                    if curr_item in all_results[curr_round_config['name']]['items'] and prev_item in all_results[prev_round_config['name']]['items']:
                        curr_mean = all_results[curr_round_config['name']]['items'][curr_item]['mean']
                        prev_mean = all_results[prev_round_config['name']]['items'][prev_item]['mean']
                        
                        if prev_mean != 0:
                            stability = abs(curr_mean - prev_mean) / prev_mean
                            all_results[curr_round_config['name']]['items'][curr_item]['stability'] = stability


        print(json.dumps({'results': all_results}, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
