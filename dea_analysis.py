
import sys
import json
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import io
import base64

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

warnings.filterwarnings('ignore')

def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class DEAAnalyzer:
    def __init__(self, data: pd.DataFrame, input_cols: list, output_cols: list, dmu_col: str):
        self.dmu_names = data[dmu_col].tolist()
        self.inputs = data[input_cols].values
        self.outputs = data[output_cols].values
        
    def _generate_interpretation(self, results, orientation):
        scores = np.array([s for s in results['efficiency_scores'].values() if s is not None and not np.isnan(s)])
        if len(scores) == 0:
            return "No efficiency scores could be calculated."

        mean_score = np.mean(scores)
        std_dev = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)

        efficient_units = [dmu for dmu, score in results['efficiency_scores'].items() if score is not None and score >= 0.9999]
        
        interpretation = (
            f"Data Envelopment Analysis revealed varying efficiency levels across the "
            f"{results['summary']['total_dmus']} examined decision-making units, with efficiency scores ranging from {min_score:.3f} "
            f"to {max_score:.3f} (M = {mean_score:.3f}, SD = {std_dev:.3f}).\n"
        )

        interpretation += (
            f"{len(efficient_units)} unit(s) ({', '.join(efficient_units)}) achieved full efficiency (score ≈ 1.000), "
            f"serving as benchmark references for suboptimal performers.\n"
        )
        
        inefficient_units = {dmu: score for dmu, score in results['efficiency_scores'].items() if score is not None and score < 0.9999}
        if inefficient_units:
            min_eff_dmu = min(inefficient_units, key=inefficient_units.get)
            min_eff_score = inefficient_units[min_eff_dmu]

            if orientation == 'input':
                input_reduction = (1 - min_eff_score) * 100
                interpretation += (
                    f"Unit '{min_eff_dmu}' demonstrated the lowest efficiency (score = {min_eff_score:.3f}), "
                    f"indicating that it could potentially reduce its inputs by {input_reduction:.1f}% "
                    f"while maintaining current output levels.\n"
                )
            else: # output-oriented
                output_increase = ((1 / min_eff_score) - 1) * 100 if min_eff_score > 0 else float('inf')
                interpretation += (
                    f"Unit '{min_eff_dmu}' demonstrated the lowest efficiency (score = {min_eff_score:.3f}), "
                    f"indicating that it could potentially increase its outputs by {output_increase:.1f}% "
                    f"using its existing resources.\n"
                )

            # Peer weights for the most inefficient unit
            peer_weights = results['lambdas'].get(min_eff_dmu)
            ref_set = results['reference_sets'].get(min_eff_dmu)
            if peer_weights and ref_set:
                peer_info = [f"{ref} (weight: {weight:.2f})" for ref, weight in zip(ref_set, [peer_weights[results['dmu_names'].index(r)] for r in ref_set]) if weight > 1e-6]
                if peer_info:
                    interpretation += (
                        f"To improve, '{min_eff_dmu}' should benchmark against the practices of its peer reference set: {', '.join(peer_info)}.\n"
                    )

        interpretation += (
            f"Overall, {results['summary']['inefficient_dmus']} out of {results['summary']['total_dmus']} units ({results['summary']['inefficient_dmus']/results['summary']['total_dmus']*100:.1f}%) "
            "operate below the efficiency frontier, suggesting opportunities for widespread operational improvements."
        )

        return interpretation


    def analyze(self, orientation='input', rts='crs'):
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy library is not installed. Please install it via 'pip install scipy'.")
        
        n_dmus, n_inputs = self.inputs.shape
        n_outputs = self.outputs.shape[1]
        
        efficiencies = np.zeros(n_dmus)
        lambdas_list = []

        for k in range(n_dmus):
            if orientation == 'input':
                # Objective function: minimize theta
                c = np.zeros(n_dmus + 1)
                c[0] = 1

                # Constraints
                A_ub = np.zeros((n_inputs, n_dmus + 1))
                A_ub[:, 1:] = self.inputs.T
                A_ub[:, 0] = -self.inputs[k, :]
                b_ub = np.zeros(n_inputs)

                A_ub_output = np.zeros((n_outputs, n_dmus + 1))
                A_ub_output[:, 1:] = -self.outputs.T
                b_ub_output = -self.outputs[k, :]

                A_ub = np.vstack([A_ub, A_ub_output])
                b_ub = np.concatenate([b_ub, b_ub_output])

                if rts == 'vrs':
                    A_eq = np.ones((1, n_dmus + 1))
                    A_eq[0, 0] = 0
                    b_eq = np.array([1])
                else: # CRS
                    A_eq = None
                    b_eq = None

                bounds = [(0, None)] + [(0, None) for _ in range(n_dmus)]

                res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

                if res.success:
                    efficiencies[k] = res.fun
                    lambdas_list.append(res.x[1:].tolist())
                else:
                    efficiencies[k] = np.nan
                    lambdas_list.append([np.nan] * n_dmus)
            
            else: # Output-oriented
                c = np.zeros(n_dmus + 1)
                c[0] = -1

                A_ub = np.zeros((n_inputs, n_dmus + 1))
                A_ub[:, 1:] = self.inputs.T
                b_ub = self.inputs[k, :]

                A_ub_output = np.zeros((n_outputs, n_dmus + 1))
                A_ub_output[:, 1:] = -self.outputs.T
                A_ub_output[:, 0] = self.outputs[k, :]
                b_ub_output = np.zeros(n_outputs)
                
                A_ub = np.vstack([A_ub, A_ub_output])
                b_ub = np.concatenate([b_ub, b_ub_output])
                
                if rts == 'vrs':
                    A_eq = np.ones((1, n_dmus + 1))
                    A_eq[0, 0] = 0
                    b_eq = np.array([1])
                else: # CRS
                    A_eq = None
                    b_eq = None

                bounds = [(1, None)] + [(0, None) for _ in range(n_dmus)]

                res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

                if res.success:
                    phi = -res.fun
                    efficiencies[k] = 1/phi if phi != 0 else np.inf
                    lambdas_list.append(res.x[1:].tolist())
                else:
                    efficiencies[k] = np.nan
                    lambdas_list.append([np.nan] * n_dmus)
        
        efficiency_scores = {self.dmu_names[i]: eff for i, eff in enumerate(efficiencies)}
        lambdas = {self.dmu_names[i]: l for i, l in enumerate(lambdas_list)}
        reference_sets = {dmu: [self.dmu_names[j] for j, l_val in enumerate(lambdas_list[i]) if l_val > 1e-6]
                          for i, dmu in enumerate(self.dmu_names)}
        
        # Prepare final results object
        eff_scores_list = [s for s in efficiency_scores.values() if s is not None and not np.isnan(s)]
        summary = {
            'total_dmus': len(self.dmu_names),
            'efficient_dmus': sum(1 for e in eff_scores_list if e >= 0.9999),
            'inefficient_dmus': sum(1 for e in eff_scores_list if e < 0.9999),
            'average_efficiency': np.mean(eff_scores_list) if eff_scores_list else 0
        }
        
        final_results = {
            'efficiency_scores': efficiency_scores,
            'reference_sets': reference_sets,
            'lambdas': lambdas,
            'summary': summary,
            'dmu_col': self.dmu_names[0] if self.dmu_names else "",
            'dmu_names': self.dmu_names,
        }

        final_results['interpretation'] = self._generate_interpretation(final_results, orientation)
        return final_results
    
    def plot_frontier(self, results):
        if self.inputs.shape[1] != 1 or self.outputs.shape[1] != 1:
            return None # Can only plot for 1 input, 1 output
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        efficient_dmus = {dmu for dmu, score in results['efficiency_scores'].items() if score is not None and score >= 0.9999}
        
        dmu_data = []
        for i, dmu_name in enumerate(self.dmu_names):
             dmu_data.append({
                'name': dmu_name,
                'input': self.inputs[i, 0],
                'output': self.outputs[i, 0],
                'is_efficient': dmu_name in efficient_dmus
             })
             
        df_plot = pd.DataFrame(dmu_data)
        
        sns.scatterplot(data=df_plot, x='input', y='output', hue='is_efficient', style='is_efficient', s=100, ax=ax, palette={True: 'green', False: 'red'})
        
        for i, row in df_plot.iterrows():
            ax.text(row['input'], row['output'], row['name'], fontsize=9, ha='right')

        # Draw frontier
        frontier_points = df_plot[df_plot['is_efficient']].sort_values(by='input')
        if len(frontier_points) > 1:
            ax.plot(frontier_points['input'], frontier_points['output'], color='green', linestyle='-', marker='o', label='Efficiency Frontier')

        ax.set_title('DEA Efficiency Frontier')
        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        ax.legend()
        ax.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')


def main():
    try:
        payload = json.load(sys.stdin)
        data_json = payload.get('data')
        dmu_col = payload.get('dmu_col')
        input_cols = payload.get('input_cols')
        output_cols = payload.get('output_cols')
        orientation = payload.get('orientation', 'input')
        rts = payload.get('rts', 'crs')

        if not all([data_json, dmu_col, input_cols, output_cols]):
            raise ValueError("Missing required parameters.")
            
        df = pd.DataFrame(data_json)
        
        for col in input_cols + output_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=[col], inplace=True)
            if (df[col] <= 0).any():
                raise ValueError(f"All values in input/output column '{col}' must be positive for DEA.")

        if df.empty:
             raise ValueError("No valid numeric data for analysis.")

        analyzer = DEAAnalyzer(df, input_cols, output_cols, dmu_col)
        results = analyzer.analyze(orientation, rts)
        
        plot_image = None
        if len(input_cols) == 1 and len(output_cols) == 1:
             plot_image = analyzer.plot_frontier(results)


        response = {
            'results': results,
            'plot': f"data:image/png;base64,{plot_image}" if plot_image else None
        }
        
        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
