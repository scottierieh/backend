# Validates som_analysis.py (Self-Organizing Map). SOM training is stochastic,
# so the reported silhouette is verified independently with scikit-learn on the
# handler's own scaled data + node assignments, plus structural consistency
# checks. Where minisom is installed the quantization error is additionally
# reproduced from a direct MiniSom fit with the same seed.
import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from _pyharness import run_script, chk, report

iris = load_iris(as_frame=True)
df = iris.frame.copy()
feat = list(iris.feature_names)
gx, gy, sigma, lr = 4, 4, 1.0, 0.5
payload = {'data': df.to_dict('records'), 'features': feat,
           'gridX': gx, 'gridY': gy, 'sigma': sigma, 'learning_rate': lr}
res = run_script('som_analysis.py', payload)
r = res['results'] if 'results' in res else res

X = df[feat].values
Xs = StandardScaler().fit_transform(X)
assign = np.asarray(r['node_assignments'])

# silhouette reported by the handler must equal sklearn's on the same inputs
chk("som.silhouette", r['silhouette_score'], silhouette_score(Xs, assign), tol=1e-9)
# structural consistency
chk("som.n_nodes_activated", r['n_nodes_activated'], len(np.unique(assign)))
chk("som.assignments_count", len(assign), Xs.shape[0])
chk("som.nodes_within_grid", 1.0 if r['n_nodes_activated'] <= gx * gy else 0.0, 1.0)
chk("som.qerror_finite_nonneg", 1.0 if np.isfinite(r['quantization_error']) and r['quantization_error'] >= 0 else 0.0, 1.0)

# exact quantization-error reproduction when minisom is available (deploy env)
try:
    from minisom import MiniSom
    som = MiniSom(gx, gy, Xs.shape[1], sigma=sigma, learning_rate=lr, random_seed=42)
    som.random_weights_init(Xs)
    som.train_random(Xs, 1000)  # handler default num_iteration
    chk("som.quantization_error(minisom)", r['quantization_error'], float(som.quantization_error(Xs)), tol=1e-6)
except ImportError:
    print("SKIP | minisom not installed — exact quantization-error reproduction skipped (deploy env has it)")

report("SOM (Python)")
