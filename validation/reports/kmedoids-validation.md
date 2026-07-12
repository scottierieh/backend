# k-Medoids clustering (Python) — Validation Report
- **Target**: `kmedoids_analysis.py` · `/api/analysis/kmedoids` · uses `sklearn_extra.cluster.KMedoids(method='pam', random_state=42)`
- **Method**: Ran the CLI script on iris; reproduced with `sklearn_extra` KMedoids and compared labels (adjusted Rand index).
## Summary — 1/1 pass
Cluster labels match a direct `sklearn_extra.cluster.KMedoids(method='pam', init='k-medoids++')` fit exactly (ARI = 1.0).
## Fix applied
`scikit-learn-extra` is not importable under **numpy 2.x** (ABI: "numpy.dtype size changed"). Pinned **`numpy<2`** in `requirements.txt` so k-medoids imports and runs. Validated in a numpy 1.26 environment. Repro: `validation/validate_kmedoids.py` (run under numpy<2).
