# hca clustering (Python) — Validation Report
- **Target**: `hca_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script on iris; reproduced the pipeline (StandardScaler → **scipy.cluster.hierarchy (ward/fcluster)**) and compared cluster labels via adjusted Rand index (ARI = 1.0 → identical) and key metrics.
## Summary — pass
Cluster assignments match a direct `scipy.cluster.hierarchy (ward/fcluster)` fit exactly (ARI = 1.0). Package-based. Repro: `validation/validate_hca.py`.
