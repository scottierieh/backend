# hdbscan clustering (Python) — Validation Report
- **Target**: `hdbscan_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script on iris; reproduced the pipeline (StandardScaler → **hdbscan.HDBSCAN**) and compared cluster labels via adjusted Rand index (ARI = 1.0 → identical) and key metrics.
## Summary — pass
Cluster assignments match a direct `hdbscan.HDBSCAN` fit exactly (ARI = 1.0). Package-based. Repro: `validation/validate_hdbscan.py`.
