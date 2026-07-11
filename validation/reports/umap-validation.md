# UMAP (Python) — Validation Report
- **Target**: `umap_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script; reproduced the pipeline (StandardScaler → **umap.UMAP**, random_state=42) and compared the 2D embedding exactly.
## Summary — pass
Embedding matches a direct `umap.UMAP` fit exactly (max coordinate diff = 0). With `random_state` set, UMAP runs single-threaded and is fully deterministic. Effective `n_neighbors = min(15, max(2, n-1))` verified. Package (`umap-learn`) already pinned in `requirements.txt`. Repro: `validation/validate_umap.py`.
