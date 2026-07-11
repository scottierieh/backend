# t-SNE (Python) — Validation Report
- **Target**: `tsne_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script; reproduced the pipeline (StandardScaler → **sklearn.manifold.TSNE**, random_state=42, init='pca', learning_rate='auto') and compared the 2D embedding and KL divergence exactly.
## Summary — pass
Embedding matches a direct `sklearn.manifold.TSNE` fit exactly (max coordinate diff = 0, KL divergence identical to 1e-4). Effective perplexity capping `min(perplexity, max(5,(n-1)/3))` verified. Package-based. Repro: `validation/validate_tsne.py`.
