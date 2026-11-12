# mlp (Python) — Validation Report
- **Target**: `mlp_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script; reproduced its computation with **sklearn.neural_network.MLPClassifier** and compared the key result.
## Summary — pass
Result matches a direct reproduction with sklearn.neural_network.MLPClassifier. Repro: `validation/validate_mlp.py`.

## Exploration-tab payloads (30 checks in total)
- `threshold_curve` (binary breast-cancer run, 10 features): equals `sklearn.metrics.roc_curve(drop_intermediate=False)` on the test rows and positive-class probabilities of an independently reproduced model (same split, scaling and class weighting) — every threshold/FPR/TPR, `n_pos`/`n_neg`, labels, `tpr·n_pos` and `fpr·n_neg` integer-valued, and the curve's AUC equals the reported `metrics.auc`.
