# decision-tree (Python) — Validation Report
- **Target**: `decision_tree_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script on iris; reproduced its pipeline and compared test accuracy to a direct `sklearn.tree.DecisionTreeClassifier` fit.
## Summary — pass
`metrics.accuracy` matches a direct `sklearn.tree.DecisionTreeClassifier` fit on the same stratified split. Package-based. Repro: `validation/validate_decision_tree.py`.

## Defect found & fixed
`decision_tree_analysis.py:425` referenced an undefined `y_test_encoded` (the variable is `y_test_enc`), which crashed the **multiclass** classification path (macro-AUC). Fixed to `y_test_enc`.

## Exploration-tab payloads (55 checks in total)
- `threshold_curve` (binary breast-cancer run, 10 features): equals `sklearn.metrics.roc_curve(drop_intermediate=False)` on the test rows and positive-class probabilities of an independently reproduced model (same split, scaling and class weighting) — every threshold/FPR/TPR, `n_pos`/`n_neg`, labels, `tpr·n_pos` and `fpr·n_neg` integer-valued, and the curve's AUC equals the reported `metrics.auc`.
- `tree_structure`: every row (iris, breast cancer, diabetes depth 6) is walked down the returned node arrays and the leaf matches the reproduced tree — argmax = `predict`, leaf proportions = `predict_proba`, leaf mean = `predict`; an unlimited-depth regression tree is truncated to ≤ 255 shown nodes whose values equal `tree_.value`.
