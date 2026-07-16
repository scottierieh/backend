# decision-tree (Python) — Validation Report
- **Target**: `decision_tree_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script on iris; reproduced its pipeline and compared test accuracy to a direct `sklearn.tree.DecisionTreeClassifier` fit.
## Summary — pass (48 checks, 0 fail)
`metrics.accuracy` matches a direct `sklearn.tree.DecisionTreeClassifier` fit on the same stratified split. Package-based. Repro: `validation/validate_decision_tree.py`.

Beyond the generic classifier metrics, the validator now independently recomputes and cross-checks the method-specific blocks a STEP6 page renders:
- **tree_info** — `n_nodes`, `max_depth_actual`, `n_leaves` vs the reproduced `model.tree_`.
- **feature_importance** — raw Gini `importance`, `normalized_importance` (imp/max), `importance_pct` (imp×100), and the descending `rank` ordering vs `model.feature_importances_`.
- **cv_results** — `cv_folds`, per-fold `cv_scores`, `cv_mean`, `cv_std` vs a direct 5-fold `cross_val_score` (accuracy) on the full dataset.
- **tree_rules** — leaf count (`n_leaves`) from the rule extractor vs `model.get_n_leaves()`.

## Defect found & fixed
`decision_tree_analysis.py:425` referenced an undefined `y_test_encoded` (the variable is `y_test_enc`), which crashed the **multiclass** classification path (macro-AUC). Fixed to `y_test_enc`.
