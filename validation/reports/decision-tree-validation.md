# decision-tree (Python) — Validation Report
- **Target**: `decision_tree_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script on iris; reproduced its pipeline and compared test accuracy to a direct `sklearn.tree.DecisionTreeClassifier` fit.
## Summary — pass
`metrics.accuracy` matches a direct `sklearn.tree.DecisionTreeClassifier` fit on the same stratified split. Package-based. Repro: `validation/validate_decision_tree.py`.

## Defect found & fixed
`decision_tree_analysis.py:425` referenced an undefined `y_test_encoded` (the variable is `y_test_enc`), which crashed the **multiclass** classification path (macro-AUC). Fixed to `y_test_enc`.
