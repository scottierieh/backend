# mlp (Python) — Validation Report
- **Target**: `mlp_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script; reproduced its computation with **sklearn.neural_network.MLPClassifier** and compared the key result.
## Summary — pass (44 PASS, 0 FAIL)
Result matches a direct reproduction with sklearn.neural_network.MLPClassifier. Repro: `validation/validate_mlp.py`.

Beyond the generic classifier metrics, the validator now independently recomputes and cross-checks the MLP method-specific blocks the backend emits (the ones surfaced on the STEP6 page):
- **Training / convergence**: `n_iterations` (model.n_iter_) and `final_loss` (model.loss_)
- **Per-class metrics**: precision / recall / f1 / support per class, in LabelEncoder class order
- **Macro ROC-AUC**: one-vs-rest macro average over predict_proba
- **Cross-validation**: `cv_mean`, `cv_std`, `cv_folds`, and each fold score (StratifiedKFold shuffle, rs=42, accuracy)
- **Permutation feature importance**: per-feature mean importance and the descending rank ordering
