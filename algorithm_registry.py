"""
algorithm_registry.py — maps an Auto Compare algorithm label to a real,
persistable sklearn-compatible estimator.

register-model.ts sends `algorithm` verbatim from the winning run's label —
the exact strings in src/lib/model-lab/predictive-registry.ts's `autoCompare.label`
fields on the frontend (Random Forest, XGBoost, GBM, ...). It sends no
hyperparameters (Auto Compare's preview scripts run with their own defaults
and are never persisted — see models_api.py's docstring), so the estimators
built here use reasonable defaults of their own; they don't need to be
identical to the discarded preview run's, since the persisted model's own
reported metrics (computed in models_api.py after fitting) are what the
registry and the user actually see and judge it by.
"""

from typing import Literal

Task = Literal['classification', 'regression']


def build_estimator(algorithm: str, task: Task):
    """Return a fresh, unfitted estimator for `algorithm` + `task`. Raises
    ValueError for an algorithm/task combination this backend can't serve
    (e.g. Naive Bayes for regression — it's classification-only on the
    frontend's own lineup, so this should never actually be reached, but a
    clear error beats a confusing one if the lineup ever changes)."""
    clf = task == 'classification'

    if algorithm == 'Random Forest':
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        return RandomForestClassifier(n_estimators=300, random_state=42) if clf \
            else RandomForestRegressor(n_estimators=300, random_state=42)

    if algorithm == 'XGBoost':
        from xgboost import XGBClassifier, XGBRegressor
        return XGBClassifier(n_estimators=300, eval_metric='logloss', random_state=42) if clf \
            else XGBRegressor(n_estimators=300, random_state=42)

    if algorithm == 'LightGBM':
        from lightgbm import LGBMClassifier, LGBMRegressor
        return LGBMClassifier(n_estimators=300, random_state=42, verbosity=-1) if clf \
            else LGBMRegressor(n_estimators=300, random_state=42, verbosity=-1)

    if algorithm == 'CatBoost':
        from catboost import CatBoostClassifier, CatBoostRegressor
        return CatBoostClassifier(iterations=300, random_state=42, verbose=False) if clf \
            else CatBoostRegressor(iterations=300, random_state=42, verbose=False)

    if algorithm == 'GBM':
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        return GradientBoostingClassifier(n_estimators=200, random_state=42) if clf \
            else GradientBoostingRegressor(n_estimators=200, random_state=42)

    if algorithm == 'AdaBoost':
        from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
        return AdaBoostClassifier(n_estimators=200, random_state=42) if clf \
            else AdaBoostRegressor(n_estimators=200, random_state=42)

    if algorithm == 'Decision Tree':
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        return DecisionTreeClassifier(random_state=42) if clf else DecisionTreeRegressor(random_state=42)

    if algorithm == 'Support Vector Machine (SVM)':
        from sklearn.svm import SVC, SVR
        return SVC(probability=True, random_state=42) if clf else SVR()

    if algorithm == 'K-Nearest Neighbors (KNN)':
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        return KNeighborsClassifier() if clf else KNeighborsRegressor()

    if algorithm == 'Naive Bayes':
        if not clf:
            raise ValueError('Naive Bayes is classification-only')
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()

    if algorithm == 'Discriminant Analysis (LDA)':
        if not clf:
            raise ValueError('LDA is classification-only')
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis()

    if algorithm == 'Elastic Net Regression':
        if clf:
            raise ValueError('Elastic Net Regression is regression-only')
        from sklearn.linear_model import ElasticNet
        return ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)

    if algorithm == 'Artificial Neural Network (MLP)':
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42) if clf \
            else MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

    if algorithm == 'Voting / Stacking Ensemble':
        # No base_estimators selection reaches this endpoint (Auto Compare's
        # own ensemble step picks those for the *preview* run only) — build a
        # small, fixed, generally-solid voting ensemble instead.
        from sklearn.ensemble import (
            VotingClassifier, VotingRegressor, RandomForestClassifier, RandomForestRegressor,
            GradientBoostingClassifier, GradientBoostingRegressor,
        )
        if clf:
            from sklearn.linear_model import LogisticRegression
            return VotingClassifier(estimators=[
                ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=150, random_state=42)),
                ('lr', LogisticRegression(max_iter=1000)),
            ], voting='soft')
        from sklearn.linear_model import Ridge
        return VotingRegressor(estimators=[
            ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=150, random_state=42)),
            ('ridge', Ridge()),
        ])

    raise ValueError(f'Unknown algorithm: {algorithm!r}')
