"""check_target_encoding.py — the leak, measured, and what closes it.

    python scripts/check_target_encoding.py          # a few seconds

Target encoding replaces a category with the target's mean in that category,
which puts the answer into the question. Fitted on the training rows and
applied to those same rows, every row's feature contains that row's own label.
The damage is not that the model is worse -- it is that the model looks BETTER:
cross-validation reads the label back out of the feature and reports a score
the sealed holdout does not reproduce.

The first block measures exactly that, on a column drawn independently of the
target, by scoring a real model both ways. That gap is the reason
FeatureEngineer.fit_transform() is not fit().transform().

`check_pipeline_parity.py` separately proves this engine and the browser's
produce the same numbers, cell for cell, on all three tables. This is the other
half: that the numbers are the right ones.
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# model_store, in memory, registered before models_api imports it -- the real
# one writes to Cloud Storage. Everything else in models_api is the real module.
# (Same stand-in check_autogluon_serving.py uses.)
_store: dict = {}
_fake = types.ModuleType('model_store')
_fake.save_pipeline = lambda model_id, artifact: (
    _store.__setitem__(f'gs://check/{model_id}', artifact) or f'gs://check/{model_id}')
_fake.load_pipeline = lambda uri: _store[uri]
_fake.delete_artifact = lambda uri: _store.pop(uri, None)
sys.modules['model_store'] = _fake

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

import models_api  # noqa: E402  -- after the stand-in is registered

from feature_pipeline import (  # noqa: E402
    FeatureEngineer, _prepare_target_y, _TARGET_ENCODE_FOLDS,
    _TARGET_ENCODE_SMOOTHING,
)

_ok = 0
_failed = 0


def check(cond, msg, *extra):
    global _ok, _failed
    if cond:
        _ok += 1
        print(f'ok    {msg}')
    else:
        _failed += 1
        print(f'FAIL  {msg}')
        for e in extra:
            print(f'        {e}')


STEP = [{'id': 'te', 'kind': 'target_encode', 'columns': ['sku'], 'keepSource': False}]


def noise_frame(n, levels, seed=20260930):
    """A high-cardinality column drawn with NO relation to the target -- the
    shape of a customer id, a postcode, an SKU. An honest encoding of it
    carries nothing, so any score above chance is the leak talking."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'sku': [f's{i}' for i in rng.integers(0, levels, n)],
        'churn': rng.integers(0, 2, n),
    })


def main():
    # ---- 1. the leak, scored ------------------------------------------
    train = noise_frame(900, 180)
    sealed = noise_frame(300, 180, seed=7)

    fe = FeatureEngineer(STEP, task='classification')
    honest = fe.fit_transform(train[['sku']], train['churn'])
    leaky = fe.transform(train[['sku']])
    held = fe.transform(sealed[['sku']])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    model = LogisticRegression(max_iter=500)

    cv_leaky = cross_val_score(model, leaky, train['churn'], cv=cv, scoring='roc_auc').mean()
    cv_honest = cross_val_score(model, honest, train['churn'], cv=cv, scoring='roc_auc').mean()
    held_auc = roc_auc_score(
        sealed['churn'],
        LogisticRegression(max_iter=500).fit(honest, train['churn']).predict_proba(held)[:, 1])

    print(f'      ROC-AUC on a column that carries NOTHING — truth is 0.500')
    print(f'        cross-validated, re-applied fit : {cv_leaky:.3f}')
    print(f'        cross-validated, out-of-fold    : {cv_honest:.3f}')
    print(f'        sealed holdout, out-of-fold     : {held_auc:.3f}')

    check(cv_leaky > 0.70,
          f're-applying the fit to its own rows scores {cv_leaky:.3f} on noise — '
          f'cross-validation cannot see the leak, because the leak is in the feature')
    check(abs(cv_honest - 0.5) < 0.08,
          f'out-of-fold, the same column scores {cv_honest:.3f} — which is what '
          f'a column carrying nothing should score')
    check(abs(held_auc - 0.5) < 0.12 and abs(cv_honest - held_auc) < 0.12,
          f'and the sealed holdout agrees with it ({held_auc:.3f}), which is the '
          f'whole point: a CV score you can believe')
    check(cv_leaky - held_auc > 0.15,
          f'the gap the leak would have opened between CV and holdout: '
          f'{cv_leaky - held_auc:+.3f}')

    # ---- 2. the two passes really are different ----------------------
    check(not np.allclose(honest['sku_te'].to_numpy(), leaky['sku_te'].to_numpy()),
          'fit_transform() and fit().transform() return different columns, which '
          'is the contract and not an accident')
    fe2 = FeatureEngineer(STEP, task='classification')
    fe2.fit(train[['sku']], train['churn'])
    check(np.allclose(fe2.transform(train[['sku']])['sku_te'].to_numpy(),
                      leaky['sku_te'].to_numpy()),
          'fit() leaves the same fitted state as fit_transform(), so only the '
          'returned table differs')

    # ---- 3. a sklearn Pipeline gets the out-of-fold one --------------
    # Pipeline.fit calls fit_transform on every step but the last. A comment in
    # feature_pipeline.py says so; this is that claim, checked.
    # Asking the FeatureEngineer again would only re-check point 2. What matters
    # is what the ESTIMATOR was handed, so a recorder sits between them and keeps
    # it.
    seen = {}

    class Record:
        def fit(self, X, y=None):
            seen['X'] = X.copy()
            return self

        def transform(self, X):
            return X

    pipe = Pipeline([('fe', FeatureEngineer(STEP, task='classification')),
                     ('rec', Record()),
                     ('clf', LogisticRegression(max_iter=500))])
    pipe.fit(train[['sku']], train['churn'])
    check('X' in seen
          and np.allclose(seen['X']['sku_te'].to_numpy(), honest['sku_te'].to_numpy())
          and not np.allclose(seen['X']['sku_te'].to_numpy(), leaky['sku_te'].to_numpy()),
          'inside a sklearn Pipeline the next step is handed the OUT-OF-FOLD '
          'column, because Pipeline.fit calls fit_transform on its intermediates')

    # ---- 4. what /predict needs, and does not ------------------------
    check(fe.input_columns_ == ['sku'],
          "the target is not an input column, so /predict's `raw_columns` does "
          "not demand a label the caller has no way to supply",
          fe.input_columns_)
    one = fe.transform(pd.DataFrame([{'sku': 's3'}]))
    check('sku_te' in one.columns and pd.notna(one['sku_te'].iloc[0]),
          'and one row with no target transforms fine — the encoding is a '
          'lookup once it is fitted')

    # ---- 5. the arithmetic, against the formula written down ---------
    small = pd.DataFrame({'g': ['x' if i % 2 else 'z' for i in range(20)],
                          'y': list(range(20))})
    reg = FeatureEngineer([{'id': 't', 'kind': 'target_encode', 'columns': ['g'],
                            'keepSource': True}], task='regression')
    out = reg.fit_transform(small[['g']], small['y'])
    others = [(i, v) for i, v in enumerate(small['y']) if i % _TARGET_ENCODE_FOLDS != 0]
    prior = sum(v for _, v in others) / len(others)
    mine = [v for i, v in others if small['g'].iloc[i] == 'z']
    expected = ((sum(mine) + _TARGET_ENCODE_SMOOTHING * prior)
                / (len(mine) + _TARGET_ENCODE_SMOOTHING))
    check(abs(out['g_te'].iloc[0] - expected) < 1e-12,
          f"row 0's value is the m-estimate over the other four folds "
          f'({expected:.4f})', out['g_te'].iloc[0])

    # ---- 6. smoothing, and the fallbacks ----------------------------
    rare = pd.DataFrame({'g': ['common'] * 50 + ['once'],
                         'y': [i % 2 for i in range(50)] + [1]})
    fe3 = FeatureEngineer([{'id': 't', 'kind': 'target_encode', 'columns': ['g'],
                            'keepSource': True}], task='classification')
    fe3.fit(rare[['g']], rare['y'])
    p = fe3.fitted_steps_[0]['params']
    lone, pr = p['teMeans']['once'], p['tePrior']
    check(abs(lone - pr) * 10 < abs(lone - 1.0),
          f'a level seen once encodes to {lone:.3f} — {abs(lone - 1.0) / abs(lone - pr):.0f}x '
          f'nearer the prior {pr:.3f} than its own label 1.000')
    fallback = fe3.transform(pd.DataFrame([{'g': 'never_seen'}, {'g': None}, {'g': ''}]))
    check(np.allclose(fallback['g_te'].to_numpy(), pr),
          'an unseen level, a null and a blank all fall back to the prior — never '
          'to NaN, which would drop the row at predict time',
          list(fallback['g_te']))

    # ---- 7. three classes is not one column -------------------------
    check(_prepare_target_y(['a', 'b', 'c'], 'classification') is None,
          'three classes yields no usable target rather than averaging class codes')
    multi = pd.DataFrame({'g': [f'g{i % 3}' for i in range(30)],
                          'y': [f'c{i % 3}' for i in range(30)]})
    fe4 = FeatureEngineer(STEP[:1], task='classification')
    fe4.steps = [{'id': 't', 'kind': 'target_encode', 'columns': ['g'], 'keepSource': True}]
    got = fe4.fit_transform(multi[['g']], multi['y'])
    check('g_te' not in got.columns,
          'and the step adds no column at all, instead of a plausible wrong one')
    check(_prepare_target_y(['no', 'yes', 'yes'], 'classification') == [0.0, 1.0, 1.0]
          and _prepare_target_y([0, 1, 1], 'classification') == [0.0, 1.0, 1.0],
          'sorted labels, last one counts as 1 — the same rule for "yes"/"no" and 0/1')

    # ---- 8. no target at all ----------------------------------------
    fe5 = FeatureEngineer([{'id': 't', 'kind': 'target_encode', 'columns': ['g'],
                            'keepSource': True}], task='classification')
    bare = fe5.fit_transform(pd.DataFrame({'g': ['a', 'b']}))
    check('g_te' not in bare.columns,
          'with no y passed the step does nothing, rather than encoding against '
          'a target it does not have')

    # ---- 9. the whole way to what gets served ------------------------
    # Everything above is about the engine. This is about the seam: a recipe
    # that reads the target has to survive /train, the artifact, and /predict --
    # where there IS no target and never will be.
    deterministic = [
        {'sku': f'sku-{i % 30}', 'age': 20 + i % 40,
         'churn': 'yes' if (i % 30) % 3 == 0 else 'no'}
        for i in range(300)
    ]
    trained = models_api.train_model('check-te', models_api.TrainRequest(
        data=deterministic, algorithm='Decision Tree', target='churn',
        features=['sku_te', 'age'], task='classification',
        pipeline=[{'id': 't', 'kind': 'target_encode', 'columns': ['sku'],
                   'keepSource': False}]))
    check(trained.get('rawColumns') == ['sku', 'age'],
          "/train reports the recipe's RAW inputs, target excluded — those are "
          'the columns /predict will ask for',
          trained.get('rawColumns'))

    artifact = _store[trained['artifactUri']]
    persisted = artifact['feature_engineer'].fitted_steps_[0]['params']
    check('teMeans' in persisted and 'tePrior' in persisted
          and 'teOof' not in persisted,
          f"the artifact carries the {len(persisted['teMeans'])} level lookups and "
          'NOT the per-row out-of-fold values, which are a fitting detail rather '
          'than state',
          sorted(persisted.keys()))
    check(artifact['feature_engineer'].task == 'classification',
          'and remembers the task it was fitted for, so a reload encodes the '
          'same way')

    served = models_api.predict_model('check-te', models_api.PredictRequest(
        rows=[{'sku': 'sku-3', 'age': 31}, {'sku': 'never-seen', 'age': 31}],
        artifactUri=trained['artifactUri']))
    check(len(served.get('predictions') or []) == 2,
          'and /predict answers for a raw row AND for a level it has never seen, '
          'from the lookup alone',
          served.get('predictions'))
    try:
        models_api.predict_model('check-te', models_api.PredictRequest(
            rows=[{'age': 31}], artifactUri=trained['artifactUri']))
        detail, status = '<no exception>', 0
    except Exception as exc:
        detail, status = str(getattr(exc, 'detail', exc)), getattr(exc, 'status_code', 0)
    check(status == 400 and 'sku' in detail and 'churn' not in detail,
          'and a row without the raw column is a 400 naming sku — never asking '
          'for the target back',
          f'{status} {detail}')

    print(f'\n{_ok} ok, {_failed} failure(s)')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main())
