"""
feature_scoring_analysis.py — what each column is worth, and what it already knows.

Route: /api/analysis/feature-scoring. Model Lab's Features screen asks this
before anything is trained, so the person choosing columns is choosing them on
evidence instead of on the column names.

Two questions at once, and they are the same computation read twice:

  WORTH      how much this one column tells you about the target -- mutual
             information (which sees non-linear and categorical structure that
             a correlation misses) and the univariate F-test beside it.

  ALREADY KNOWS   whether a column is a copy of the answer. The existing
             guardrail (guardrails.py) catches this AFTER a model is trained
             and only through linear correlation, so a leaked categorical --
             an account status that is set when the customer churns, say --
             sails past it and shows up as a model that is too good. Here each
             column is fitted ALONE, cross-validated, and a column that nails
             the target by itself is named before anything is trained on it.

Both readings are guidance for a person, not a score for a model. The response
says which rows it read, because scoring on rows the model will later be
measured on is a small leak of its own: the caller sends the training half when
it has one, and `scored_on` records what arrived.

CLI-script contract, like every *_analysis.py here: one JSON object in on
stdin, one JSON object out on stdout; on error print {"error": ...} to stderr
and exit(1).
"""

import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    f_classif, f_regression, mutual_info_classif, mutual_info_regression,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from analysis_common import _to_native_type

# A constant column makes f_classif warn and corrcoef divide by zero. Both are
# expected here -- finding those columns is half the job -- and both go to
# stderr, which this script's contract reserves for a real failure.
warnings.filterwarnings('ignore')
np.seterr(invalid='ignore', divide='ignore')

# Mutual information estimates a density; past a few thousand rows the answer
# stops moving and the cost does not. The alone-fit below is capped the same
# way, and both are deterministic given the seed.
MAX_ROWS = 5000
# Pairwise redundancy is O(p^2) in the number of columns. Past this the table
# is wide enough that the person is not reading a redundancy list anyway.
MAX_COLS_FOR_REDUNDANCY = 200
SEED = 42

# A column that alone reaches this is not a good feature, it is the answer.
# Deliberately not 1.0: a leaked column usually carries a little noise (a few
# rows corrected by hand, a rounding), and an exact-match rule misses those.
ALONE_LEAK_CLF = 0.98      # ROC-AUC, or accuracy against the majority baseline
ALONE_LEAK_REG = 0.98      # R2
# Below this a column tells you nothing a constant would not.
NEAR_ZERO_MI = 1e-4


def _err(msg):
    print(json.dumps({'error': str(msg)}), file=sys.stderr)
    sys.exit(1)


def _encode(frame: pd.DataFrame):
    """Numeric matrix + a mask of which columns are discrete.

    Categoricals are ordinal-coded, which is meaningless as an ORDER but is
    exactly right for mutual information and for a tree: both split on
    equality of the code, never on its magnitude. The mask is what tells
    mutual_info to treat them that way.
    """
    out = pd.DataFrame(index=frame.index)
    discrete = []
    for col in frame.columns:
        s = frame[col]
        if pd.api.types.is_numeric_dtype(s):
            filled = s.astype(float)
            filled = filled.fillna(filled.median() if filled.notna().any() else 0.0)
            out[col] = filled
            # An integer column with few levels behaves like a category for
            # mutual information, and saying so makes its estimate far less
            # noisy than treating it as a continuum with five points in it.
            discrete.append(filled.nunique() <= 10 and float(filled.dropna().mod(1).abs().max() or 0) == 0)
        else:
            codes = s.astype('string').fillna('__missing__').astype('category').cat.codes
            out[col] = codes.astype(float)
            discrete.append(True)
    return out, np.asarray(discrete, dtype=bool)


def _alone_score(x: np.ndarray, y: np.ndarray, task_type: str, splitter) -> float:
    """How well this ONE column predicts the target, cross-validated.

    A shallow tree, because the question is not "how good a model is this" but
    "does this column contain the answer" -- and a column that is a copy of
    the target is found by the first split. Depth also keeps a continuous
    column from memorising the rows one leaf at a time, which would make every
    high-cardinality column look like a leak.
    """
    X = x.reshape(-1, 1)
    if task_type == 'classification':
        model = DecisionTreeClassifier(max_depth=4, random_state=SEED)
        scoring = 'roc_auc' if len(np.unique(y)) == 2 else 'accuracy'
    else:
        model = DecisionTreeRegressor(max_depth=4, random_state=SEED)
        scoring = 'r2'
    try:
        scores = cross_val_score(model, X, y, cv=splitter, scoring=scoring, error_score=np.nan)
        value = float(np.nanmean(scores))
        return value if np.isfinite(value) else None
    except Exception:
        return None


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        target = payload.get('target_col') or payload.get('target')
        features = payload.get('feature_cols') or payload.get('features')
        task_req = payload.get('task_type', 'auto')
        # What the caller sent: Model Lab seals a test set before anything is
        # trained and sends the training half here, so choosing columns on
        # these scores does not read the rows the final score comes from.
        scored_on = payload.get('scored_on') or 'unspecified'

        if not data or not target or not features:
            raise ValueError('Missing data, target_col, or feature_cols')

        df = pd.DataFrame(data)
        missing = [c for c in [target] + list(features) if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not in the data: {', '.join(missing)}")

        n_input = len(df)
        df = df[df[target].notna()]
        n_target_missing = n_input - len(df)
        if len(df) < 20:
            raise ValueError(f'Not enough rows with a target to score on ({len(df)}, need >= 20)')

        y_raw = df[target]
        if task_req in ('classification', 'regression'):
            task_type = task_req
        else:
            task_type = ('regression'
                         if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique() > 20
                         else 'classification')

        # Every per-column statistic below is computed on the SAME rows, so
        # the ranking is one comparison and not several.
        if len(df) > MAX_ROWS:
            df = df.sample(MAX_ROWS, random_state=SEED)
            y_raw = df[target]
        sampled = len(df) < n_input - n_target_missing

        X_raw = df[list(features)]
        X, discrete = _encode(X_raw)

        if task_type == 'classification':
            y = LabelEncoder().fit_transform(y_raw.astype(str))
            smallest = int(np.min(np.bincount(y)))
            folds = max(2, min(3, smallest))
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
            mi = mutual_info_classif(X, y, discrete_features=discrete, random_state=SEED)
            f_stat, p_val = f_classif(X, y)
            baseline = float(np.max(np.bincount(y)) / len(y))
        else:
            y = pd.to_numeric(y_raw, errors='coerce').to_numpy(dtype=float)
            keep = np.isfinite(y)
            X, y = X[keep], y[keep]
            X_raw = X_raw[keep]
            splitter = KFold(n_splits=3, shuffle=True, random_state=SEED)
            mi = mutual_info_regression(X, y, discrete_features=discrete, random_state=SEED)
            f_stat, p_val = f_regression(X, y)
            baseline = 0.0

        y_num = pd.to_numeric(y_raw, errors='coerce') if task_type == 'regression' else pd.Series(y, index=X.index)

        rows = []
        n_rows = len(X)
        for i, col in enumerate(features):
            values = X[col].to_numpy(dtype=float)
            raw_values = X_raw[col]
            nunique = int(raw_values.nunique(dropna=True))
            top_share = float(raw_values.value_counts(normalize=True, dropna=True).max()) if nunique else 1.0

            alone = _alone_score(values, y, task_type, splitter)
            corr = None
            if values.std() > 0 and np.std(y) > 0:
                c = float(np.corrcoef(values, np.asarray(y, dtype=float))[0, 1])
                corr = c if np.isfinite(c) else None

            flags = []
            # Nothing to learn from: one value, or one value in all but a
            # handful of rows. The variance filter, stated in the units a
            # reader has (share of rows), not as a variance threshold whose
            # scale depends on the column.
            if nunique <= 1:
                flags.append('constant')
            elif top_share >= 0.99:
                flags.append('near_constant')
            # A key: it identifies rows rather than describing them. All
            # values distinct is NOT enough on its own -- a continuous
            # measurement is all-distinct too, and flagging every float column
            # as an identifier is a false positive on the most ordinary kind
            # of feature there is. A key is a label or a whole number.
            if nunique == n_rows and n_rows > 20:
                is_whole = (pd.api.types.is_numeric_dtype(raw_values)
                            and float(pd.to_numeric(raw_values, errors='coerce')
                                      .dropna().mod(1).abs().max() or 0) == 0)
                if is_whole or not pd.api.types.is_numeric_dtype(raw_values):
                    flags.append('identifier_like')
            # An exact copy of the target, whatever its dtype.
            if len(values) == len(y_num) and float(
                    (pd.Series(values).reset_index(drop=True)
                     == pd.Series(np.asarray(y_num)).reset_index(drop=True)).mean()) > 0.999:
                flags.append('target_duplicate')
            elif corr is not None and abs(corr) > 0.97:
                flags.append('leakage_suspect')
            # The reading the linear check cannot make: this column alone
            # answers the question.
            elif alone is not None and (
                    (task_type == 'classification' and alone >= max(ALONE_LEAK_CLF, baseline + 1e-9))
                    or (task_type == 'regression' and alone >= ALONE_LEAK_REG)):
                flags.append('alone_almost_perfect')
            if float(mi[i]) < NEAR_ZERO_MI and 'constant' not in flags:
                flags.append('no_information')

            rows.append({
                'feature': col,
                'mutual_info': _to_native_type(float(mi[i])),
                'f_stat': _to_native_type(float(f_stat[i])) if np.isfinite(f_stat[i]) else None,
                'p_value': _to_native_type(float(p_val[i])) if np.isfinite(p_val[i]) else None,
                'corr_with_target': _to_native_type(corr) if corr is not None else None,
                'alone_score': _to_native_type(alone) if alone is not None else None,
                'n_unique': nunique,
                'top_value_share': _to_native_type(top_share),
                'flags': flags,
            })

        # Rank on mutual information: it is the one score that is comparable
        # across a numeric and a categorical column, which is the whole point
        # of ranking them in one table.
        order = sorted(range(len(rows)), key=lambda i: rows[i]['mutual_info'], reverse=True)
        for rank, i in enumerate(order, start=1):
            rows[i]['rank'] = rank

        # The correlation filter: which columns carry the same thing as an
        # earlier, better-ranked one. Reported, never applied -- two columns
        # at 0.96 can still be two different measurements.
        redundant = {}
        if len(features) <= MAX_COLS_FOR_REDUNDANCY:
            corr_m = np.corrcoef(X.to_numpy(dtype=float).T)
            ranked_names = [rows[i]['feature'] for i in order]
            index_of = {c: i for i, c in enumerate(features)}
            for later_pos, name in enumerate(ranked_names):
                a = index_of[name]
                for earlier in ranked_names[:later_pos]:
                    b = index_of[earlier]
                    c = corr_m[a, b] if corr_m.ndim == 2 else np.nan
                    if np.isfinite(c) and abs(float(c)) >= 0.95:
                        redundant[name] = {'with': earlier, 'corr': _to_native_type(float(c))}
                        break
        for r in rows:
            r['redundant_with'] = redundant.get(r['feature'])

        by_rank = sorted(rows, key=lambda r: r['rank'])
        leaks = [r['feature'] for r in by_rank
                 if {'target_duplicate', 'leakage_suspect', 'alone_almost_perfect'} & set(r['flags'])]
        # Disjoint on purpose. A column cannot be both the answer and nothing,
        # and a reader shown it in both lists learns neither -- so a leak
        # suspect is a leak suspect, whatever else is true of it.
        dead = [r['feature'] for r in by_rank
                if r['feature'] not in leaks
                and {'constant', 'near_constant', 'no_information', 'identifier_like'} & set(r['flags'])]

        print(json.dumps({
            'task_type': task_type,
            'target': target,
            'ranking_metric': 'mutual_info',
            'features': by_rank,
            # Two lists, never merged: one is "this column is the answer" and
            # the other is "this column says nothing". Dropping the first is
            # usually correct and dropping the second is usually harmless, and
            # they are not the same decision.
            'leak_suspects': leaks,
            'no_signal': dead,
            'row_counts': {
                'n_input': n_input,
                'n_target_missing_dropped': n_target_missing,
                'n_scored': n_rows,
                'sampled': bool(sampled),
            },
            # Which rows these scores were read from. Choosing columns on the
            # rows a model is later measured on leaks, mildly but really, so
            # the screen has to be able to say.
            'scored_on': scored_on,
            'baseline': _to_native_type(baseline),
        }, default=_to_native_type))

    except Exception as e:  # noqa: BLE001 — CLI contract: any failure -> stderr + exit(1)
        _err(e)


if __name__ == '__main__':
    main()
