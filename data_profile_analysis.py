"""
Dataset Profiling — CLI script

Backs Model Lab's Data screen ("can we run a model on this data?"). Given a
table and an optional target column, returns everything that screen displays:
per-column kinds and statistics, dataset-level missing/duplicate/outlier rates,
the target's problem type and class balance, and a readiness score with its
deductions itemised.

This is the only place those numbers are computed. The frontend parses the CSV
and renders the response; it does not re-derive any statistic, so there is one
implementation to keep correct rather than two that can drift.

Follows the stdin/stdout CLI contract used by main.py's generic script runner:
one JSON object in, one JSON object out, errors to stderr with exit(1).

Request
-------
{
  "data":   [ {col: value, ...}, ... ],   # required, non-empty
  "target": "churn",                      # optional; omitted => suggested
  "histogram_bins": 24                    # optional, default 24
}

Response
--------
{
  "row_count", "column_count",
  "columns": [ {name, kind, missing, missing_pct, unique, mean, std, min, max,
                q1, q3, outliers, histogram} ],
  "missing_pct", "duplicate_pct", "outlier_pct",
  "kind_counts": {numeric, categorical, date, id},
  "suggested_target",
  "target": {column, kind, classes: [{value, count, pct}], minority_pct} | null,
  "readiness": {score, penalties: [{id, points, detail}]}
}
"""

import sys
import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from analysis_common import _to_native_type

# A column whose name says "identifier" is one even when a few rows repeat.
ID_NAME = re.compile(r'(^|_)(id|uuid|guid|key|no|code)$', re.I)
DATE_NAME = re.compile(r'(date|time|_at$|^dt_|timestamp)', re.I)
# Dates must carry a separator, or a bare year parses as one.
DATE_VALUE = re.compile(r'^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}')

# Above this share of distinct values a column is "one value per row".
ID_UNIQUE_RATIO = 0.95
# At or below this many distinct values a numeric target is still a class label.
MAX_CLASSES = 15
# Enough shares to describe a column at a glance; the rest folds into 'other'
# rather than becoming a legend nobody reads.
TOP_CATEGORIES = 6
# Under this many complete pairs, a column-to-target figure is noise wearing
# three decimal places. The screen draws "—" instead.
MIN_RELATION_ROWS = 20
# A name that says the value is about a person rather than their behaviour.
# Matched on the name alone, so this flags rather than excludes: whether the
# column belongs in the model is the reader's call, not a regex's.
#
# Age is deliberately not here. It is a protected attribute in some settings
# and an ordinary predictor in most, and a mark on every table that has an age
# column is a mark nobody reads by the third screen. The ones below are the
# ones whose presence in a model is a question worth stopping on.
SENSITIVE_NAME = re.compile(
    r'(^|_)(gender|sex|race|ethnic\w*|religion|disab\w*|pregnan\w*|marital|'
    r'nationality|성별|인종|종교|장애|결혼|국적)($|_)', re.I)


def _infer_kind(name: str, series: pd.Series) -> str:
    """numeric | categorical | date | id, inferred from values rather than headers."""
    present = series.dropna()
    if present.empty:
        return 'categorical'

    n = len(present)
    unique = present.nunique()
    numeric = pd.api.types.is_numeric_dtype(present)

    # An identifier is (near enough) one value per row. A high-cardinality
    # *numeric* column with no id-ish name stays numeric -- a continuous
    # measurement is legitimately unique in almost every row.
    if n > 1 and unique / n > ID_UNIQUE_RATIO:
        if ID_NAME.search(name):
            return 'id'
        if not numeric:
            return 'id'

    if not numeric:
        as_text = present.astype(str)
        date_like = as_text.str.match(DATE_VALUE).sum()
        if date_like / n > 0.9:
            return 'date'
        if DATE_NAME.search(name) and date_like > 0:
            return 'date'

    return 'numeric' if numeric else 'categorical'


def _histogram(values: pd.Series, lower: float, upper: float, bins: int) -> Dict[str, Any]:
    """Bin counts plus the Tukey fences, so the UI can shade what it reports."""
    lo = float(values.min())
    hi = float(values.max())
    if hi == lo:
        return {
            'bins': [{'from': lo, 'to': hi, 'count': int(len(values))}],
            'lower_fence': lower,
            'upper_fence': upper,
        }

    counts, edges = np.histogram(values.to_numpy(dtype=float), bins=bins, range=(lo, hi))
    return {
        'bins': [
            {'from': float(edges[i]), 'to': float(edges[i + 1]), 'count': int(counts[i])}
            for i in range(len(counts))
        ],
        'lower_fence': lower,
        'upper_fence': upper,
    }


def _relation(series: pd.Series, kind: str, y: pd.Series,
              target_kind: str) -> Optional[Dict[str, Any]]:
    """
    How much this column has to say about the target, in the terms the Data
    screen reads out loud.

    Three shapes, because one number cannot describe both a measurement and a
    set of labels:

      auc        a numeric column against a two-class target, folded to >= 0.5.
                 0.50 is "knows nothing"; 1.00 would separate the classes
                 perfectly, which in real data usually means the column was
                 written down AFTER the outcome was known.
      rate_range a categorical column against a two-class target: the positive
                 rate of its own levels. "9.4% ~ 14.7%" says what a correlation
                 coefficient cannot -- which value is the risky one.
      corr       anything against a numeric target: |Spearman|, which catches a
                 monotone relationship that is not a straight line.

    Returns None rather than a zero when the question does not apply (the
    target itself, an identifier, a column with one value, too few rows). The
    screen draws "—" there; a zero would read as "measured, and unrelated".
    """
    pair = pd.DataFrame({'x': series, 'y': y}).dropna()
    if len(pair) < MIN_RELATION_ROWS or pair['x'].nunique() < 2:
        return None

    if target_kind == 'regression':
        x = pd.to_numeric(pair['x'], errors='coerce') if kind != 'categorical' else None
        if x is not None:
            ok = x.notna()
            if ok.sum() < MIN_RELATION_ROWS or x[ok].nunique() < 2:
                return None
            rho = x[ok].corr(pair['y'][ok], method='spearman')
            return None if pd.isna(rho) else {'kind': 'corr', 'value': abs(float(rho))}
        groups = pair.groupby(pair['x'].astype(str))['y']
        means = groups.mean()
        if len(means) < 2:
            return None
        return {
            'kind': 'mean_range',
            'min': float(means.min()),
            'max': float(means.max()),
            'levels': _levels(groups.mean(), groups.size()),
        }

    # Two classes: the positive one is the minority, which is the one the
    # screen is about ("이탈률"). Multiclass gets no single rate to quote.
    classes = pair['y'].astype(str).value_counts()
    if len(classes) != 2:
        return None
    positive = str(classes.index[-1])
    hit = (pair['y'].astype(str) == positive)

    if kind == 'categorical' or not pd.api.types.is_numeric_dtype(pair['x']):
        groups = hit.groupby(pair['x'].astype(str))
        rates = groups.mean() * 100
        if len(rates) < 2:
            return None
        return {
            'kind': 'rate_range',
            'positive': positive,
            'min': float(rates.min()),
            'max': float(rates.max()),
            'levels': _levels(rates, groups.size()),
        }

    x = pd.to_numeric(pair['x'], errors='coerce')
    ok = x.notna()
    if ok.sum() < MIN_RELATION_ROWS or hit[ok].nunique() < 2:
        return None
    # Mann-Whitney U / (n1*n2) is the ROC AUC, and needs no sklearn import:
    # the rank sum of the positive class says how often a positive row
    # outranks a negative one, which is what the area under the curve is.
    ranks = x[ok].rank()
    pos = hit[ok]
    n1 = int(pos.sum())
    n0 = int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return None
    u = float(ranks[pos].sum()) - n1 * (n1 + 1) / 2
    auc = u / (n1 * n0)
    return {'kind': 'auc', 'value': max(auc, 1.0 - auc)}


def _levels(rates: pd.Series, sizes: pd.Series) -> List[Dict[str, Any]]:
    """The per-level figures behind a range, biggest level first, capped."""
    out = []
    for value, rate in rates.items():
        out.append({
            'value': str(value),
            'rate': float(rate),
            'count': int(sizes.get(value, 0)),
        })
    out.sort(key=lambda d: d['count'], reverse=True)
    return out[:TOP_CATEGORIES]


def _profile_columns(df: pd.DataFrame, bins: int) -> List[Dict[str, Any]]:
    rows = len(df)
    out: List[Dict[str, Any]] = []

    for name in df.columns:
        series = df[name]
        kind = _infer_kind(str(name), series)
        missing = int(series.isna().sum())

        col: Dict[str, Any] = {
            'name': str(name),
            'kind': kind,
            'missing': missing,
            'missing_pct': (missing / rows * 100) if rows else 0.0,
            'unique': int(series.nunique(dropna=True)),
            # Flagged, never dropped: a model that must not use gender is a
            # decision about the model, and this script does not get to make
            # it. The screen shows the mark and leaves the column checked.
            'sensitive': bool(SENSITIVE_NAME.search(str(name))),
        }

        # Category shares and a date range. Without these the Data screen's
        # ledger has an empty Distribution cell for every non-numeric column,
        # and it will not fill one in locally: a second implementation of the
        # same statistic is a second thing that can disagree with this one.
        if kind == 'categorical':
            counts = series.dropna().astype(str).value_counts()
            total = int(counts.sum())
            if total:
                head = counts.head(TOP_CATEGORIES)
                col['top'] = [
                    {'value': str(v), 'count': int(c), 'pct': c / total * 100}
                    for v, c in head.items()
                ]
                rest = total - int(head.sum())
                if rest > 0:
                    col['other'] = {'count': rest, 'pct': rest / total * 100}

        if kind == 'date':
            parsed = pd.to_datetime(series, errors='coerce').dropna()
            if not parsed.empty:
                col['first'] = parsed.min().isoformat()
                col['last'] = parsed.max().isoformat()

        if kind == 'numeric':
            nums = pd.to_numeric(series, errors='coerce').dropna()
            if not nums.empty:
                q1 = float(nums.quantile(0.25))
                q3 = float(nums.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = int(((nums < lower) | (nums > upper)).sum()) if iqr > 0 else 0

                col.update({
                    'mean': float(nums.mean()),
                    # Fisher-Pearson skew. The Features screen suggests a log or
                    # Yeo-Johnson from this rather than from a rule of thumb, and
                    # computing it in the browser would be a second
                    # implementation of the same number.
                    'skew': float(nums.skew()) if len(nums) > 2 else 0.0,
                    # Population sd, matching how the rate is described to the reader.
                    'std': float(nums.std(ddof=0)),
                    'min': float(nums.min()),
                    'max': float(nums.max()),
                    'q1': q1,
                    'q3': q3,
                    'outliers': outliers,
                    'histogram': _histogram(nums, lower if iqr > 0 else None,
                                            upper if iqr > 0 else None, bins),
                })

        out.append(col)

    return out


def _suggest_target(columns: List[Dict[str, Any]]) -> Optional[str]:
    usable = [c for c in columns if c['kind'] not in ('id', 'date')]
    if not usable:
        return None
    for c in usable:
        if re.fullmatch(r'churn|target|label|y|outcome|class', c['name'], re.I):
            return c['name']
    binary = [c for c in usable if c['unique'] == 2]
    if binary:
        return binary[-1]['name']
    return usable[-1]['name']


def _profile_target(df: pd.DataFrame, column: str,
                    columns: List[Dict[str, Any]]) -> Dict[str, Any]:
    meta = next((c for c in columns if c['name'] == column), None)
    values = df[column].dropna()

    # A numeric column with many distinct values is a regression target; a
    # numeric flag with two levels is still classification.
    if meta and meta['kind'] == 'numeric' and meta['unique'] > MAX_CLASSES:
        return {'column': column, 'kind': 'regression', 'classes': [], 'minority_pct': None}

    counts = values.astype(str).value_counts()
    total = int(counts.sum())
    classes = [
        {'value': str(v), 'count': int(c), 'pct': (c / total * 100) if total else 0.0}
        for v, c in counts.items()
    ]

    return {
        'column': column,
        'kind': 'binary' if len(classes) == 2 else 'multiclass',
        'classes': classes,
        'minority_pct': classes[-1]['pct'] if classes else None,
    }


def _readiness(missing_pct: float, duplicate_pct: float, outlier_pct: float,
               target: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    One 0-100 number for "can we model this yet?", with the deductions kept so
    the screen can show why rather than asking the reader to trust it.
    """
    penalties = []

    missing = min(25.0, missing_pct)
    if missing > 0.05:
        penalties.append({'id': 'missing', 'points': missing, 'detail': f'{missing_pct:.1f}%'})

    duplicate = min(15.0, duplicate_pct * 3)
    if duplicate > 0.05:
        penalties.append({'id': 'duplicate', 'points': duplicate, 'detail': f'{duplicate_pct:.1f}%'})

    outlier = min(15.0, outlier_pct * 1.5)
    if outlier > 0.05:
        penalties.append({'id': 'outlier', 'points': outlier, 'detail': f'{outlier_pct:.1f}%'})

    # Imbalance only counts below a 30% minority share; balanced classes cost nothing.
    if target and target.get('minority_pct') is not None and target['minority_pct'] < 30:
        minority = target['minority_pct']
        penalties.append({
            'id': 'imbalance',
            'points': min(25.0, (30 - minority) * 0.8),
            'detail': f'{minority:.0f}:{100 - minority:.0f}',
        })

    total = sum(p['points'] for p in penalties)
    return {'score': max(0, round(100 - total)), 'penalties': penalties}


def main():
    try:
        payload = json.load(sys.stdin)

        data = payload.get('data')
        if not data:
            raise ValueError('data is required and must be a non-empty array of rows')

        df = pd.DataFrame(data)
        if df.empty or len(df.columns) == 0:
            raise ValueError('data has no rows or no columns')

        # Empty strings arrive from CSV cells that were blank; they are missing
        # values, not the string "".
        df = df.replace('', np.nan)

        bins = int(payload.get('histogram_bins') or 24)
        rows = len(df)

        columns = _profile_columns(df, bins)

        total_cells = rows * len(df.columns)
        missing_cells = sum(c['missing'] for c in columns)

        # Duplicates are exact whole-row repeats, compared on every column.
        duplicate_rows = int(df.duplicated().sum())

        numeric_cols = [c for c in columns if c['kind'] == 'numeric']
        numeric_cells = sum(rows - c['missing'] for c in numeric_cols)
        outlier_cells = sum(c.get('outliers', 0) for c in numeric_cols)

        kind_counts = {'numeric': 0, 'categorical': 0, 'date': 0, 'id': 0}
        for c in columns:
            kind_counts[c['kind']] += 1

        suggested = _suggest_target(columns)
        requested = payload.get('target') or suggested
        target = (
            _profile_target(df, requested, columns)
            if requested and requested in df.columns
            else None
        )

        # Relation to the target, in a second pass: it is the one per-column
        # figure that cannot be computed until the target is known, and the
        # target is only settled here.
        if target:
            y = df[target['column']]
            for c in columns:
                # Identifiers and dates get none. An id has a different value
                # per row, so any figure is about the row order. A raw date is
                # one level per day, which produces a 2%~31% spread read off
                # a handful of rows each -- the engine turns dates into month
                # and weekday before it models them, and those are the columns
                # worth a relation, not the timestamp.
                if c['name'] == target['column'] or c['kind'] in ('id', 'date'):
                    continue
                c['relation'] = _relation(df[c['name']], c['kind'], y, target['kind'])

        missing_pct = (missing_cells / total_cells * 100) if total_cells else 0.0
        duplicate_pct = (duplicate_rows / rows * 100) if rows else 0.0
        outlier_pct = (outlier_cells / numeric_cells * 100) if numeric_cells else 0.0

        response = {
            'row_count': rows,
            'column_count': int(len(df.columns)),
            'columns': columns,
            'missing_pct': missing_pct,
            'duplicate_pct': duplicate_pct,
            # The count as well as the share: the screen says "중복 14행",
            # and rounding a percentage back into rows is how a 14 becomes
            # a 13.
            'duplicate_rows': duplicate_rows,
            'outlier_pct': outlier_pct,
            'kind_counts': kind_counts,
            'suggested_target': suggested,
            'target': target,
            'readiness': _readiness(missing_pct, duplicate_pct, outlier_pct, target),
        }

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
