"""
Kaplan-Meier Survival Curves Router for FastAPI
Non-parametric survival analysis with group comparisons
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, ClassVar
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


# ═══════════════════════════════════════════════════════════════════════
# Request Model
# ═══════════════════════════════════════════════════════════════════════

class KaplanMeierRequest(BaseModel):
    data: List[Dict[str, Any]]
    time_col: str           # Time to event
    event_col: str          # Event indicator (1=event, 0=censored)
    group_col: Optional[str] = None
    confidence_level: float = 0.95   # Actually passed to lifelines alpha
    show_censors: bool = True        # Show censor tick marks on KM plot
    show_ci: bool = True             # Show confidence interval band
    at_risk_counts: bool = True      # Include at-risk table plot in response
    time_unit: str = "days"          # Label unit for axes

    VALID_TIME_UNITS: ClassVar[set] = {
        'days', 'weeks', 'months', 'years', 'cycles', 'hours'
    }

    def validate_inputs(self):
        if not (0.50 <= self.confidence_level <= 0.99):
            raise ValueError("confidence_level must be between 0.50 and 0.99.")
        if self.time_unit not in self.VALID_TIME_UNITS:
            raise ValueError(
                f"Invalid time_unit '{self.time_unit}'. "
                f"Supported: {sorted(self.VALID_TIME_UNITS)}"
            )


# ═══════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════

def _to_native_type(obj):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_list()
    return obj


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return encoded


def _get_palette(n: int):
    """Return n distinct colors; fall back to tab20 for large groups."""
    if n <= 8:
        return plt.cm.Set1(np.linspace(0, 0.85, n))
    return plt.cm.tab20(np.linspace(0, 1, n))


def _warn_annotation(ax, msg: str):
    """Add a yellow warning box annotation to an axes."""
    ax.annotate(
        msg, xy=(0.01, 0.02), xycoords='axes fraction',
        fontsize=9, color='#92400e',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef3c7', alpha=0.85)
    )


# ═══════════════════════════════════════════════════════════════════════
# Plot functions
# ═══════════════════════════════════════════════════════════════════════

def generate_survival_curve_plot(
    kmf_data: Dict[str, Any],
    group_col: Optional[str] = None,
    show_ci: bool = True,
    show_censors: bool = True,
    time_unit: str = "days",
) -> str:
    """Kaplan-Meier survival curve.
    
    Respects show_ci, show_censors, time_unit.
    Annotates warning when n_groups > 8.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    n_groups = len(kmf_data['groups'])
    colors    = _get_palette(n_groups)
    is_multi  = group_col and n_groups > 1

    for (group_name, data), color in zip(kmf_data['groups'].items(), colors):
        times    = np.array(data['time'])
        survival = np.array(data['survival'])

        # Step-function line (KM is piecewise constant)
        ax.step(times, survival, linewidth=2.5, label=str(group_name),
                color=color, where='post')

        # ── show_ci ────────────────────────────────────────────────
        if show_ci:
            ci_lo = np.array(data['ci_lower'])
            ci_hi = np.array(data['ci_upper'])
            ax.fill_between(times, ci_lo, ci_hi,
                            alpha=0.15, color=color, step='post')

        # ── show_censors: + tick marks at censored time points ─────
        if show_censors and data.get('censor_times'):
            ct = np.array(data['censor_times'])
            if len(ct) > 0:
                cs = np.interp(ct, times, survival)
                ax.scatter(ct, cs, marker='+', s=70,
                           color=color, zorder=5, linewidths=1.8,
                           label=f'{group_name} censored' if is_multi else 'Censored')

    ax.set_xlabel(f'Time ({time_unit})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
    ax.set_title(
        'Kaplan-Meier Survival Curves by Group' if is_multi
        else 'Kaplan-Meier Survival Curve',
        fontsize=14, fontweight='bold'
    )
    ax.set_ylim([0, 1.05])
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    if n_groups > 8:
        _warn_annotation(ax, f'⚠ {n_groups} groups — consider filtering for readability')

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_cumulative_events_plot(
    kmf_data: Dict[str, Any],
    group_col: Optional[str] = None,
    time_unit: str = "days",
) -> str:
    """Cumulative events (incidence) over time."""
    fig, ax = plt.subplots(figsize=(12, 7))

    n_groups = len(kmf_data['groups'])
    colors   = _get_palette(n_groups)
    is_multi = group_col and n_groups > 1

    for (group_name, data), color in zip(kmf_data['groups'].items(), colors):
        # cumulative observed events from event_table
        cum_events = [data['event_table'][t] for t in data['time']]
        ax.plot(data['time'], cum_events, linewidth=2.5,
                label=str(group_name), marker='o', markersize=4, color=color)

    ax.set_xlabel(f'Time ({time_unit})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Number of Events', fontsize=12, fontweight='bold')
    ax.set_title(
        'Cumulative Events by Group' if is_multi else 'Cumulative Events Over Time',
        fontsize=14, fontweight='bold'
    )
    ax.grid(True, linestyle='--', alpha=0.3)
    if n_groups > 1:
        ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_cumulative_hazard_plot(
    kmf_data: Dict[str, Any],
    group_col: Optional[str] = None,
    time_unit: str = "days",
) -> str:
    """Nelson-Aalen cumulative hazard H(t) = -log S(t)."""
    fig, ax = plt.subplots(figsize=(12, 7))

    n_groups = len(kmf_data['groups'])
    colors   = _get_palette(n_groups)
    is_multi = group_col and n_groups > 1

    for (group_name, data), color in zip(kmf_data['groups'].items(), colors):
        survival = np.clip(np.array(data['survival']), 1e-10, 1.0)
        cumhaz   = -np.log(survival)
        ax.step(data['time'], cumhaz, linewidth=2.5,
                label=str(group_name), color=color, where='post')

    ax.set_xlabel(f'Time ({time_unit})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Hazard H(t)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Cumulative Hazard Function by Group' if is_multi
        else 'Cumulative Hazard Function',
        fontsize=14, fontweight='bold'
    )
    ax.grid(True, linestyle='--', alpha=0.3)
    if n_groups > 1:
        ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_at_risk_table_plot(
    kmf_data: Dict[str, Any],
    time_unit: str = "days",
) -> str:
    """Number-at-risk heatmap table."""
    fig, ax = plt.subplots(figsize=(12, max(3, len(kmf_data['groups']) * 0.8 + 2)))

    all_times = sorted(set(
        t for data in kmf_data['groups'].values() for t in data['time']
    ))
    time_points = all_times[::max(1, len(all_times) // 10)]

    group_names = list(kmf_data['groups'].keys())
    table_data  = []
    for gname in group_names:
        d     = kmf_data['groups'][gname]
        times = d['time']
        row   = []
        for tp in time_points:
            idx = min(range(len(times)), key=lambda i: abs(times[i] - tp))
            row.append(d['at_risk'][idx])
        table_data.append(row)

    im = ax.imshow(table_data, cmap='YlGn', aspect='auto', interpolation='nearest')
    ax.set_xticks(range(len(time_points)))
    ax.set_xticklabels([f'{int(t)}' for t in time_points], rotation=45, ha='right')
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names)
    ax.set_xlabel(f'Time ({time_unit})', fontsize=11, fontweight='bold')
    ax.set_title('Number at Risk Over Time', fontsize=13, fontweight='bold')

    # Minor grid for cell borders
    ax.set_xticks(np.arange(len(time_points)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(group_names)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)

    for i in range(len(group_names)):
        for j in range(len(time_points)):
            ax.text(j, i, str(int(table_data[i][j])),
                    ha='center', va='center', fontsize=9, color='black')

    fig.colorbar(im, ax=ax, label='At Risk')
    plt.tight_layout()
    return _fig_to_base64(fig)


# ═══════════════════════════════════════════════════════════════════════
# Statistical tests
# ═══════════════════════════════════════════════════════════════════════

def perform_logrank_test(T1, E1, T2, E2,
                         group1_name: str, group2_name: str) -> Dict[str, Any]:
    """Log-rank test for 2 groups."""
    test = logrank_test(T1, T2, E1, E2)
    return {
        'test_name':      'Log-rank test',
        'test_statistic': _to_native_type(test.test_statistic),
        'p_value':        _to_native_type(test.p_value),
        'degrees_of_freedom': 1,
        'significant':    bool(test.p_value < 0.05),
        'interpretation': (
            f'Significant survival difference between {group1_name} and {group2_name}'
            if test.p_value < 0.05
            else f'No significant survival difference (p = {test.p_value:.4f})'
        )
    }


def perform_multivariate_logrank_test(
    T_all: pd.Series,
    E_all: pd.Series,
    groups: pd.Series,
) -> Dict[str, Any]:
    """Multivariate log-rank test for ≥ 3 groups."""
    result = multivariate_logrank_test(T_all, groups, E_all)
    return {
        'test_name':      'Multivariate log-rank test',
        'test_statistic': _to_native_type(result.test_statistic),
        'p_value':        _to_native_type(result.p_value),
        'degrees_of_freedom': int(len(groups.unique()) - 1),
        'significant':    bool(result.p_value < 0.05),
        'interpretation': (
            f'Significant overall survival difference across groups (p = {result.p_value:.4f})'
            if result.p_value < 0.05
            else f'No significant overall survival difference (p = {result.p_value:.4f})'
        )
    }


# ═══════════════════════════════════════════════════════════════════════
# Median survival  (lifelines-based, labelled as estimate)
# ═══════════════════════════════════════════════════════════════════════

def calculate_median_survival(kmf_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return median survival time per group.
    
    Uses lifelines' median_survival_times_ where available.
    Fields use *_estimate suffix to avoid implying formal CI.
    """
    result = {}
    for group_name, data in kmf_data['groups'].items():
        # Prefer lifelines median if stored
        if data.get('lifelines_median') is not None:
            median = _to_native_type(data['lifelines_median'])
        else:
            # Fallback: first time S(t) ≤ 0.5
            survival = np.array(data['survival'])
            idx      = np.where(survival <= 0.5)[0]
            median   = _to_native_type(data['time'][idx[0]]) if len(idx) > 0 else None

        result[group_name] = {
            'median_survival':        median,
            'median_reached':         median is not None,
            # Formal median CI requires Brookmeyer-Crowley — not estimated here
            'median_ci_note':         'Formal CI not estimated (Brookmeyer-Crowley method needed)',
        }
    return result


# ═══════════════════════════════════════════════════════════════════════
# Survival table
# ═══════════════════════════════════════════════════════════════════════

def generate_survival_table(kmf_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return sampled survival statistics table (~20 rows per group)."""
    rows = []
    for group_name, data in kmf_data['groups'].items():
        step = max(1, len(data['time']) // 20)
        for i, t in enumerate(data['time']):
            if i % step == 0:
                rows.append({
                    'group':    group_name,
                    'time':     _to_native_type(t),
                    'at_risk':  _to_native_type(data['at_risk'][i]),
                    'events':   _to_native_type(data['event_table'][t]),
                    'survival': _to_native_type(data['survival'][i]),
                    'ci_lower': _to_native_type(data['ci_lower'][i]),
                    'ci_upper': _to_native_type(data['ci_upper'][i]),
                })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# Interpretation
# ═══════════════════════════════════════════════════════════════════════

def generate_interpretation(
    kmf_data: Dict[str, Any],
    logrank_results: Optional[Dict] = None,
    time_unit: str = "days",
    group_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate interpretation with group comparison support."""
    insights = []

    n_groups       = len(kmf_data['groups'])
    total_events   = sum(d['total_events'] for d in kmf_data['groups'].values())
    total_subjects = kmf_data['n_subjects']
    censoring_rate = (total_subjects - total_events) / total_subjects

    # ── 1. Study population ────────────────────────────────────────
    insights.append({
        'title':       'Study Population',
        'description': (
            f'N = {total_subjects} subjects; {total_events} events '
            f'({total_events/total_subjects:.1%}); '
            f'{total_subjects - total_events} censored ({censoring_rate:.1%}).'
        ),
        'status': 'neutral'
    })

    # ── 2. Median survival ─────────────────────────────────────────
    medians = calculate_median_survival(kmf_data)
    med_parts = []
    for g, m in medians.items():
        if m['median_reached']:
            med_parts.append(f"{g}: {m['median_survival']} {time_unit}")
        else:
            med_parts.append(f"{g}: not reached")
    insights.append({
        'title':       'Median Survival Time',
        'description': ', '.join(med_parts),
        'status':      'neutral'
    })

    # ── 3. Group comparison ────────────────────────────────────────
    if n_groups == 2 and logrank_results:
        status = 'positive' if logrank_results['significant'] else 'neutral'
        insights.append({
            'title':       'Log-rank Test (2 groups)',
            'description': logrank_results['interpretation'],
            'status':      status
        })
    elif n_groups >= 3 and logrank_results:
        status = 'positive' if logrank_results['significant'] else 'neutral'
        insights.append({
            'title':       f'Multivariate Log-rank Test ({n_groups} groups)',
            'description': logrank_results['interpretation'],
            'status':      status
        })

    # ── 4. Censoring note ──────────────────────────────────────────
    if censoring_rate > 0.5:
        insights.append({
            'title':       'High Censoring Rate',
            'description': (
                f'{censoring_rate:.1%} of subjects are censored. '
                'KM estimates may be unstable at later time points.'
            ),
            'status': 'warning'
        })

    # ── 5. Small group warnings ────────────────────────────────────
    if group_warnings:
        for warn in group_warnings:
            insights.append({
                'title':       'Small Group Warning',
                'description': warn,
                'status':      'warning'
            })

    rec_map = {
        True: (
            'Statistically significant survival differences detected. '
            'Consider Cox proportional hazards regression to adjust for covariates.'
        ),
        False: (
            'No significant survival difference detected. '
            'Check assumptions and consider increasing sample size.'
        )
    }
    has_sig = logrank_results.get('significant', False) if logrank_results else False

    return {
        'key_insights':   insights,
        'recommendation': rec_map[has_sig]
    }


# ═══════════════════════════════════════════════════════════════════════
# Main endpoint
# ═══════════════════════════════════════════════════════════════════════

@router.post("/survival-curves")
async def run_kaplan_meier_analysis(request: KaplanMeierRequest) -> Dict[str, Any]:
    """
    Perform Kaplan-Meier survival analysis.

    Supports:
    - Survival curve estimation with proper confidence_level passed to lifelines
    - show_ci / show_censors / at_risk_counts actually control output
    - Group comparisons: log-rank (2 groups) or multivariate log-rank (≥ 3 groups)
    - time_unit reflected in all plot axes and interpretation
    - Median survival using lifelines' built-in estimate (no fake CI)
    - Per-group minimum sample / event validation
    """
    try:
        # ── Validate request ────────────────────────────────────────
        try:
            request.validate_inputs()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not request.data:
            raise HTTPException(status_code=400, detail="Data not provided.")

        df        = pd.DataFrame(request.data)
        time_col  = request.time_col
        event_col = request.event_col
        group_col = request.group_col

        # ── Column validation ───────────────────────────────────────
        required = [time_col, event_col] + ([group_col] if group_col else [])
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing)}")

        # ── Prepare T / E ───────────────────────────────────────────
        T = pd.to_numeric(df[time_col],  errors='coerce')
        E = pd.to_numeric(df[event_col], errors='coerce').astype('Int64')

        valid_mask = (T > 0) & (E.isin([0, 1]))
        T = T[valid_mask].reset_index(drop=True)
        E = E[valid_mask].reset_index(drop=True)

        groups = (
            df.loc[valid_mask, group_col].astype(str).reset_index(drop=True)
            if group_col else None
        )

        if len(T) < 10:
            raise HTTPException(status_code=400, detail="At least 10 valid observations required.")

        # lifelines uses alpha = 1 - confidence_level
        alpha = 1.0 - request.confidence_level

        # ── Fit KM per group ────────────────────────────────────────
        kmf_data = {
            'groups':         {},
            'n_subjects':     len(T),
            'total_events':   int(E.sum()),
            'total_censored': int((E == 0).sum()),
        }

        group_warnings = []
        unique_groups  = sorted(groups.unique()) if groups is not None else ['Overall']

        for gname in unique_groups:
            if groups is not None:
                mask    = groups == gname
                T_g     = T[mask]
                E_g     = E[mask]
            else:
                T_g, E_g = T, E

            n_g      = len(T_g)
            n_events = int(E_g.sum())

            # Per-group validation warnings
            if n_g < 5:
                group_warnings.append(
                    f"Group '{gname}' has only {n_g} subjects — estimates may be unreliable."
                )
            if n_events == 0:
                group_warnings.append(
                    f"Group '{gname}' has 0 events — survival curve will be flat (uninformative)."
                )

            kmf = KaplanMeierFitter(alpha=alpha)
            kmf.fit(T_g, E_g, label=str(gname))

            # Censor times: where event == 0
            censor_times = T_g[E_g == 0].tolist()

            # lifelines median (robust)
            lifelines_median = (
                _to_native_type(kmf.median_survival_time_)
                if not (np.isnan(kmf.median_survival_time_) or np.isinf(kmf.median_survival_time_))
                else None
            )

            kmf_data['groups'][str(gname)] = {
                'time':             _to_native_type(kmf.survival_function_.index.values),
                'survival':         _to_native_type(kmf.survival_function_.values.flatten()),
                'ci_lower':         _to_native_type(kmf.confidence_interval_.iloc[:, 0].values),
                'ci_upper':         _to_native_type(kmf.confidence_interval_.iloc[:, 1].values),
                'at_risk':          _to_native_type(kmf.event_table.at_risk.values),
                'event_table':      {
                    _to_native_type(t): _to_native_type(kmf.event_table.loc[t, 'observed'])
                    for t in kmf.event_table.index
                },
                'censor_times':     censor_times,        # for show_censors
                'total_events':     n_events,
                'n_subjects':       n_g,
                'lifelines_median': lifelines_median,
            }

        # ── Log-rank / multivariate log-rank ────────────────────────
        logrank_results = None
        n_unique = len(unique_groups)

        if groups is not None and n_unique >= 2:
            if n_unique == 2:
                g1, g2   = unique_groups[0], unique_groups[1]
                mask1    = groups == g1
                mask2    = groups == g2
                logrank_results = perform_logrank_test(
                    T[mask1], E[mask1], T[mask2], E[mask2], g1, g2
                )
            else:
                # ≥ 3 groups → multivariate log-rank
                logrank_results = perform_multivariate_logrank_test(T, E, groups)

        # ── Generate plots ──────────────────────────────────────────
        survival_plot  = generate_survival_curve_plot(
            kmf_data, group_col,
            show_ci       = request.show_ci,
            show_censors  = request.show_censors,
            time_unit     = request.time_unit,
        )
        hazard_plot    = generate_cumulative_hazard_plot(
            kmf_data, group_col, time_unit=request.time_unit
        )
        events_plot    = generate_cumulative_events_plot(
            kmf_data, group_col, time_unit=request.time_unit
        )
        # at_risk_counts controls whether we generate this plot
        atrisk_plot    = (
            generate_at_risk_table_plot(kmf_data, time_unit=request.time_unit)
            if request.at_risk_counts else None
        )

        # ── Tables & interpretation ─────────────────────────────────
        survival_table = generate_survival_table(kmf_data)
        medians        = calculate_median_survival(kmf_data)

        interpretation = generate_interpretation(
            kmf_data, logrank_results,
            time_unit      = request.time_unit,
            group_warnings = group_warnings,
        )

        # ── Build group_results ─────────────────────────────────────
        group_results = []
        for gname, gdata in kmf_data['groups'].items():
            med = medians[gname]
            group_results.append({
                'label':            gname,
                'n_subjects':       gdata['n_subjects'],
                'n_events':         gdata['total_events'],
                'n_censored':       gdata['n_subjects'] - gdata['total_events'],
                'censoring_rate':   float(
                    (gdata['n_subjects'] - gdata['total_events']) / gdata['n_subjects'] * 100
                ),
                # median using lifelines estimate — no fabricated CI
                'median_survival':  med['median_survival'],
                'median_reached':   med['median_reached'],
                'median_ci_note':   med['median_ci_note'],
                'timeline':         gdata['time'],
                'survival_prob':    gdata['survival'],
                'ci_lower':         gdata['ci_lower'],
                'ci_upper':         gdata['ci_upper'],
            })

        # ── Format logrank for response ─────────────────────────────
        logrank_formatted = None
        if logrank_results:
            logrank_formatted = {
                **logrank_results,
                'groups_compared': list(kmf_data['groups'].keys()),
            }

        # ── Response ────────────────────────────────────────────────
        response = {
            'n_subjects':         kmf_data['n_subjects'],
            'n_events':           kmf_data['total_events'],
            'n_censored':         kmf_data['total_censored'],
            'event_rate':         float(kmf_data['total_events'] / kmf_data['n_subjects'] * 100),
            'confidence_level':   request.confidence_level,
            'time_unit':          request.time_unit,
            'has_groups':         groups is not None and n_unique > 1,
            'group_col':          group_col,
            'n_groups':           n_unique,
            'group_results':      group_results,
            'group_warnings':     group_warnings,
            'logrank_test':       logrank_formatted,
            'survival_table':     survival_table,
            'km_plot':            survival_plot,
            'hazard_plot':        hazard_plot,
            'events_plot':        events_plot,
            'survival_table_plot': atrisk_plot,   # None if at_risk_counts=False
            'interpretation':     interpretation,
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Survival analysis failed: {str(e)}")
