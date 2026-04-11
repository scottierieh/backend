"""
Multidimensional Scaling (MDS) Router for FastAPI
Visualize similarity/dissimilarity data in low-dimensional space
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import io
import base64
import logging
import warnings
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# ── FIX 1: warnings 전역 무시 제거 → 로거로 캡처 ──────────────────────────────
logger = logging.getLogger(__name__)

# MDS/sklearn 내부 ConvergenceWarning만 캡처해서 logger.warning으로 전달
class _WarningToLogger(logging.Handler):
    pass

def _capture_warnings():
    """warnings를 억제 대신 logger로 리디렉션"""
    logging.captureWarnings(True)          # warnings → py.warnings 로거
    py_warnings_logger = logging.getLogger("py.warnings")
    if not py_warnings_logger.handlers:
        py_warnings_logger.addHandler(logging.StreamHandler())
    py_warnings_logger.setLevel(logging.WARNING)

_capture_warnings()

# ── FIX 2: 전역 rcParams 제거 → plot 함수 내부 context로 이동 ─────────────────
_PLOT_RC = {
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
}

router = APIRouter()

# ── FIX 5: 표준화 옵션화 ──────────────────────────────────────────────────────
class MDSRequest(BaseModel):
    data: List[Dict[str, Any]]
    variables: List[str]           # 거리 계산에 사용할 변수
    label_col: Optional[str] = None
    n_dimensions: int = 2          # 2 or 3
    metric: bool = True            # True = metric MDS, False = non-metric
    distance_metric: str = "euclidean"   # euclidean | manhattan | cosine | correlation
    n_init: int = 10
    max_iter: int = 300
    standardize: bool = True       # FIX 5: 표준화 여부 선택 가능
    # FIX 6: scree plot 옵션
    compute_stress_scree: bool = True
    stress_scree_max_dims: int = 6


def _to_native_type(obj):
    """numpy/pandas 타입을 JSON 직렬화 가능한 Python 타입으로 변환"""
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
    return obj


def _fig_to_base64(fig) -> str:
    """matplotlib figure → base64 문자열"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


# ── FIX 3: correlation distance 계산 수정 ─────────────────────────────────────
def compute_distance_matrix(df: pd.DataFrame, variables: List[str], metric: str) -> np.ndarray:
    """
    객체 간 pairwise 거리 행렬 계산.

    correlation 옵션:
      - 각 행(관측치)을 하나의 프로파일 벡터로 보고
        두 관측치 프로파일 간의 피어슨 상관을 거리로 변환
        distance = 1 - pearson_r(row_i, row_j)
      - np.corrcoef(data) 는 기본적으로 행 간 상관을 계산하므로
        data 가 n_samples × n_variables 일 때 의도와 일치.
        단, 이 점을 명시적으로 처리한다.
    """
    data = df[variables].values  # shape: (n_samples, n_variables)

    if metric == "euclidean":
        return pairwise_distances(data, metric='euclidean')

    elif metric == "manhattan":
        return pairwise_distances(data, metric='manhattan')

    elif metric == "cosine":
        return pairwise_distances(data, metric='cosine')

    elif metric == "correlation":
        # 명시적으로 행 간(row-wise) 피어슨 상관 계산
        # np.corrcoef 는 (n_samples × n_variables) 입력 시 행 벡터 간 상관 행렬 반환
        n_samples = data.shape[0]
        corr_matrix = np.corrcoef(data)          # (n_samples, n_samples)
        if corr_matrix.shape != (n_samples, n_samples):
            raise ValueError(
                f"corrcoef 결과가 ({n_samples},{n_samples})가 아닙니다. "
                "데이터 형태를 확인하세요."
            )
        distances = 1.0 - corr_matrix
        distances = np.clip(distances, 0.0, 2.0)  # 수치 오차 방어
        return distances

    else:
        logger.warning("알 수 없는 distance_metric '%s' → euclidean 사용", metric)
        return pairwise_distances(data, metric='euclidean')


def perform_mds(distance_matrix: np.ndarray, n_dimensions: int, metric: bool,
                n_init: int, max_iter: int) -> Dict[str, Any]:
    """
    MDS 분석 수행.

    stress normalization 전략:
      - sklearn >= 1.2 의 normalized_stress=True 옵션을 사용.
        내부적으로 sqrt(stress / sum(d²)) 와 동일하지만,
        sklearn이 직접 관리하므로 버전 간 일관성이 보장됨.
      - mds.stress_ 는 normalized 값(normalized_stress=True 시),
        raw_stress_ 는 비정규화 원값으로 별도 보존.
    """
    mds = MDS(
        n_components=n_dimensions,
        metric=metric,
        dissimilarity='precomputed',
        n_init=n_init,
        max_iter=max_iter,
        random_state=42,
        normalized_stress=True,    # sklearn >= 1.2: Kruskal stress-1 정규화
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coordinates = mds.fit_transform(distance_matrix)
        for w in caught:
            logger.warning("MDS 수렴 경고: %s", w.message)

    normalized_stress = mds.stress_   # normalized_stress=True 이므로 이미 정규화된 값
    # raw stress 도 보존 (역산: normalized = sqrt(raw / (sum(d²)/2)))
    denom = np.sum(distance_matrix ** 2) / 2
    raw_stress = (normalized_stress ** 2) * denom if denom > 0 else np.nan

    return {
        'coordinates':        coordinates,
        'stress':             _to_native_type(raw_stress),
        'normalized_stress':  _to_native_type(normalized_stress),
        'n_iter':             mds.n_iter_ if hasattr(mds, 'n_iter_') else None,
    }


# ── FIX 4: non-metric MDS일 때 disparity 기반 해석 분리 ──────────────────────
def compute_fit_statistics(
    original_distances: np.ndarray,
    coordinates: np.ndarray,
    is_metric: bool = True,
) -> Dict[str, Any]:
    """
    MDS 해의 적합 통계 계산.

    metric=True  → 원거리 vs MDS거리 직접 비교 (metric 해석)
    metric=False → monotone regression 을 통한 disparity 기반 계산 (non-metric 해석)
    """
    mds_distances = pairwise_distances(coordinates, metric='euclidean')

    n = original_distances.shape[0]
    orig_flat = original_distances[np.triu_indices(n, k=1)]
    mds_flat = mds_distances[np.triu_indices(n, k=1)]

    # 상관 / R²
    correlation = float(np.corrcoef(orig_flat, mds_flat)[0, 1])
    r_squared = correlation ** 2

    if is_metric:
        # ── Metric: 원거리 = disparity ──
        disparities = orig_flat
        stress_1 = float(np.sqrt(
            np.sum((mds_flat - disparities) ** 2) / np.sum(disparities ** 2)
        ))
        mean_disp = float(np.mean(disparities))
        stress_2 = float(np.sqrt(
            np.sum((mds_flat - disparities) ** 2) /
            np.sum((disparities - mean_disp) ** 2)
        ))
        disparity_note = "metric: disparities = original distances"

    else:
        # ── Non-metric: isotonic regression으로 disparities 추정 ──
        # 단조 증가 제약을 만족하는 optimal disparities 계산
        try:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
            disparities = iso.fit_transform(orig_flat, mds_flat)
        except Exception:
            # fallback: rank 기반 단조 변환
            order = np.argsort(orig_flat)
            disparities = np.empty_like(orig_flat)
            disparities[order] = np.sort(mds_flat)

        stress_1 = float(np.sqrt(
            np.sum((mds_flat - disparities) ** 2) / np.sum(disparities ** 2)
        ))
        mean_disp = float(np.mean(disparities))
        stress_2 = float(np.sqrt(
            np.sum((mds_flat - disparities) ** 2) /
            np.sum((disparities - mean_disp) ** 2)
        ))
        disparity_note = "non-metric: disparities via isotonic regression (monotone)"

    return {
        'correlation': _to_native_type(correlation),
        'r_squared': _to_native_type(r_squared),
        'stress_1': _to_native_type(stress_1),
        'stress_2': _to_native_type(stress_2),
        'disparity_method': disparity_note,
    }


# ── FIX 2: plot 함수마다 rc context 사용 ─────────────────────────────────────
def generate_mds_plot_2d(coordinates: np.ndarray, labels: List[str], stress: float) -> str:
    with mpl.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(12, 10))

        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1],
                             s=100, c=range(len(coordinates)), cmap='viridis',
                             alpha=0.7, edgecolors='white', linewidth=1)

        for i, label in enumerate(labels):
            ax.annotate(str(label)[:15], (coordinates[i, 0], coordinates[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title(f'MDS Configuration (Stress = {stress:.4f})', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.colorbar(scatter, ax=ax, label='Point Index')
        plt.tight_layout()
        return _fig_to_base64(fig)


def generate_mds_plot_3d(coordinates: np.ndarray, labels: List[str], stress: float) -> str:
    with mpl.rc_context(_PLOT_RC):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
                             s=100, c=range(len(coordinates)), cmap='viridis',
                             alpha=0.7, edgecolors='white', linewidth=1)

        for i, label in enumerate(labels):
            ax.text(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2],
                    str(label)[:10], fontsize=8, alpha=0.8)

        ax.set_xlabel('Dimension 1', fontsize=11)
        ax.set_ylabel('Dimension 2', fontsize=11)
        ax.set_zlabel('Dimension 3', fontsize=11)
        ax.set_title(f'3D MDS Configuration (Stress = {stress:.4f})', fontsize=14, fontweight='bold')

        plt.colorbar(scatter, ax=ax, label='Point Index', shrink=0.6)
        plt.tight_layout()
        return _fig_to_base64(fig)


def generate_shepard_plot(original_distances: np.ndarray, coordinates: np.ndarray,
                          is_metric: bool = True) -> str:
    with mpl.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(10, 8))

        mds_distances = pairwise_distances(coordinates, metric='euclidean')
        n = original_distances.shape[0]
        orig_flat = original_distances[np.triu_indices(n, k=1)]
        mds_flat = mds_distances[np.triu_indices(n, k=1)]

        ax.scatter(orig_flat, mds_flat, alpha=0.5, s=30, c='steelblue', edgecolors='white',
                   label='Observations')

        z = np.polyfit(orig_flat, mds_flat, 1)
        p = np.poly1d(z)
        x_line = np.linspace(orig_flat.min(), orig_flat.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Linear fit')

        max_val = max(orig_flat.max(), mds_flat.max())
        ax.plot([0, max_val], [0, max_val], 'g-', linewidth=1, alpha=0.5, label='Perfect fit')

        # Non-metric일 때 disparity 선 추가
        if not is_metric:
            try:
                from sklearn.isotonic import IsotonicRegression
                iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
                disparities = iso.fit_transform(orig_flat, mds_flat)
                sort_idx = np.argsort(orig_flat)
                ax.plot(orig_flat[sort_idx], disparities[sort_idx], 'm-',
                        linewidth=1.5, alpha=0.7, label='Disparity (isotonic)')
            except Exception:
                pass

        corr = np.corrcoef(orig_flat, mds_flat)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}\nR² = {corr**2:.3f}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Original Distances', fontsize=12)
        ax.set_ylabel('MDS Distances', fontsize=12)
        ax.set_title('Shepard Diagram', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        return _fig_to_base64(fig)


def generate_distance_heatmap(distance_matrix: np.ndarray, labels: List[str]) -> str:
    with mpl.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(12, 10))

        display_labels = [str(l)[:12] for l in labels[:50]]
        display_matrix = distance_matrix[:50, :50]

        sns.heatmap(display_matrix, annot=len(display_labels) <= 15, fmt='.2f',
                    xticklabels=display_labels, yticklabels=display_labels,
                    cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Distance'})

        ax.set_title('Distance Matrix Heatmap', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        return _fig_to_base64(fig)


# ── FIX 6: scree plot에 객체 수 경고 + 서브샘플 옵션 ─────────────────────────
_SCREE_OBJECT_WARN_THRESHOLD = 100
_SCREE_OBJECT_MAX = 200

def generate_stress_scree_plot(
    distance_matrix: np.ndarray,
    max_dims: int = 6,
    n_objects: int = 0,
) -> str:
    """
    차원별 stress scree plot.

    객체 수가 크면 계산이 오래 걸릴 수 있어:
    - _SCREE_OBJECT_WARN_THRESHOLD 초과 시 logger.warning 발생
    - _SCREE_OBJECT_MAX 초과 시 서브샘플링 후 계산
    """
    actual_n = distance_matrix.shape[0]

    if actual_n > _SCREE_OBJECT_WARN_THRESHOLD:
        logger.warning(
            "stress scree plot: 객체 수 %d개로 계산이 느릴 수 있습니다.", actual_n
        )

    if actual_n > _SCREE_OBJECT_MAX:
        logger.warning(
            "stress scree plot: 객체 수 %d개 > %d → 서브샘플 %d개로 계산",
            actual_n, _SCREE_OBJECT_MAX, _SCREE_OBJECT_MAX,
        )
        rng = np.random.default_rng(42)
        idx = rng.choice(actual_n, size=_SCREE_OBJECT_MAX, replace=False)
        distance_matrix = distance_matrix[np.ix_(idx, idx)]

    with mpl.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(10, 6))

        dimensions = list(range(1, min(max_dims + 1, distance_matrix.shape[0])))
        stress_values = []

        for n_dim in dimensions:
            mds_obj = MDS(
                n_components=n_dim, metric=True, dissimilarity='precomputed',
                n_init=2, max_iter=200, random_state=42, normalized_stress=True,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                mds_obj.fit(distance_matrix)
                for w in caught:
                    logger.warning("scree MDS 경고 (dim=%d): %s", n_dim, w.message)

            stress_values.append(float(mds_obj.stress_))  # normalized_stress=True → 이미 정규화

        ax.plot(dimensions, stress_values, 'bo-', linewidth=2, markersize=10)
        ax.axhline(y=0.2, color='red',    linestyle='--', alpha=0.7, label='Poor (0.2)')
        ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Fair (0.1)')
        ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='Good (0.05)')

        subtitle = ""
        if actual_n > _SCREE_OBJECT_MAX:
            subtitle = f" [서브샘플 {_SCREE_OBJECT_MAX}/{actual_n}]"

        ax.set_xlabel('Number of Dimensions', fontsize=12)
        ax.set_ylabel('Stress', fontsize=12)
        ax.set_title(f'Stress vs Dimensions (Scree Plot){subtitle}', fontsize=14, fontweight='bold')
        ax.set_xticks(dimensions)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        return _fig_to_base64(fig)


def interpret_stress(stress: float) -> str:
    if stress <= 0.025:
        return "Excellent"
    elif stress <= 0.05:
        return "Good"
    elif stress <= 0.1:
        return "Fair"
    elif stress <= 0.2:
        return "Poor"
    else:
        return "Very Poor"


def generate_interpretation(
    mds_result: Dict, fit_stats: Dict, n_points: int,
    n_dimensions: int, metric: bool,
) -> Dict[str, Any]:
    stress = mds_result['normalized_stress'] or 0
    stress_quality = interpret_stress(stress)

    key_insights = []

    key_insights.append({
        'title': 'Model Fit',
        'description': (
            f"Stress = {stress:.4f} ({stress_quality}). " +
            ("The MDS solution accurately represents the original distances."
             if stress <= 0.1
             else "Consider adding more dimensions or checking data quality.")
        ),
    })

    r_sq = fit_stats.get('r_squared', 0) or 0
    key_insights.append({
        'title': 'Variance Explained',
        'description': f"R² = {r_sq:.3f}. {r_sq*100:.1f}% of the variance in original distances is captured.",
    })

    key_insights.append({
        'title': 'Configuration',
        'description': (
            f"{n_points} objects mapped to {n_dimensions}D space "
            f"using {'metric' if metric else 'non-metric'} MDS."
        ),
    })

    # FIX 4: non-metric일 때 disparity 해석 별도 안내
    if not metric:
        key_insights.append({
            'title': 'Non-metric Interpretation',
            'description': (
                "Non-metric MDS: stress는 isotonic regression 기반 disparity로 계산됩니다. "
                "원거리의 순위 구조가 MDS 배치에 반영됩니다."
            ),
        })

    if stress > 0.1:
        key_insights.append({
            'title': 'Recommendation',
            'description': "High stress suggests adding dimensions or that the data doesn't have a clear low-dimensional structure.",
        })

    return {
        'stress_quality': stress_quality,
        'key_insights': key_insights,
        'overall_fit': 'Good' if stress <= 0.1 and r_sq >= 0.8 else 'Acceptable' if stress <= 0.2 else 'Poor',
    }


# ── FIX 7: 강화된 validation ─────────────────────────────────────────────────
def _validate_request(df: pd.DataFrame, variables: List[str],
                      label_col: Optional[str], n_dimensions: int) -> List[str]:
    """
    운영용 추가 validation.
    경고(warning) 레벨 메시지 리스트를 반환하고,
    치명적 문제는 HTTPException으로 직접 raise.
    """
    warnings_list: List[str] = []

    # 변수 분산 0 체크
    zero_var_cols = [col for col in variables if df[col].std(ddof=0) == 0]
    if zero_var_cols:
        raise HTTPException(
            status_code=400,
            detail=f"분산이 0인 변수 제거 필요: {', '.join(zero_var_cols)}"
        )

    # 너무 많은 객체 경고
    n_objects = len(df)
    if n_objects > 500:
        warnings_list.append(
            f"객체 수 {n_objects}개: MDS 계산이 O(n²) 이상으로 느릴 수 있습니다."
        )

    # 중복 객체 경고 (모든 변수값이 동일한 행)
    dup_count = df.duplicated(subset=variables).sum()
    if dup_count > 0:
        warnings_list.append(
            f"중복 관측치 {dup_count}개 발견. 거리 행렬에 0이 포함될 수 있습니다."
        )

    # label_col 결측 처리 경고
    if label_col and label_col in df.columns:
        null_label_count = df[label_col].isna().sum()
        if null_label_count > 0:
            warnings_list.append(
                f"label_col '{label_col}'에 결측값 {null_label_count}개 → 자동으로 'NA_i' 대체됩니다."
            )

    return warnings_list


@router.post("/mds")
async def run_mds_analysis(request: MDSRequest) -> Dict[str, Any]:
    """
    Multidimensional Scaling (MDS) 분석.

    고차원 데이터 또는 거리 행렬을 2D/3D 공간으로 투영하여
    개체 간 거리 구조를 시각화합니다.
    """
    try:
        data          = request.data
        variables     = request.variables
        label_col     = request.label_col
        n_dimensions  = request.n_dimensions
        metric        = request.metric
        distance_metric = request.distance_metric
        n_init        = request.n_init
        max_iter      = request.max_iter
        standardize   = request.standardize

        # ── 기본 validation ──
        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")

        if len(variables) < 2:
            raise HTTPException(status_code=400, detail="At least 2 variables required.")

        if n_dimensions not in [2, 3]:
            raise HTTPException(status_code=400, detail="n_dimensions must be 2 or 3.")

        df = pd.DataFrame(data)

        missing_cols = [col for col in variables if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing_cols)}")

        for col in variables:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df_clean = df.dropna(subset=variables).copy()

        if len(df_clean) < 4:
            raise HTTPException(status_code=400, detail="At least 4 complete observations required.")

        # ── FIX 7: 강화된 validation (label_col 결측 전 수행) ──
        validation_warnings = _validate_request(df_clean, variables, label_col, n_dimensions)
        for w in validation_warnings:
            logger.warning("MDS validation: %s", w)

        # ── labels (label_col 결측 → 자동 대체) ──
        if label_col and label_col in df_clean.columns:
            raw_labels = df_clean[label_col].copy()
            null_mask = raw_labels.isna()
            raw_labels[null_mask] = [f"NA_{i}" for i in np.where(null_mask)[0]]
            labels = raw_labels.astype(str).tolist()
        else:
            labels = [f"Obj_{i+1}" for i in range(len(df_clean))]

        # ── FIX 5: 표준화 조건부 적용 ──
        if standardize:
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(df_clean[variables])
            df_scaled = pd.DataFrame(scaled_values, columns=variables)
        else:
            df_scaled = df_clean[variables].reset_index(drop=True)

        # ── 거리 행렬 계산 ──
        distance_matrix = compute_distance_matrix(df_scaled, variables, distance_metric)

        # ── MDS 수행 ──
        mds_result  = perform_mds(distance_matrix, n_dimensions, metric, n_init, max_iter)
        coordinates = mds_result['coordinates']

        # ── FIX 4: metric 여부를 fit_statistics에 전달 ──
        fit_stats = compute_fit_statistics(distance_matrix, coordinates, is_metric=metric)

        # ── 시각화 ──
        norm_stress = mds_result['normalized_stress'] or 0

        if n_dimensions == 2:
            mds_plot = generate_mds_plot_2d(coordinates, labels, norm_stress)
        else:
            mds_plot = generate_mds_plot_3d(coordinates, labels, norm_stress)

        # FIX 4: Shepard에도 metric 여부 전달
        shepard_plot    = generate_shepard_plot(distance_matrix, coordinates, is_metric=metric)
        distance_heatmap = generate_distance_heatmap(distance_matrix, labels)

        # FIX 6: scree plot toggle + 객체 수 인자 전달
        stress_scree = None
        if request.compute_stress_scree:
            stress_scree = generate_stress_scree_plot(
                distance_matrix,
                max_dims=request.stress_scree_max_dims,
                n_objects=len(df_clean),
            )

        interpretation = generate_interpretation(
            mds_result, fit_stats, len(df_clean), n_dimensions, metric
        )

        # ── 좌표 출력 ──
        coord_output = []
        for i, label in enumerate(labels):
            point = {
                'label': label,
                'dim1': _to_native_type(coordinates[i, 0]),
                'dim2': _to_native_type(coordinates[i, 1]),
            }
            if n_dimensions == 3:
                point['dim3'] = _to_native_type(coordinates[i, 2])
            coord_output.append(point)

        return {
            'coordinates':        coord_output,
            'stress':             mds_result['stress'],
            'normalized_stress':  mds_result['normalized_stress'],
            'stress_quality':     interpret_stress(norm_stress),
            'fit_statistics':     fit_stats,
            'distance_metric':    distance_metric,
            'n_dimensions':       n_dimensions,
            'metric_mds':         metric,
            'standardized':       standardize,
            'n_objects':          len(df_clean),
            'n_variables':        len(variables),
            'mds_plot':           mds_plot,
            'shepard_plot':       shepard_plot,
            'distance_heatmap':   distance_heatmap,
            'stress_scree':       stress_scree,
            'interpretation':     interpretation,
            'validation_warnings': validation_warnings,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("MDS analysis failed")
        raise HTTPException(status_code=500, detail=f"MDS analysis failed: {str(e)}")
