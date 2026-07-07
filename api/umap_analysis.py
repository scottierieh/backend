"""
UMAP (Uniform Manifold Approximation and Projection) Router for FastAPI
Non-linear dimensionality reduction, faster and more globally-structure-preserving than t-SNE
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class UMAPRequest(BaseModel):
    data: List[Dict[str, Any]]
    feature_cols: List[str]
    label_col: Optional[str] = None
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 42


def _to_native_type(obj):
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
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def generate_embedding_plot(embedding: np.ndarray, labels: Optional[pd.Series], label_col: Optional[str]) -> str:
    fig, ax = plt.subplots(figsize=(9, 8))
    if labels is not None:
        unique_labels = labels.unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for lbl, color in zip(unique_labels, colors):
            mask = labels == lbl
            ax.scatter(embedding[mask, 0], embedding[mask, 1], color=color, label=str(lbl), alpha=0.75, s=40, edgecolor='white', linewidth=0.3)
        ax.legend(title=label_col, bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], color='#3b82f6', alpha=0.7, s=40, edgecolor='white', linewidth=0.3)
    ax.set_xlabel('UMAP Dimension 1', fontsize=11)
    ax.set_ylabel('UMAP Dimension 2', fontsize=11)
    ax.set_title('UMAP Embedding', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(n_neighbors_used: int, n_neighbors_requested: int, min_dist: float,
                             silhouette: Optional[float], label_col: Optional[str]) -> Dict[str, Any]:
    key_insights = []

    if n_neighbors_used != n_neighbors_requested:
        key_insights.append({
            'title': 'n_neighbors Adjusted',
            'description': f'Requested n_neighbors ({n_neighbors_requested}) was reduced to {n_neighbors_used} to stay below the sample size.',
            'status': 'neutral'
        })

    key_insights.append({
        'title': 'Local vs. Global Structure',
        'description': f'n_neighbors={n_neighbors_used} controls the local/global tradeoff: smaller values emphasize fine local structure, larger values emphasize broader global structure.',
        'status': 'neutral'
    })

    key_insights.append({
        'title': 'Cluster Tightness',
        'description': f'min_dist={min_dist} controls how tightly points are packed; lower values produce tighter, more separated clusters, higher values preserve a more even spread.',
        'status': 'neutral'
    })

    if silhouette is not None:
        sil_desc = 'well separated' if silhouette >= 0.5 else 'moderately separated' if silhouette >= 0.25 else 'overlapping'
        key_insights.append({
            'title': 'Cluster Separation',
            'description': f"Silhouette score of the embedding w.r.t. '{label_col}' = {silhouette:.3f} — groups appear {sil_desc} in the projection.",
            'status': 'positive' if silhouette >= 0.5 else 'neutral' if silhouette >= 0.25 else 'warning'
        })

    key_insights.append({
        'title': 'Interpretation Caveat',
        'description': 'Unlike t-SNE, UMAP better preserves some global structure, but inter-cluster distances should still be interpreted cautiously rather than as precise dissimilarities.',
        'status': 'warning'
    })

    return {
        'key_insights': key_insights,
        'recommendation': (
            'Try a range of n_neighbors (5-50) and min_dist (0.0-0.5) to check how stable the cluster '
            'structure is. Because UMAP is stochastic, re-run with a few random seeds if the embedding '
            'will inform downstream decisions.'
        )
    }


@router.post("/umap")
async def run_umap_analysis(request: UMAPRequest) -> Dict[str, Any]:
    """
    Compute a UMAP embedding for visualization / exploratory clustering assessment.
    """
    try:
        data = request.data
        feature_cols = request.feature_cols
        label_col = request.label_col

        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")
        if len(feature_cols) < 2:
            raise HTTPException(status_code=400, detail="At least 2 feature columns are required.")

        df = pd.DataFrame(data)
        all_cols = feature_cols + ([label_col] if label_col else [])
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing)}")

        clean_df = df[all_cols].copy()
        for col in feature_cols:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
        clean_df = clean_df.dropna()

        if len(clean_df) < 10:
            raise HTTPException(status_code=400, detail="At least 10 valid samples required after removing missing values.")

        labels = clean_df[label_col] if label_col else None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(clean_df[feature_cols].values)

        n_samples = X_scaled.shape[0]
        effective_neighbors = min(request.n_neighbors, max(2, n_samples - 1))

        reducer = umap.UMAP(
            n_components=request.n_components,
            n_neighbors=effective_neighbors,
            min_dist=request.min_dist,
            metric=request.metric,
            random_state=request.random_state
        )
        embedding = reducer.fit_transform(X_scaled)

        silhouette = None
        if labels is not None and labels.nunique() > 1:
            try:
                silhouette = _to_native_type(silhouette_score(embedding, labels))
            except Exception:
                silhouette = None

        embedding_plot = generate_embedding_plot(embedding, labels, label_col) if request.n_components >= 2 else None
        interpretation = generate_interpretation(effective_neighbors, request.n_neighbors, request.min_dist, silhouette, label_col)

        return {
            'n_samples': n_samples,
            'n_features': len(feature_cols),
            'n_components': request.n_components,
            'n_neighbors_requested': request.n_neighbors,
            'n_neighbors_used': int(effective_neighbors),
            'min_dist': request.min_dist,
            'metric': request.metric,
            'silhouette_score': silhouette,
            'embedding': embedding.tolist(),
            'embedding_plot': embedding_plot,
            'interpretation': interpretation
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"UMAP analysis failed: {str(e)}")
