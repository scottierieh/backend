"""
t-SNE (t-Distributed Stochastic Neighbor Embedding) Router for FastAPI
Non-linear dimensionality reduction for visualization
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
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class TSNERequest(BaseModel):
    data: List[Dict[str, Any]]
    feature_cols: List[str]
    label_col: Optional[str] = None  # optional column used only for coloring/silhouette, not the embedding itself
    n_components: int = 2
    perplexity: float = 30.0
    learning_rate: str = "auto"
    n_iter: int = 1000
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
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.set_title('t-SNE Embedding', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(kl_divergence: float, perplexity_used: float, perplexity_requested: float,
                             silhouette: Optional[float], label_col: Optional[str]) -> Dict[str, Any]:
    key_insights = []

    key_insights.append({
        'title': 'Embedding Quality (KL Divergence)',
        'description': f'Final KL divergence = {kl_divergence:.3f} (lower means the 2D layout better preserves local neighborhoods from the original space; not comparable across different datasets).',
        'status': 'neutral'
    })

    if abs(perplexity_used - perplexity_requested) > 0.01:
        key_insights.append({
            'title': 'Perplexity Adjusted',
            'description': f'Requested perplexity ({perplexity_requested:.1f}) was reduced to {perplexity_used:.1f} because it must stay well below the sample size.',
            'status': 'neutral'
        })

    if silhouette is not None:
        sil_desc = 'well separated' if silhouette >= 0.5 else 'moderately separated' if silhouette >= 0.25 else 'overlapping'
        key_insights.append({
            'title': 'Cluster Separation',
            'description': f"Silhouette score of the embedding w.r.t. '{label_col}' = {silhouette:.3f} — groups appear {sil_desc} in the 2D projection.",
            'status': 'positive' if silhouette >= 0.5 else 'neutral' if silhouette >= 0.25 else 'warning'
        })

    key_insights.append({
        'title': 'Interpretation Caveat',
        'description': 'Cluster sizes, inter-cluster distances, and densities in t-SNE are not directly meaningful — only relative neighborhood grouping should be interpreted.',
        'status': 'warning'
    })

    return {
        'key_insights': key_insights,
        'recommendation': (
            'Try a few different perplexity values (typically 5-50) since results can vary noticeably, '
            'especially on smaller datasets. For a more globally faithful and reproducible alternative, '
            'consider UMAP or PCA.'
        )
    }


@router.post("/tsne")
async def run_tsne_analysis(request: TSNERequest) -> Dict[str, Any]:
    """
    Compute a t-SNE embedding for visualization / exploratory clustering assessment.
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
        effective_perplexity = min(request.perplexity, max(5.0, (n_samples - 1) / 3))

        learning_rate = request.learning_rate
        if learning_rate != 'auto':
            try:
                learning_rate = float(learning_rate)
            except ValueError:
                learning_rate = 'auto'

        tsne = TSNE(
            n_components=request.n_components,
            perplexity=effective_perplexity,
            learning_rate=learning_rate,
            max_iter=request.n_iter,
            random_state=request.random_state,
            init='pca'
        )
        embedding = tsne.fit_transform(X_scaled)

        silhouette = None
        if labels is not None and labels.nunique() > 1:
            try:
                silhouette = _to_native_type(silhouette_score(embedding, labels))
            except Exception:
                silhouette = None

        embedding_plot = generate_embedding_plot(embedding, labels, label_col) if request.n_components >= 2 else None
        interpretation = generate_interpretation(
            _to_native_type(tsne.kl_divergence_), effective_perplexity, request.perplexity, silhouette, label_col
        )

        return {
            'n_samples': n_samples,
            'n_features': len(feature_cols),
            'n_components': request.n_components,
            'perplexity_requested': request.perplexity,
            'perplexity_used': _to_native_type(effective_perplexity),
            'n_iterations_run': int(tsne.n_iter_),
            'kl_divergence': _to_native_type(tsne.kl_divergence_),
            'silhouette_score': silhouette,
            'embedding': embedding.tolist(),
            'embedding_plot': embedding_plot,
            'interpretation': interpretation
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"t-SNE analysis failed: {str(e)}")
