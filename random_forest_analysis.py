
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def _to_native_type(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        features = payload.get('features')
        target = payload.get('target')
        
        n_estimators = int(payload.get('n_estimators', 100))
        max_depth = int(payload.get('max_depth')) if payload.get('max_depth') else None
        min_samples_split = int(payload.get('min_samples_split', 2))
        min_samples_leaf = int(payload.get('min_samples_leaf', 1))

        df = pd.DataFrame(data)

        X = df[features]
        y = df[target]

        X = pd.get_dummies(X, drop_first=True)
        feature_names = X.columns.tolist()
        
        # Scale numeric features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        # No-Information Rate (NIR)
        nir = y_test.value_counts(normalize=True).max()

        roc_auc_data = {}
        if len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=model.classes_[1])
            roc_auc = auc(fpr, tpr)
            roc_auc_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc
            }

        results = {
            'accuracy': accuracy,
            'nir': nir,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'feature_importance': dict(zip(feature_names, model.feature_importances_)),
            'roc_auc_data': roc_auc_data,
            'class_names': [str(c) for c in model.classes_]
        }

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Random Forest Classifier Analysis', fontsize=16)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], xticklabels=model.classes_, yticklabels=model.classes_)
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted Label')
        axes[0, 0].set_ylabel('True Label')

        importance_df = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).head(15)
        sns.barplot(x='importance', y='feature', data=importance_df, ax=axes[0, 1])
        axes[0, 1].set_title('Top 15 Feature Importances')

        if roc_auc_data:
            axes[1, 0].plot(roc_auc_data['fpr'], roc_auc_data['tpr'], color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc_data['auc']:.2f})")
            axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].set_title('Receiver Operating Characteristic (ROC) Curve')
            axes[1, 0].legend(loc="lower right")

        axes[1, 1].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        plot_image = base64.b64encode(buf.read()).decode('utf-8')

        response = {
            'results': results,
            'plot': f"data:image/png;base64,{plot_image}"
        }
        
        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
