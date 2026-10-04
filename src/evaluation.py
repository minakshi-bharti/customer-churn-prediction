"""Model evaluation: metrics, comparison table, best-model selection."""
from __future__ import annotations
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.utils import get_logger
from src.visualization import ChurnVisualizer

logger = get_logger(__name__)


class ModelEvaluator:
    def __init__(self, visualizer: ChurnVisualizer = None):
        self.visualizer = visualizer or ChurnVisualizer()
        self.results = {}

    def evaluate_model(self, model, name, X_test, y_test, plots=True):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        }
        logger.info(f"[{name}] " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        if plots:
            self.visualizer.confusion_matrix_plot(y_test, y_pred, name)
            self.visualizer.roc_curve_plot(y_test, y_proba, name)
        self.results[name] = metrics
        return metrics

    def evaluate_all_models(self, models, X_test, y_test):
        for name, model in models.items():
            self.evaluate_model(model, name, X_test, y_test)
        comparison = self.build_comparison_table()
        self.visualizer.model_comparison_bar(comparison)
        return comparison

    def build_comparison_table(self) -> pd.DataFrame:
        if not self.results:
            raise RuntimeError("No results yet — call evaluate_all_models first.")
        rows = [{"Model": n, **m} for n, m in self.results.items()]
        df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
        cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        df[cols] = df[cols].round(4)
        return df

    def select_best_model(self, models, metric="F1 Score"):
        if not self.results:
            raise RuntimeError("No results yet.")
        best = max(self.results, key=lambda n: self.results[n][metric])
        logger.info(f"Best model: {best} ({metric}={self.results[best][metric]:.4f})")
        return best, models[best]

    def feature_importance(self, model, feature_names, model_name):
        if not hasattr(model, "feature_importances_"):
            logger.warning(f"{model_name} has no feature_importances_.")
            return
        self.visualizer.feature_importance_plot(feature_names, model.feature_importances_, model_name)
