"""Generates and saves all charts (Matplotlib/Seaborn) to the images/ dir."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve

from src.config import IMAGES_DIR, NUMERIC_COLS, TARGET
from src.utils import get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")
LABELS = {0: "Retained", 1: "Churned"}
PALETTE = {"Retained": "#2ecc71", "Churned": "#e74c3c"}


class ChurnVisualizer:
    def __init__(self, out_dir: Path = IMAGES_DIR):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name):
        path = self.out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return path

    def pie_chart(self, df, name="churn_pie_chart.png"):
        counts = df[TARGET].value_counts().rename(index=LABELS)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90,
               colors=[PALETTE[l] for l in counts.index], explode=[0.03] * len(counts))
        ax.set_title("Customer Churn Distribution", fontweight="bold")
        return self._save(fig, name)

    def subscription_churn(self, df, name="subscription_wise_churn.png"):
        g = df.groupby("SubscriptionType")[TARGET].mean().mul(100).round(2).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 6))
        bars = ax.bar(g.index, g.values, color="#3498db", edgecolor="black")
        ax.bar_label(bars, fmt="%.1f%%")
        ax.set_title("Churn Rate by Subscription Type", fontweight="bold")
        ax.set_ylabel("Churn Rate (%)")
        return self._save(fig, name)

    def correlation_heatmap(self, df, name="correlation_heatmap.png"):
        cols = [c for c in NUMERIC_COLS if c in df.columns] + [TARGET]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Feature Correlation Heatmap", fontweight="bold")
        return self._save(fig, name)

    def charges_histogram(self, df, name="monthly_charges_histogram.png"):
        fig, ax = plt.subplots(figsize=(7, 6))
        for v, label in LABELS.items():
            ax.hist(df.loc[df[TARGET] == v, "MonthlyCharges"], bins=30, alpha=0.6,
                    label=label, color=PALETTE[label], edgecolor="black")
        ax.set_title("Distribution of Monthly Charges", fontweight="bold")
        ax.legend()
        return self._save(fig, name)

    def boxplot_charges(self, df, name="boxplot_charges_by_churn.png"):
        d = df.copy(); d["Status"] = d[TARGET].map(LABELS)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.boxplot(data=d, x="Status", y="MonthlyCharges", hue="Status", palette=PALETTE, legend=False, ax=ax)
        ax.set_title("Monthly Charges by Churn Status", fontweight="bold")
        return self._save(fig, name)

    def scatter_tenure_charges(self, df, name="scatter_tenure_vs_charges.png"):
        d = df.copy(); d["Status"] = d[TARGET].map(LABELS)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=d, x="TenureMonths", y="MonthlyCharges", hue="Status", palette=PALETTE, alpha=0.5, ax=ax)
        ax.set_title("Tenure vs. Monthly Charges", fontweight="bold")
        return self._save(fig, name)

    def confusion_matrix_plot(self, y_true, y_pred, model_name, name=None):
        name = name or f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"]).plot(ax=ax, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
        return self._save(fig, name)

    def roc_curve_plot(self, y_true, y_proba, model_name, name=None):
        name = name or f"roc_curve_{model_name.replace(' ', '_').lower()}.png"
        fig, ax = plt.subplots(figsize=(6, 6))
        RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name=model_name)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_title(f"ROC Curve — {model_name}", fontweight="bold")
        return self._save(fig, name)

    def feature_importance_plot(self, feature_names, importances, model_name, name="feature_importance.png", top_n=12):
        imp = pd.DataFrame({"feature": feature_names, "importance": importances}) \
            .sort_values("importance", ascending=False).head(top_n).sort_values("importance")
        fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.4)))
        ax.barh(imp.feature, imp.importance, color="#9b59b6", edgecolor="black")
        ax.set_title(f"Top Feature Importances — {model_name}", fontweight="bold")
        return self._save(fig, name)

    def model_comparison_bar(self, comparison_df, name="model_comparison.png"):
        metrics = [m for m in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"] if m in comparison_df.columns]
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_df.set_index("Model")[metrics].plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black")
        ax.set_title("Model Performance Comparison", fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=0)
        return self._save(fig, name)

    def generate_all_eda(self, df):
        logger.info("Generating EDA visualizations...")
        paths = [
            self.pie_chart(df), self.subscription_churn(df), self.correlation_heatmap(df),
            self.charges_histogram(df), self.boxplot_charges(df), self.scatter_tenure_charges(df),
        ]
        logger.info(f"Generated {len(paths)} EDA charts.")
        return paths
