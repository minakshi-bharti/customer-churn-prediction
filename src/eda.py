"""Exploratory data analysis routines."""
from __future__ import annotations
import pandas as pd
from src.config import NUMERIC_COLS, TARGET
from src.utils import get_logger

logger = get_logger(__name__)


class ChurnEDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def overview(self) -> dict:
        return {
            "shape": self.df.shape,
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "memory_mb": round(self.df.memory_usage(deep=True).sum() / 1024**2, 3),
        }

    def missing_report(self) -> pd.DataFrame:
        miss = self.df.isnull().sum()
        return pd.DataFrame({"missing": miss, "pct": (miss / len(self.df) * 100).round(2)})

    def summary_stats(self) -> pd.DataFrame:
        cols = [c for c in NUMERIC_COLS if c in self.df.columns]
        return self.df[cols].describe().round(2)

    def churn_rate(self) -> dict:
        rate = self.df[TARGET].mean() * 100
        return {"churn_pct": round(rate, 2), "retained_pct": round(100 - rate, 2)}

    def avg_satisfaction_by_churn(self) -> pd.Series:
        return self.df.groupby(TARGET)["SatisfactionScore"].mean().round(2)

    def subscription_analysis(self) -> pd.DataFrame:
        g = self.df.groupby("SubscriptionType")[TARGET].agg(total="count", churned="sum")
        g["churn_rate_pct"] = (g.churned / g.total * 100).round(2)
        return g.sort_values("churn_rate_pct", ascending=False)

    def payment_method_analysis(self) -> pd.DataFrame:
        g = self.df.groupby("PaymentMethod")[TARGET].agg(total="count", churned="sum")
        g["churn_rate_pct"] = (g.churned / g.total * 100).round(2)
        return g.sort_values("churn_rate_pct", ascending=False)

    def correlation(self) -> pd.DataFrame:
        cols = [c for c in NUMERIC_COLS if c in self.df.columns] + [TARGET]
        return self.df[cols].corr().round(3)

    def run_full_eda(self) -> dict:
        logger.info("Running EDA suite...")
        return {
            "overview": self.overview(),
            "missing": self.missing_report(),
            "summary": self.summary_stats(),
            "churn_rate": self.churn_rate(),
            "avg_satisfaction_by_churn": self.avg_satisfaction_by_churn(),
            "subscription_analysis": self.subscription_analysis(),
            "payment_method_analysis": self.payment_method_analysis(),
            "correlation": self.correlation(),
        }
