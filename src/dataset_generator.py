"""Generates a realistic synthetic customer churn dataset with rule-based
(not random) churn labels: risk rises with low satisfaction, low usage,
short tenure, high charges, many support tickets, and Basic-tier plans.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.config import (
    CHARGES_BY_SUB, DATASET_PATH, GENDERS, NUM_RECORDS, PAYMENT_METHODS,
    PAYMENT_WEIGHTS, RANDOM_SEED, SUBSCRIPTION_TYPES, SUBSCRIPTION_WEIGHTS,
)
from src.utils import get_logger

logger = get_logger(__name__)


class ChurnDatasetGenerator:
    def __init__(self, n: int = NUM_RECORDS, seed: int = RANDOM_SEED):
        self.n = n
        self.rng = np.random.default_rng(seed)

    def generate(self) -> pd.DataFrame:
        rng = self.rng
        subs = rng.choice(SUBSCRIPTION_TYPES, self.n, p=SUBSCRIPTION_WEIGHTS)
        pay = rng.choice(PAYMENT_METHODS, self.n, p=PAYMENT_WEIGHTS)
        gender = rng.choice(GENDERS, self.n)
        age = np.clip(rng.normal(40, 13, self.n), 18, 75).astype(int)
        tenure = np.clip(rng.exponential(18, self.n), 0, 72).astype(int)
        tickets = np.clip(rng.poisson(2.0, self.n), 0, 12)
        satisfaction = np.clip(np.round(rng.normal(6, 2.2, self.n)), 1, 10).astype(int)
        usage = np.clip(rng.gamma(2, 6, self.n), 0, 40).round(1)
        charges = np.array([rng.uniform(*CHARGES_BY_SUB[s]) for s in subs]).round(2)

        df = pd.DataFrame({
            "CustomerID": [f"CUST{i:05d}" for i in range(1, self.n + 1)],
            "Age": age, "Gender": gender, "SubscriptionType": subs,
            "MonthlyCharges": charges, "TenureMonths": tenure, "SupportTickets": tickets,
            "PaymentMethod": pay, "SatisfactionScore": satisfaction, "UsageHoursPerWeek": usage,
        })
        df["Churn"] = self._assign_churn(df)
        df = self._inject_missing(df)
        df = self._inject_duplicates(df)

        logger.info(f"Generated {len(df)} rows. Churn rate: {df['Churn'].mean():.2%}")
        return df

    def _assign_churn(self, df: pd.DataFrame) -> np.ndarray:
        sat_n = 1 - (df.SatisfactionScore - 1) / 9
        use_n = 1 - (df.UsageHoursPerWeek / 40).clip(0, 1)
        ten_n = 1 - (df.TenureMonths / 72).clip(0, 1)
        chg_n = (df.MonthlyCharges / 119.99).clip(0, 1)
        tick_n = (df.SupportTickets / 12).clip(0, 1)
        basic = (df.SubscriptionType == "Basic").astype(float)

        risk = 1.4 * sat_n + 1.1 * use_n + 1.2 * ten_n + 0.9 * chg_n + 1.0 * tick_n + 0.6 * basic
        risk_c = (risk - risk.mean()) / risk.std()
        noise = self.rng.normal(0, 0.35, len(df))
        prob = 1 / (1 + np.exp(-(0.9 * risk_c + noise)))
        return (self.rng.uniform(0, 1, len(df)) < prob).astype(int)

    def _inject_missing(self, df: pd.DataFrame, frac: float = 0.015) -> pd.DataFrame:
        df = df.copy()
        for col in ["SatisfactionScore", "UsageHoursPerWeek", "PaymentMethod"]:
            idx = self.rng.choice(df.index, size=int(len(df) * frac), replace=False)
            df.loc[idx, col] = np.nan
        return df

    def _inject_duplicates(self, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        dupes = df.sample(n=n, random_state=RANDOM_SEED)
        return pd.concat([df, dupes], ignore_index=True)

    def generate_and_save(self, path=DATASET_PATH) -> pd.DataFrame:
        df = self.generate()
        df.to_csv(path, index=False)
        logger.info(f"Dataset saved to {path}")
        return df


if __name__ == "__main__":
    ChurnDatasetGenerator().generate_and_save()
