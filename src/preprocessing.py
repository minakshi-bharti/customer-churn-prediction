"""Data cleaning, encoding, scaling, and train/test splitting."""
from __future__ import annotations
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    LABEL_ENCODE_COLS, NUMERIC_COLS, ONEHOT_COL, RANDOM_SEED,
    RAW_FEATURES, TARGET, TEST_SIZE,
)
from src.utils import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    pass


class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.onehot_columns = []
        self.scaler = StandardScaler()
        self.feature_columns = []
        self._fitted = False

    @staticmethod
    def validate(df: pd.DataFrame) -> None:
        missing = (set(RAW_FEATURES) | {TARGET}) - set(df.columns)
        if missing:
            raise DataValidationError(f"Missing required columns: {sorted(missing)}")
        if not df[TARGET].dropna().isin([0, 1]).all():
            raise DataValidationError(f"'{TARGET}' must be binary.")

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(keep="first")
        if "CustomerID" in df.columns:
            df = df.drop_duplicates(subset=["CustomerID"], keep="first")
        removed = before - len(df)
        if removed:
            logger.info(f"Removed {removed} duplicate row(s).")
        return df.reset_index(drop=True)

    @staticmethod
    def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in NUMERIC_COLS:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        for col in LABEL_ENCODE_COLS + [ONEHOT_COL]:
            if col in df.columns and df[col].isnull().any():
                mode = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
        return df

    def encode(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        for col in LABEL_ENCODE_COLS:
            if fit:
                enc = LabelEncoder()
                df[col] = enc.fit_transform(df[col].astype(str))
                self.label_encoders[col] = enc
            else:
                enc = self.label_encoders[col]
                df[col] = df[col].astype(str).map(
                    lambda v, e=enc: e.transform([v])[0] if v in e.classes_ else -1
                )
        onehot = pd.get_dummies(df[ONEHOT_COL], prefix=ONEHOT_COL)
        if fit:
            self.onehot_columns = list(onehot.columns)
        else:
            onehot = onehot.reindex(columns=self.onehot_columns, fill_value=0)
        return pd.concat([df.drop(columns=[ONEHOT_COL]), onehot], axis=1)

    def scale(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        cols = [c for c in NUMERIC_COLS if c in df.columns]
        df[cols] = self.scaler.fit_transform(df[cols]) if fit else self.scaler.transform(df[cols])
        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        self.validate(df)
        df = self.remove_duplicates(df)
        df = self.handle_missing(df)

        y = df[TARGET].astype(int)
        X = df[RAW_FEATURES].copy()
        X = self.encode(X, fit=True)
        X = self.scale(X, fit=True)

        self.feature_columns = list(X.columns)
        self._fitted = True
        logger.info(f"Preprocessing complete. Feature matrix: {X.shape}")
        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        df = self.handle_missing(df.copy())
        X = df[RAW_FEATURES].copy()
        X = self.encode(X, fit=False)
        X = self.scale(X, fit=False)
        return X.reindex(columns=self.feature_columns, fill_value=0)

    @staticmethod
    def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Split: train={len(X_train)}, test={len(X_test)}")
        return X_train, X_test, y_train, y_test
