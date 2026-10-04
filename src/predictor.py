"""Full training pipeline + ChurnPredictor inference class."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import pandas as pd

from src.config import GENDERS, MODEL_PATH, PAYMENT_METHODS, RAW_FEATURES, SUBSCRIPTION_TYPES
from src.dataset_generator import ChurnDatasetGenerator
from src.evaluation import ModelEvaluator
from src.model_training import ModelTrainer
from src.preprocessing import DataPreprocessor
from src.utils import get_logger, load_pickle, save_pickle
from src.visualization import ChurnVisualizer

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    prediction: str
    churn_probability: float
    retain_probability: float
    model_used: str

    def __str__(self):
        return f"{self.prediction} (Churn probability: {self.churn_probability:.1%}, Model: {self.model_used})"


class InvalidCustomerInputError(Exception):
    pass


def train_and_save_pipeline(model_path=MODEL_PATH, generate_data: bool = True) -> Dict:
    from src.config import DATASET_PATH
    df = ChurnDatasetGenerator().generate_and_save() if generate_data else pd.read_csv(DATASET_PATH)

    pre = DataPreprocessor()
    X, y = pre.fit_transform(df)
    X_train, X_test, y_train, y_test = pre.split_data(X, y)

    trainer = ModelTrainer()
    models = trainer.train_all_models(X_train, y_train)

    viz = ChurnVisualizer()
    evaluator = ModelEvaluator(viz)
    comparison = evaluator.evaluate_all_models(models, X_test, y_test)
    best_name, best_model = evaluator.select_best_model(models)
    evaluator.feature_importance(models["Random Forest"], list(X.columns), "Random Forest")
    viz.generate_all_eda(df)

    bundle = {
        "model": best_model, "model_name": best_name, "preprocessor": pre,
        "comparison_table": comparison, "feature_columns": list(X.columns),
    }
    save_pickle(bundle, model_path)
    logger.info(f"Pipeline complete. Best model: {best_name}")
    return bundle


class ChurnPredictor:
    def __init__(self, model_path=MODEL_PATH):
        bundle = load_pickle(model_path)
        self.model = bundle["model"]
        self.model_name = bundle["model_name"]
        self.preprocessor: DataPreprocessor = bundle["preprocessor"]
        self.feature_columns = bundle["feature_columns"]
        self.comparison_table = bundle.get("comparison_table")

    @staticmethod
    def validate(record: dict) -> None:
        missing = [f for f in RAW_FEATURES if f not in record]
        if missing:
            raise InvalidCustomerInputError(f"Missing field(s): {missing}")
        if not (0 <= record["Age"] <= 120):
            raise InvalidCustomerInputError("Age must be between 0 and 120.")
        if record["Gender"] not in GENDERS:
            raise InvalidCustomerInputError(f"Gender must be one of {GENDERS}.")
        if record["SubscriptionType"] not in SUBSCRIPTION_TYPES:
            raise InvalidCustomerInputError(f"SubscriptionType must be one of {SUBSCRIPTION_TYPES}.")
        if record["MonthlyCharges"] < 0:
            raise InvalidCustomerInputError("MonthlyCharges must be non-negative.")
        if record["TenureMonths"] < 0:
            raise InvalidCustomerInputError("TenureMonths must be non-negative.")
        if record["SupportTickets"] < 0:
            raise InvalidCustomerInputError("SupportTickets must be non-negative.")
        if record["PaymentMethod"] not in PAYMENT_METHODS:
            raise InvalidCustomerInputError(f"PaymentMethod must be one of {PAYMENT_METHODS}.")
        if not (1 <= record["SatisfactionScore"] <= 10):
            raise InvalidCustomerInputError("SatisfactionScore must be between 1 and 10.")
        if record["UsageHoursPerWeek"] < 0:
            raise InvalidCustomerInputError("UsageHoursPerWeek must be non-negative.")

    def predict(self, record: dict) -> PredictionResult:
        self.validate(record)
        X = self.preprocessor.transform(pd.DataFrame([record]))
        X = X.reindex(columns=self.feature_columns, fill_value=0)
        churn_p = float(self.model.predict_proba(X)[0, 1])
        label = "Likely to Churn" if churn_p >= 0.5 else "Likely to Stay"
        return PredictionResult(label, round(churn_p, 4), round(1 - churn_p, 4), self.model_name)


if __name__ == "__main__":
    train_and_save_pipeline()
