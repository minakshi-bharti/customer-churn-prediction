"""Central configuration: paths, constants, and hyperparameters."""
from __future__ import annotations
import os
from pathlib import Path

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"
DATASET_PATH = DATA_DIR / "customer_churn_dataset.csv"
MODEL_PATH = MODELS_DIR / "churn_model.pkl"

for d in (DATA_DIR, MODELS_DIR, IMAGES_DIR):
    os.makedirs(d, exist_ok=True)

NUM_RECORDS = 3000

SUBSCRIPTION_TYPES = ["Basic", "Standard", "Premium"]
SUBSCRIPTION_WEIGHTS = [0.45, 0.35, 0.20]
GENDERS = ["Male", "Female"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "Bank Transfer", "E-Wallet"]
PAYMENT_WEIGHTS = [0.35, 0.20, 0.20, 0.15, 0.10]
CHARGES_BY_SUB = {"Basic": (9.99, 29.99), "Standard": (29.99, 59.99), "Premium": (59.99, 119.99)}

RAW_FEATURES = [
    "Age", "Gender", "SubscriptionType", "MonthlyCharges", "TenureMonths",
    "SupportTickets", "PaymentMethod", "SatisfactionScore", "UsageHoursPerWeek",
]
NUMERIC_COLS = [
    "Age", "MonthlyCharges", "TenureMonths", "SupportTickets",
    "SatisfactionScore", "UsageHoursPerWeek",
]
LABEL_ENCODE_COLS = ["Gender", "SubscriptionType"]
ONEHOT_COL = "PaymentMethod"
TARGET = "Churn"

TEST_SIZE = 0.2

LOGREG_PARAMS = {"max_iter": 1000, "random_state": RANDOM_SEED}
DTREE_PARAMS = {"max_depth": 6, "min_samples_leaf": 5, "random_state": RANDOM_SEED}
RF_PARAMS = {"n_estimators": 150, "max_depth": 8, "random_state": RANDOM_SEED, "n_jobs": -1}
RF_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [8, 12],
    "min_samples_split": [2, 5],
}
GRID_CV_FOLDS = 3
GRID_SCORING = "f1"
