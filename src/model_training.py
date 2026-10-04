"""Trains Logistic Regression, Decision Tree, and Random Forest (tuned)."""
from __future__ import annotations
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from src.config import DTREE_PARAMS, GRID_CV_FOLDS, GRID_SCORING, LOGREG_PARAMS, RF_GRID, RF_PARAMS
from src.utils import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_rf_params = {}
        self.best_rf_score = 0.0

    def train_logistic_regression(self, X_train, y_train):
        model = LogisticRegression(**LOGREG_PARAMS).fit(X_train, y_train)
        self.models["Logistic Regression"] = model
        logger.info("Logistic Regression trained.")
        return model

    def train_decision_tree(self, X_train, y_train):
        model = DecisionTreeClassifier(**DTREE_PARAMS).fit(X_train, y_train)
        self.models["Decision Tree"] = model
        logger.info("Decision Tree trained.")
        return model

    def tune_random_forest(self, X_train, y_train):
        logger.info("Running GridSearchCV for Random Forest...")
        base = RandomForestClassifier(random_state=RF_PARAMS["random_state"], n_jobs=-1)
        grid = GridSearchCV(base, RF_GRID, cv=GRID_CV_FOLDS, scoring=GRID_SCORING, n_jobs=-1)
        grid.fit(X_train, y_train)
        logger.info(f"Best RF params: {grid.best_params_} | best {GRID_SCORING}: {grid.best_score_:.4f}")
        self.best_rf_params = grid.best_params_
        self.best_rf_score = grid.best_score_
        self.models["Random Forest"] = grid.best_estimator_
        return grid.best_estimator_

    def train_all_models(self, X_train, y_train):
        self.train_logistic_regression(X_train, y_train)
        self.train_decision_tree(X_train, y_train)
        self.tune_random_forest(X_train, y_train)
        return self.models
