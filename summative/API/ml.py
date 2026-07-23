"""Shared logic for training and retraining the salary model.

Kept separate from the API so the prediction and retrain endpoints use the same
feature engineering and preprocessing that the notebook used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

TARGET = "salary_in_usd"
NUMERIC = ["work_year", "remote_ratio"]
ORDINAL = ["experience_level", "company_size"]
NOMINAL = ["employment_type", "job_category", "company_location_grp"]
FEATURES = NUMERIC + ORDINAL + NOMINAL

EXP_ORDER = ["EN", "MI", "SE", "EX"]
SIZE_ORDER = ["S", "M", "L"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering shared with the notebook. Safe to run on the raw data or on
    already engineered data, because missing columns are skipped."""
    df = df.copy()
    df = df.drop(columns=["salary", "salary_currency", "employee_residence"], errors="ignore")

    def job_cat(t: str) -> str:
        t = str(t).lower()
        if any(k in t for k in ["manager", "lead", "head", "director", "principal"]):
            return "Management"
        if any(k in t for k in ["machine learning", "ml engineer", "ai ", " ai", "deep learning", "computer vision", "nlp"]):
            return "ML/AI Engineer"
        if any(k in t for k in ["engineer", "developer", "architect"]):
            return "Data Engineer"
        if any(k in t for k in ["scientist", "research"]):
            return "Data Scientist"
        if any(k in t for k in ["analyst", "analytics"]):
            return "Data Analyst"
        return "Other"

    if "job_title" in df.columns:
        df["job_category"] = df["job_title"].apply(job_cat)
        df = df.drop(columns=["job_title"])

    top_locs = ["US", "GB", "CA", "ES", "IN", "DE"]
    if "company_location" in df.columns:
        df["company_location_grp"] = df["company_location"].apply(lambda x: x if x in top_locs else "Other")
        df = df.drop(columns=["company_location"])
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC),
        ("ord", Pipeline([
            ("enc", OrdinalEncoder(categories=[EXP_ORDER, SIZE_ORDER],
                                   handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scale", StandardScaler()),
        ]), ORDINAL),
        ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL),
    ])


class BatchGDRegressor(BaseEstimator, RegressorMixin):
    """Linear regression trained with full-batch gradient descent (from scratch)."""

    def __init__(self, lr: float = 0.1, n_epochs: int = 800):
        self.lr = lr
        self.n_epochs = n_epochs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._y_mean, self._y_std = y.mean(), y.std() + 1e-12
        y_s = (y - self._y_mean) / self._y_std
        n, m = X.shape
        self.w_ = np.zeros(m)
        self.b_ = 0.0
        for _ in range(self.n_epochs):
            err = (X @ self.w_ + self.b_) - y_s
            self.w_ -= self.lr * (2 / n) * (X.T @ err)
            self.b_ -= self.lr * (2 / n) * err.sum()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X @ self.w_ + self.b_) * self._y_std + self._y_mean


def _candidate_models() -> dict[str, Pipeline]:
    """The 4 models: two gradient descent linear, one ensemble, one tree.

    Uses fixed hyperparameters so retraining stays fast. The notebook does the
    heavier grid search.
    """
    return {
        "SGD Linear Regression (GD)": Pipeline([
            ("pre", build_preprocessor()),
            ("model", SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-3,
                                   learning_rate="invscaling", eta0=0.01,
                                   max_iter=2000, early_stopping=True, random_state=42)),
        ]),
        "Batch GD Linear Regression": Pipeline([
            ("pre", build_preprocessor()),
            ("model", BatchGDRegressor(lr=0.1, n_epochs=800)),
        ]),
        "Random Forest": Pipeline([
            ("pre", build_preprocessor()),
            ("model", RandomForestRegressor(n_estimators=200, max_depth=12,
                                            min_samples_leaf=3, random_state=42, n_jobs=-1)),
        ]),
        "Decision Tree": Pipeline([
            ("pre", build_preprocessor()),
            ("model", DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42)),
        ]),
    }


def train_and_select(df: pd.DataFrame) -> tuple[Pipeline, str, dict]:
    """Engineer the data, split it, train the 4 models, and return the best pipeline,
    its name, and the metrics. Best means the lowest test RMSE, which is our loss metric.
    """
    data = engineer_features(df)
    missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
    if missing:
        raise ValueError(f"Training data missing required columns: {missing}")

    X, y = data[FEATURES], data[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    metrics: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for name, pipe in _candidate_models().items():
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        metrics[name] = {
            "test_rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
            "test_mae": float(mean_absolute_error(y_te, pred)),
            "test_r2": float(r2_score(y_te, pred)),
        }
        fitted[name] = pipe

    best_name = min(metrics, key=lambda k: metrics[k]["test_rmse"])
    return fitted[best_name], best_name, metrics
