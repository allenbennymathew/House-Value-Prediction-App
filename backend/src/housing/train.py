import yaml
import logging
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


class CustomFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, 3] / X[:, 6]
        bedrooms_per_room = X[:, 4] / X[:, 3]
        population_per_household = X[:, 5] / X[:, 6]
        return np.c_[X, rooms_per_household, bedrooms_per_room, population_per_household]

    def inverse_transform(self, X):
        return X[:, :-3]


def _build_preprocessor(num_attribs, cat_attribs):
    """Build the shared preprocessing pipeline."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CustomFeatures()),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_attribs),
    ])


def train_models(housing, housing_labels):
    logger.info("Training all three models with shared preprocessing pipeline...")

    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    models = {}

    # ── 1. Linear Regression ────────────────────────────────────────
    logger.info("Fitting Linear Regression...")
    lin_pipeline = Pipeline([
        ("preparation", _build_preprocessor(num_attribs, cat_attribs)),
        ("model", LinearRegression()),
    ])
    lin_pipeline.fit(housing, housing_labels)
    models["linear_regression"] = lin_pipeline
    logger.info("Linear Regression fitted successfully.")

    # ── 2. Decision Tree ────────────────────────────────────────────
    logger.info("Fitting Decision Tree...")
    tree_pipeline = Pipeline([
        ("preparation", _build_preprocessor(num_attribs, cat_attribs)),
        ("model", DecisionTreeRegressor(random_state=42, max_depth=12)),
    ])
    tree_pipeline.fit(housing, housing_labels)
    models["decision_tree"] = tree_pipeline
    logger.info("Decision Tree fitted successfully.")

    # ── 3. Random Forest with GridSearchCV ──────────────────────────
    logger.info("Fitting Random Forest with GridSearchCV (this takes ~1-2 min)...")
    forest_pipeline = Pipeline([
        ("preparation", _build_preprocessor(num_attribs, cat_attribs)),
        ("forest", RandomForestRegressor(random_state=42)),
    ])

    param_grid = [
        {"forest__n_estimators": [3, 10, 30], "forest__max_features": [2, 4, 6, 8]},
        {"forest__bootstrap": [False], "forest__n_estimators": [3, 10],
         "forest__max_features": [2, 3, 4]},
    ]

    if os.path.exists("config.yml"):
        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)
        if "training" in config and "param_grid" in config["training"]:
            raw = config["training"]["param_grid"]
            # normalise legacy key names
            param_grid = [
                {k.replace("forest_reg__", "forest__"): v for k, v in p.items()}
                for p in raw
            ]

    grid_search = GridSearchCV(
        forest_pipeline, param_grid,
        cv=5, scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing, housing_labels)
    models["random_forest"] = grid_search.best_estimator_
    logger.info(f"Random Forest best params: {grid_search.best_params_}")

    return models
