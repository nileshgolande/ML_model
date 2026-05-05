"""Model training module with modular model registry."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def get_model_registry(random_state: int) -> Dict[str, object]:
    """Return dictionary of model names and initialized estimators."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: object,
    models: Dict[str, object],
    feature_selector: object | None = None,
    cv_folds: int = 5,
) -> Tuple[Dict[str, Pipeline], Dict[str, float]]:
    """Train all models with a unified pipeline and collect CV scores."""
    trained_models: Dict[str, Pipeline] = {}
    cv_scores: Dict[str, float] = {}

    for model_name, estimator in models.items():
        steps = [("preprocessor", preprocessor)]
        if feature_selector is not None:
            steps.append(("feature_selector", feature_selector))
        steps.append(("model", estimator))

        pipeline = Pipeline(steps=steps)
        pipeline.fit(X_train, y_train)
        scores = cross_val_score(
            pipeline, X_train, y_train, scoring="f1_weighted", cv=cv_folds, n_jobs=-1
        )
        trained_models[model_name] = pipeline
        cv_scores[model_name] = float(np.mean(scores))

    return trained_models, cv_scores
