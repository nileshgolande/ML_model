"""Feature engineering and feature selection steps."""

from __future__ import annotations

from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline


def build_feature_selector(percentile: int = 70) -> Pipeline:
    """
    Build a feature selection pipeline component.

    Uses ANOVA F-value selection and keeps a configurable top percentile.
    """
    return Pipeline(
        steps=[
            ("select_features", SelectPercentile(score_func=f_classif, percentile=percentile))
        ]
    )
