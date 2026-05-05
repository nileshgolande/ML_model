"""Evaluation module for classification models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_single_model(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Compute classification metrics and confusion matrix for one model."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_models(
    trained_models: Dict[str, object], X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Dict]:
    """Evaluate all trained models on holdout data."""
    results: Dict[str, Dict] = {}
    for model_name, model_pipeline in trained_models.items():
        results[model_name] = evaluate_single_model(model_pipeline, X_test, y_test)
    return results


def get_best_model(
    evaluation_results: Dict[str, Dict], cv_scores: Dict[str, float]
) -> Tuple[str, Dict]:
    """
    Pick best model using test F1 score, with CV score as secondary indicator.
    """
    best_name = max(
        evaluation_results.keys(),
        key=lambda name: (
            evaluation_results[name]["f1_score"],
            cv_scores.get(name, 0.0),
        ),
    )
    best_summary = evaluation_results[best_name].copy()
    best_summary["cv_f1_weighted"] = cv_scores.get(best_name, 0.0)
    return best_name, best_summary


def save_confusion_matrix_plot(
    confusion_matrix_values: list[list[int]],
    labels: list[str],
    output_path: str | Path,
) -> None:
    """Save confusion matrix heatmap to disk."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix_values,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
