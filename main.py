"""End-to-end ML lifecycle runner."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from src.data_ingestion import collect_titanic_dataset
from src.data_preprocessing import build_preprocessor, infer_feature_types, split_features_target
from src.evaluate_model import evaluate_models, get_best_model, save_confusion_matrix_plot
from src.feature_engineering import build_feature_selector
from src.train_model import get_model_registry, train_models
from src.utils import (
    generate_requirements_file,
    load_config,
    save_joblib_model,
    save_json,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run end-to-end ML lifecycle pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--feature-selection",
        action="store_true",
        help="Enable feature selection step in pipeline.",
    )
    return parser.parse_args()


def run_pipeline(config_path: str, enable_feature_selection: bool = False) -> None:
    """Execute full ML lifecycle from ingestion to model persistence."""
    logger = setup_logger()
    req_path = generate_requirements_file("requirements.txt")
    logger.info("Generated requirements file at %s", req_path)
    logger.info("Loading config from %s", config_path)
    config = load_config(config_path)

    target_col = config["project"]["target_column"]
    random_state = config["project"]["random_state"]
    test_size = config["project"]["test_size"]
    cv_folds = config["project"]["cv_folds"]

    raw_data_dir = Path(config["paths"]["raw_data_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    best_model_path = models_dir / config["paths"]["best_model_file"]
    metrics_path = Path(config["paths"]["metrics_file"])
    confusion_matrix_path = Path(config["paths"]["confusion_matrix_file"])

    logger.info("Step 1/7: Data collection")
    data, raw_csv_path = collect_titanic_dataset(raw_data_dir)
    logger.info("Raw dataset saved to %s", raw_csv_path)

    logger.info("Step 2/7: Data cleaning and preprocessing setup")
    X_train, X_test, y_train, y_test = split_features_target(
        data, target_col=target_col, test_size=test_size, random_state=random_state
    )
    num_cols, cat_cols = infer_feature_types(X_train)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    logger.info("Step 3/7: Feature engineering")
    feature_selector = build_feature_selector(percentile=70) if enable_feature_selection else None

    logger.info("Step 4/7: Model training")
    model_registry = get_model_registry(random_state=random_state)
    trained_models, cv_scores = train_models(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        models=model_registry,
        feature_selector=feature_selector,
        cv_folds=cv_folds,
    )

    logger.info("Step 5/7: Model evaluation")
    eval_results = evaluate_models(trained_models, X_test, y_test)
    best_model_name, best_model_summary = get_best_model(eval_results, cv_scores)

    logger.info("Best model: %s", best_model_name)
    logger.info("Best model metrics: %s", best_model_summary)

    logger.info("Step 6/7: Deployment-ready artifact saving")
    save_joblib_model(trained_models[best_model_name], best_model_path)
    save_json({"cv_scores": cv_scores, "evaluation": eval_results}, metrics_path)

    labels = sorted(y_test.unique().tolist())
    save_confusion_matrix_plot(
        confusion_matrix_values=eval_results[best_model_name]["confusion_matrix"],
        labels=[str(label) for label in labels],
        output_path=confusion_matrix_path,
    )

    logger.info("Saved best model to %s", best_model_path)
    logger.info("Saved metrics report to %s", metrics_path)
    logger.info("Saved confusion matrix plot to %s", confusion_matrix_path)
    logger.info("Step 7/7: Pipeline complete")


if __name__ == "__main__":
    args = parse_args()
    try:
        run_pipeline(config_path=args.config, enable_feature_selection=args.feature_selection)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger = setup_logger()
        logger.error("Pipeline failed: %s", exc)
        logger.debug(traceback.format_exc())
        raise
