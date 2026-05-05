"""Utility helpers for logging, config loading, and persistence."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import yaml


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Configure and return a project-wide logger."""
    logger = logging.getLogger("ml_lifecycle")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML config into a dictionary."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found at: {config_file}")

    with config_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save dictionary as a prettified JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_joblib_model(model: Any, path: str | Path) -> None:
    """Persist model object with joblib."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def generate_requirements_file(output_path: str | Path = "requirements.txt") -> Path:
    """
    Generate a requirements file for this project.

    This keeps dependency generation reproducible for GitHub workflows.
    """
    dependencies = [
        "joblib>=1.4.0",
        "matplotlib>=3.8.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "PyYAML>=6.0.0",
        "scikit-learn>=1.5.0",
        "seaborn>=0.13.0",
    ]
    path = Path(output_path)
    path.write_text("\n".join(dependencies) + "\n", encoding="utf-8")
    return path
