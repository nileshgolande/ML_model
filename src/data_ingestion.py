"""Data ingestion module for downloading and storing raw data."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_openml


def collect_titanic_dataset(output_dir: str | Path) -> Tuple[pd.DataFrame, Path]:
    """
    Download the Titanic dataset from OpenML and save it as raw CSV.

    Returns:
        Tuple of DataFrame and saved CSV path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # as_frame=True returns a pandas DataFrame with target as series.
    dataset = fetch_openml(name="titanic", version=1, as_frame=True, parser="auto")
    data = dataset.frame.copy()

    raw_csv_path = output_path / "titanic_raw.csv"
    data.to_csv(raw_csv_path, index=False)

    return data, raw_csv_path
