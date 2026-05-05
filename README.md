# End-to-End Machine Learning Lifecycle Project

This project demonstrates a complete and production-style ML lifecycle in Python using the Titanic dataset from OpenML:

**Data Collection -> Data Cleaning -> Feature Engineering -> Train/Test Split -> Model Training -> Evaluation -> Deployment (Model Saving)**

## Project Structure

```text
.
├── config/
│   └── config.yaml
├── data/
├── models/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Dataset

- Source: Titanic dataset from [OpenML](https://www.openml.org/)
- Automatically downloaded via `sklearn.datasets.fetch_openml`
- Raw file is saved to `data/titanic_raw.csv`

## Features Implemented

- Automated data ingestion and raw data persistence
- Missing value handling with `SimpleImputer`
- Categorical encoding with `OneHotEncoder`
- Numerical scaling with `StandardScaler`
- Optional feature selection (`--feature-selection`)
- 80/20 train-test split with fixed `random_state`
- Trains 3 models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Cross-validation (`f1_weighted`) for each model
- Holdout evaluation:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
- Best model selection and persistence using `joblib`
- Logging and exception handling
- Simple CLI interface with `argparse`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

Run the full lifecycle pipeline:

```bash
python main.py
```

Run with feature selection enabled:

```bash
python main.py --feature-selection
```

Use a custom config:

```bash
python main.py --config config/config.yaml
```

## Artifacts Generated

After execution, the following files are created:

- `data/titanic_raw.csv` - raw dataset
- `models/best_model.joblib` - serialized best model pipeline
- `models/metrics.json` - evaluation metrics for all models + CV scores
- `models/confusion_matrix.png` - confusion matrix for best model

## Example Output (Console)

```text
2026-05-05 16:50:02 | INFO | ml_lifecycle | Step 1/7: Data collection
2026-05-05 16:50:03 | INFO | ml_lifecycle | Step 4/7: Model training
2026-05-05 16:50:09 | INFO | ml_lifecycle | Best model: random_forest
2026-05-05 16:50:09 | INFO | ml_lifecycle | Step 7/7: Pipeline complete
```

## Notes

- `requirements.txt` is generated to include all runtime dependencies.
- The saved model is a full sklearn pipeline (preprocessing + estimator), ready for inference in deployment services.
"# ML_model" 
