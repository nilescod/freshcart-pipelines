import json
from pathlib import Path

import joblib
import pandas as pd


if "data_exporter" not in globals():
    try:
        from mage_ai.data_preparation.decorators import data_exporter
    except ImportError:
        def data_exporter(func):
            return func


@data_exporter
def save_results(evaluation, *args, **kwargs):
    if not isinstance(evaluation, dict):
        raise TypeError("save_results expects the train_model evaluation dictionary.")

    project_root = Path(
        kwargs.get("repo_path")
        or Path(__file__).resolve().parents[1]
    )
    model_path = project_root / "models" / "iris_decision_tree.joblib"
    results_path = project_root / "results" / "iris_evaluation.json"
    predictions_path = project_root / "results" / "iris_predictions.csv"

    required_keys = (
        "feature_names",
        "test_indices",
        "test_features",
        "test_target",
        "predictions",
    )
    if any(key not in evaluation for key in required_keys):
        raise ValueError("Evaluation output is missing held-out prediction data.")
    if any(len(evaluation[key]) != 30 for key in required_keys[1:]):
        raise ValueError("Expected exactly 30 held-out test predictions to save.")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame(
        evaluation["test_features"],
        columns=evaluation["feature_names"],
    )
    predictions.insert(0, "source_index", evaluation["test_indices"])
    predictions["actual"] = evaluation["test_target"]
    predictions["predicted"] = evaluation["predictions"]

    joblib.dump(evaluation["model"], model_path)
    predictions.to_csv(predictions_path, index=False)
    result = {
        "accuracy": float(evaluation["accuracy"]),
        "classification_report": evaluation["classification_report"],
        "max_depth": int(evaluation["max_depth"]),
        "train_rows": int(evaluation["train_rows"]),
        "test_rows": int(evaluation["test_rows"]),
        "model_path": str(model_path),
        "predictions_path": str(predictions_path),
    }
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)

    return {
        **result,
        "results_path": str(results_path),
        "predictions_path": str(predictions_path),
    }
