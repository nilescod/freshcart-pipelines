import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


if "transformer" not in globals():
    try:
        from mage_ai.data_preparation.decorators import transformer
    except ImportError:
        def transformer(func):
            return func


@transformer
def evaluate_model(evaluation, *args, **kwargs):
    if not isinstance(evaluation, dict):
        raise TypeError("evaluate_model expects the train_model evaluation dictionary.")

    y_true = evaluation.get("actual_labels")
    if y_true is None:
        y_true = evaluation.get("test_target")
    feature_names = evaluation.get("feature_names")
    test_features = evaluation.get("test_features")
    model = evaluation.get("model")
    if not isinstance(y_true, list) or not isinstance(feature_names, list):
        raise ValueError("Evaluation input must include held-out test targets and features.")
    if not isinstance(test_features, list) or len(test_features) != 30 or len(y_true) != 30:
        raise ValueError("Evaluation requires exactly 30 held-out test records.")
    if model is None or not hasattr(model, "predict"):
        raise ValueError("Evaluation input must include the trained model.")

    test_frame = pd.DataFrame(test_features, columns=feature_names)
    y_pred = model.predict(test_frame).astype(int).tolist()
    if evaluation.get("predictions") != y_pred:
        raise ValueError("Held-out predictions do not match the trained model output.")

    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    metric_rows = [
        {
            "class": str(label),
            "precision": report[str(label)]["precision"],
            "recall": report[str(label)]["recall"],
            "f1_score": report[str(label)]["f1-score"],
            "support": report[str(label)]["support"],
        }
        for label in labels
    ]
    metric_rows.extend(
        {
            "class": average,
            "precision": report[average]["precision"],
            "recall": report[average]["recall"],
            "f1_score": report[average]["f1-score"],
            "support": report[average]["support"],
        }
        for average in ("macro avg", "weighted avg")
    )
    metric_rows.append(
        {
            "class": "accuracy",
            "precision": None,
            "recall": None,
            "f1_score": report["accuracy"],
            "support": len(y_true),
        }
    )
    metrics_table = pd.DataFrame(
        metric_rows,
        columns=["class", "precision", "recall", "f1_score", "support"],
    )
    confusion_table = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )

    print(f"Held-out test accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Confusion matrix:")
    print(confusion_table.to_string())
    print("Classification metrics:")
    print(metrics_table.to_string(index=False))

    project_root = Path(
        kwargs.get("repo_path")
        or Path(__file__).resolve().parents[1]
    )
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    confusion_matrix_path = results_dir / "iris_confusion_matrix.csv"
    metrics_path = results_dir / "iris_metrics.csv"
    evaluation_path = results_dir / "iris_model_evaluation.json"

    confusion_table.to_csv(confusion_matrix_path, index_label="actual")
    metrics_table.to_csv(metrics_path, index=False)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "labels": [int(label) for label in labels],
        "confusion_matrix": matrix.astype(int).tolist(),
        "metrics_table": metric_rows,
        "confusion_matrix_path": str(confusion_matrix_path),
        "metrics_path": str(metrics_path),
        "evaluation_path": str(evaluation_path),
    }
    with evaluation_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True, allow_nan=False)

    return result
