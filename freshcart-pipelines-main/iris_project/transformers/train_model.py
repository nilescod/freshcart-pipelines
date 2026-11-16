from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


if "transformer" not in globals() or "test" not in globals():
    try:
        from mage_ai.data_preparation.decorators import transformer, test
    except ImportError:
        def transformer(func):
            return func

        def test(func):
            return func


@transformer
def train_model(split_df, *args, **kwargs):
    project_root = Path(
        kwargs.get("repo_path")
        or Path(__file__).resolve().parents[1]
    )
    predictions_path = project_root / "results" / "iris_predictions.csv"
    predictions_path.unlink(missing_ok=True)

    if not isinstance(split_df, pd.DataFrame):
        raise TypeError("train_model expects a pandas DataFrame.")
    if "split" not in split_df or "target" not in split_df:
        raise ValueError("Input must contain 'split' and 'target' columns.")

    train_df = split_df[split_df["split"] == "train"]
    test_df = split_df[split_df["split"] == "test"]
    if len(train_df) != 120 or len(test_df) != 30:
        raise ValueError(
            f"Expected 120 training and 30 test rows; got "
            f"{len(train_df)} and {len(test_df)}."
        )

    features = [
        column for column in split_df.columns
        if column not in ("split", "target")
    ]
    if not features:
        raise ValueError("Input must contain at least one feature column.")

    configuration = kwargs.get("configuration") or {}
    runtime_variables = kwargs.get("variables") or {}
    max_depth = kwargs.get("max_depth")
    if max_depth is None:
        max_depth = runtime_variables.get(
            "max_depth",
            configuration.get("max_depth", 3),
        )
    if isinstance(max_depth, bool) or not isinstance(max_depth, int):
        raise ValueError("max_depth must be an integer from 1 through 5.")
    if not 1 <= max_depth <= 5:
        raise ValueError("max_depth must be an integer from 1 through 5.")

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(train_df[features], train_df["target"])
    test_predictions = model.predict(test_df[features])
    test_target = test_df["target"].to_numpy()
    accuracy = float(accuracy_score(test_target, test_predictions))

    return {
        "model": model,
        "accuracy": accuracy,
        "classification_report": classification_report(
            test_target,
            test_predictions,
            output_dict=True,
            zero_division=0,
        ),
        "max_depth": max_depth,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "feature_names": features,
        "test_indices": test_df.index.astype(int).tolist(),
        "test_features": test_df[features].to_numpy().tolist(),
        "test_target": test_target.astype(int).tolist(),
        "actual_labels": test_target.astype(int).tolist(),
        "predictions": test_predictions.astype(int).tolist(),
    }


@test
def test_output(output, *args):
    assert output["train_rows"] == 120
    assert output["test_rows"] == 30
    assert 0 <= output["accuracy"] <= 1


@test
def test_accuracy(output, *args):
    assert 1 <= output["max_depth"] <= 5
    assert hasattr(output["model"], "predict")
    assert len(output["actual_labels"]) == 30
    assert len(output["test_target"]) == 30
    assert len(output["predictions"]) == 30
    measured_accuracy = accuracy_score(output["actual_labels"], output["predictions"])
    assert abs(output["accuracy"] - measured_accuracy) < 1e-12, (
        "Reported accuracy must match predictions against the held-out test targets."
    )
    assert output["classification_report"]["accuracy"] == output["accuracy"]
    print(f"PASS: test_accuracy ({output['accuracy']:.4f})")
