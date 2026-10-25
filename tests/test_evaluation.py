import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from scripts.evaluation import duplicate_feature_rows, metrics_from_confusion
from scripts.single_label import save_results


def test_confusion_metrics_match_sklearn_with_unpredicted_and_unsupported_classes():
    actual = ["a", "a", "a", "b", "b"]
    predicted = ["a", "a", "c", "a", "a"]
    labels = ["a", "b", "c"]
    matrix = pd.DataFrame(confusion_matrix(actual, predicted, labels=labels), index=labels, columns=labels)
    scores, per_class = metrics_from_confusion(matrix)
    expected = classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0)
    assert scores["accuracy"] == accuracy_score(actual, predicted)
    for average in ("macro", "weighted"):
        for key, report_key in (("precision", "precision"), ("recall", "recall"), ("f1", "f1-score")):
            assert scores[f"{key}_{average}"] == pytest.approx(expected[f"{average} avg"][report_key])
    assert per_class.support.tolist() == [3, 2, 0]


@pytest.mark.parametrize("values", [[[0, 0], [0, 0]], [[1, -1], [0, 2]],
                                    [[1, np.nan], [0, 2]], [[1, .5], [0, 2]]])
def test_invalid_confusion_counts_fail(values):
    with pytest.raises(ValueError):
        metrics_from_confusion(pd.DataFrame(values, index=["a", "b"], columns=["a", "b"]))


def test_misordered_matrix_labels_fail():
    with pytest.raises(ValueError, match="matching"):
        metrics_from_confusion(pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["b", "a"]))


def test_duplicate_diagnostic_distinguishes_within_and_across_partitions():
    frame = pd.DataFrame({"filepath": ["a", "b", "c", "d"], "label": ["bird"] * 4,
                          "split": ["train", "test", "train", "train"], "value": [1, 1, 2, 2]})
    result = duplicate_feature_rows(frame, ["value"])
    assert result.cross_partition.tolist() == [True, True, False, False]


def test_exports_preserve_predicted_only_labels_and_provenance(tmp_path):
    actual, predicted = ["a", "a", "b"], ["a", "c", "c"]
    source = tmp_path / "input.csv"
    source.write_text("example\n1\n", encoding="utf-8")
    out = tmp_path / "run"
    save_results("example", out, actual, predicted, accuracy_score(actual, predicted),
                 f1_score(actual, predicted, average="macro"),
                 configuration={"seed": 42, "input_files": [str(source)]})
    matrix = pd.read_csv(out / "confusion_matrix.csv", index_col=0)
    assert matrix.to_numpy().sum() == len(actual)
    assert list(matrix.columns) == ["a", "b", "c"]
    run = json.loads((out / "run.json").read_text())
    assert run["configuration"]["seed"] == 42
    assert len(run["input_sha256"][str(source)]) == 64
    assert pd.read_csv(out / "predictions_test.csv").prediction.tolist() == predicted


def test_stale_export_metrics_are_rejected_before_writing(tmp_path):
    out = tmp_path / "bad_run"
    with pytest.raises(ValueError, match="disagree"):
        save_results("example", out, ["a", "b"], ["a", "a"], 1., 1.)
    assert not out.exists()
