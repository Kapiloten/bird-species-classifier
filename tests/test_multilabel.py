"""Protect the current main workflow while integrating historical portfolio results."""

import numpy as np
import pandas as pd
import pytest

from scripts.multilabel_data import labels_for_windows, recording_groups, split_grouped_data
from scripts.utils import load_csv_features, load_data
from scripts import utils


def test_recording_groups_keep_underscores_and_normalize_paths():
    assert recording_groups(["some_recording_5", "some_recording_10"], window_ids=True).tolist() == ["some_recording"] * 2
    assert recording_groups(["folder/bird.ogg", r"folder\bird.ogg"]).tolist() == ["bird"] * 2
    with pytest.raises(ValueError, match="window identifier"):
        recording_groups(["missing_timestamp"], window_ids=True)


def test_soundscape_targets_align_and_or_duplicate_annotations():
    annotations = pd.DataFrame({"filename": ["site.ogg"] * 3,
                                "end": ["00:00:05", "00:00:05", "01:00:05"],
                                "primary_label": ["a", "b", "b"]})
    targets = labels_for_windows(["site_3605", "site_10", "site_5"], annotations, ["b", "a", "unused"])
    np.testing.assert_array_equal(targets, [[1, 0, 0], [0, 0, 0], [1, 1, 0]])


def test_grouped_split_excludes_test_and_validation_recordings_from_pseudo_pool():
    groups = np.repeat(np.arange(20).astype(str), 2)
    X = np.column_stack((groups.astype(int), np.arange(len(groups))))
    Y = np.tile([1, 0], (len(groups), 1))
    unlabeled_groups = np.concatenate((np.arange(20).astype(str), ["independent"]))
    X_unlabeled = np.arange(42).reshape(21, 2)
    Y_unlabeled = np.zeros((21, 2), dtype=int)
    result = split_grouped_data(X, Y, groups, X_unlabeled, Y_unlabeled, unlabeled_groups)
    train, val, test = (set(result[i][:, 0].astype(str)) for i in (0, 2, 4))
    assert not train & val and not train & test and not val & test
    assert not set(result[8]) & (val | test)
    assert "independent" in result[8]
    for first, second in zip(result, split_grouped_data(X, Y, groups, X_unlabeled, Y_unlabeled, unlabeled_groups)):
        np.testing.assert_array_equal(first, second)


def test_both_current_loaders_accept_small_local_fixtures_outside_repository(tmp_path, monkeypatch):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    submission = raw / "sample_submission.csv"
    pd.DataFrame(columns=["row_id", "a", "b"]).to_csv(submission, index=False)
    annotations = pd.DataFrame({"filename": [f"s{i}.ogg" for i in range(10)],
                                "start": ["00:00:00"] * 10, "end": ["00:00:05"] * 10,
                                "primary_label": ["b"] * 10})
    annotation_path = raw / "train_soundscapes_labels.csv"
    annotations.to_csv(annotation_path, index=False)
    np.save(processed / "X_embeddings_perch_train_audio.npy", np.arange(40).reshape(20, 2))
    np.save(processed / "y_labels_train_audio.npy", np.array(["a"] * 20))
    np.save(processed / "row_ids_train_audio.npy", np.array([f"a{i}_5" for i in range(20)]))
    np.save(processed / "X_embeddings_perch_train_soundscapes.npy", np.arange(40).reshape(20, 2))
    np.save(processed / "row_ids_train_soundscapes.npy", np.array([f"s{i}_{end}" for i in range(10) for end in (5, 10)]))
    monkeypatch.chdir(tmp_path)
    perch = load_data(processed, raw)
    assert len(perch) == 9
    assert len(perch[0]) + len(perch[2]) + len(perch[4]) == 30
    assert all(part.shape[1] == 2 for part in perch[1:8:2])
    # Absolute and Windows-style audio paths must still yield the parent class.
    rows = [{"row_id": f"a{i}_5", "source": "train_audio", "filepath": rf"C:\data\train_audio\a\a{i}.ogg",
             "filename": f"a{i}.ogg", "duration": 5, "mfcc_1_mean": float(i)} for i in range(20)]
    rows.extend({"row_id": f"s{i}_{end}", "source": "train_soundscapes", "filepath": f"train_soundscapes/s{i}.ogg",
                 "filename": f"s{i}.ogg", "duration": 5, "mfcc_1_mean": float(i)} for i in range(10) for end in (5, 10))
    feature_path = tmp_path / "windows.csv"
    pd.DataFrame(rows).to_csv(feature_path, index=False)
    manual = load_csv_features(feature_path, annotation_path, submission)
    assert len(manual) == 9
    assert len(manual[0]) + len(manual[2]) + len(manual[4]) == 30
    assert manual[0].shape[1] == 1  # Metadata duration is excluded, following upstream.


def test_pr_comparison_supports_curves_with_different_threshold_counts(monkeypatch):
    truth = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    base = np.array([[.9, .2], [.1, .8], [.7, .6], [.4, .3]])
    final = np.array([[.8, .2], [.2, .8], [.8, .8], [.2, .2]])
    saved = []
    monkeypatch.setattr(utils, "_save_figure", lambda path, **kwargs: saved.append(path))
    monkeypatch.setattr(utils.plt, "show", lambda: None)
    utils.plot_pr_comparison(truth, base, final, "synthetic")
    assert len(saved) == 1
    utils.plt.close("all")
