"""Synthetic regression tests: no BirdCLEF recordings or embeddings required."""

import numpy as np
import pandas as pd
import pytest

from scripts.manual_embedding.create_audio_split import assign_splits
from scripts.project_paths import PROJECT_ROOT, first_existing_path, project_path
from scripts.single_label import increase_rare_birds, load_data2, merge_manual_features, split_embeddings


@pytest.fixture
def feature_tables():
    rows = [{"filepath": f"train_audio/bird/{i}.ogg", "filename": f"{i}.ogg", "label": "bird",
             "common_name": "Example bird", "scientific_name": "Avis example",
             "split": split, "duration": float(i + 1)}
            for i, split in enumerate(["train", "train", "val", "test"])]
    acoustic = pd.DataFrame(rows)
    mfcc = acoustic.drop(columns="duration").assign(mfcc_1_mean=[1., 2., 3., 4.])
    return acoustic, mfcc


def test_paths_are_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert project_path("classes.txt") == PROJECT_ROOT / "classes.txt"
    assert project_path(tmp_path) == tmp_path
    assert first_existing_path(tmp_path / "missing", tmp_path) == tmp_path
    with pytest.raises(ValueError, match="At least one"):
        first_existing_path()


def test_recording_split_is_deterministic_and_disjoint():
    files = list(range(100))
    split = assign_splits(files, seed=42)
    assert split == assign_splits(files, seed=42)
    assert split != assign_splits(files, seed=7)
    assert files == list(range(100))
    partitions = {name: {p for p, part in split if part == name} for name in ["train", "val", "test"]}
    assert [len(p) for p in partitions.values()] == [65, 15, 20]
    assert len(set.union(*partitions.values())) == 100
    assert not partitions["train"] & partitions["test"]


def test_merge_keeps_order_and_metadata_out_of_features(feature_tables, tmp_path):
    acoustic, mfcc = feature_tables
    merged, columns = merge_manual_features(acoustic, mfcc.iloc[::-1])
    assert merged.filepath.tolist() == acoustic.filepath.tolist()
    assert columns == ["duration", "mfcc_1_mean"]
    acoustic.to_csv(tmp_path / "features.csv", index=False)
    mfcc.to_csv(tmp_path / "mfcc_features.csv", index=False)
    x_train, y_train, x_val, y_val, x_test, y_test = load_data2(tmp_path)
    assert [len(y_train), len(y_val), len(y_test)] == [2, 1, 1]
    assert list(x_train) == columns
    assert x_val.iloc[0].duration == 3
    assert x_test.iloc[0].duration == 4


@pytest.mark.parametrize("issue", ["duplicate", "missing_row", "wrong_split", "missing_column", "infinite", "unseen_class"])
def test_merge_rejects_corrupt_or_leaking_tables(feature_tables, issue):
    acoustic, mfcc = feature_tables
    if issue == "duplicate":
        acoustic = pd.concat([acoustic, acoustic.iloc[[0]]])
    elif issue == "missing_row":
        mfcc = mfcc.drop(index=1)
    elif issue == "wrong_split":
        mfcc.loc[0, "split"] = "test"
    elif issue == "missing_column":
        mfcc = mfcc.drop(columns="label")
    elif issue == "infinite":
        acoustic.loc[0, "duration"] = np.inf
    elif issue == "unseen_class":
        acoustic.loc[3, "label"] = mfcc.loc[3, "label"] = "unseen"
    with pytest.raises(ValueError):
        merge_manual_features(acoustic, mfcc)


def test_merge_warns_without_changing_historical_duplicates(feature_tables):
    acoustic, mfcc = feature_tables
    acoustic.loc[3, "duration"] = acoustic.loc[0, "duration"]
    mfcc.loc[3, "mfcc_1_mean"] = mfcc.loc[0, "mfcc_1_mean"]
    with pytest.warns(UserWarning, match="Identical feature vectors"):
        merged, _ = merge_manual_features(acoustic, mfcc)
    assert len(merged) == 4


def test_missing_feature_files_are_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="extract_features.py"):
        load_data2(tmp_path)


def test_augmentation_uses_local_seed_and_does_not_mutate_originals():
    x = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array(["a"] * 5 + ["b"])
    before = x.copy()
    np.random.seed(123)
    expected_global_draw = np.random.random()
    np.random.seed(123)
    first, labels = increase_rare_birds(x, y, random_state=42)
    assert np.random.random() == expected_global_draw
    second, _ = increase_rare_birds(x, y, random_state=42)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(x, before)
    assert list(labels).count("b") == 5


def test_perch_evaluation_contains_only_originals_and_is_reproducible():
    x = np.arange(20, dtype=float).reshape(10, 2)
    y = np.array(["a"] * 5 + ["b"] * 5)
    split = split_embeddings(x, y, random_state=12)
    repeated = split_embeddings(x, y, random_state=12)
    for part, other in zip(split, repeated):
        np.testing.assert_array_equal(part, other)
    x_train, y_train, x_val, y_val, x_test, y_test = split
    originals = {tuple(row) for row in x}
    val, test = {tuple(row) for row in x_val}, {tuple(row) for row in x_test}
    original_train = {tuple(row) for row in x_train if tuple(row) in originals}
    assert val.issubset(originals) and test.issubset(originals)
    assert not val & test and not original_train & (val | test)
    assert len(original_train | val | test) == len(x)
    assert len(x_train) == 10  # Each training class reaches five; held-out data does not grow.
    assert len(y_val) + len(y_test) == 4
    assert set(y_train) == set(y_val) == set(y_test) == {"a", "b"}


def test_perch_excludes_unevaluable_classes_before_augmentation():
    x = np.arange(22, dtype=float).reshape(11, 2)
    y = np.array(["a"] * 5 + ["b"] * 5 + ["rare"])
    with pytest.warns(UserWarning, match="rare"):
        result = split_embeddings(x, y)
    assert all("rare" not in part for part in result[1::2])
    with pytest.raises(ValueError, match="at least five"):
        split_embeddings(x, y, min_originals=2)
