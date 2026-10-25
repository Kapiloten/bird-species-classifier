"""Historical single-label baselines; current multi-label loaders live in utils.py."""

import hashlib
import json
import platform
import warnings
from importlib.metadata import version

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import numpy as np
from sklearn.model_selection import train_test_split

try:
    from scripts.project_paths import FEATURES_DIR, PROCESSED_DATA_DIR, project_path
except ModuleNotFoundError:
    from project_paths import FEATURES_DIR, PROCESSED_DATA_DIR, project_path


META_COLUMNS = ["filepath", "filename", "label", "common_name", "scientific_name", "split"]


def increase_rare_birds(X, y, min_threshold=5, noise=0.02, random_state=12):
    """Augment rare *training* classes with Gaussian noise, using a local RNG.

    Call only after splitting original observations. The noise scale is estimated
    from this training array, never from validation or test data.
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    rare_birds = classes[counts < min_threshold]

    if len(rare_birds) == 0:
        return X, y

    new_X = []
    new_y = []

    global_std = np.std(X)
    noise_calc = global_std * noise

    for bird in rare_birds:
        existing_birds = X[y == bird]
        nb_existing = len(existing_birds)

        nb_needed = min_threshold - nb_existing

        for _ in range(nb_needed):
            base_vect = existing_birds[rng.integers(0, nb_existing)]

            noise_vect = rng.normal(loc=0.0, scale=noise_calc, size=base_vect.shape)

            mutant = base_vect + noise_vect

            new_X.append(mutant)
            new_y.append(bird)

    X_final = np.vstack([X, np.array(new_X)])
    y_final = np.concatenate([y, np.array(new_y)])

    return X_final, y_final

def split_embeddings(X, y, random_state=12, min_originals=5):
    """Split original Perch observations 65/15/20, then augment training only.

    Labels with fewer than ``min_originals`` original recordings are excluded
    explicitly: synthetic siblings cannot serve as independent test observations.
    This is a recording-level protocol; it does not establish site independence.
    """
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("Expected a 2-D embedding array and one label per row.")
    if min_originals < 5:
        raise ValueError("min_originals must be at least five for the three-way split.")
    if not np.isfinite(X).all() or pd.isna(y).any():
        raise ValueError("Embeddings and labels must not contain non-finite/missing values.")
    classes, counts = np.unique(y, return_counts=True)
    excluded = classes[counts < min_originals]
    if len(excluded):
        warnings.warn(
            "Excluded classes with too few original recordings: " + ", ".join(map(str, excluded)),
            UserWarning,
            stacklevel=2,
        )
    keep = ~np.isin(y, excluded)
    X, y = X[keep], y[keep]
    if len(np.unique(y)) < 2:
        raise ValueError("At least two eligible classes are required; collect more originals.")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1875, stratify=y_temp, random_state=random_state
    )
    expected = set(y)
    if any(set(part) != expected for part in (y_train, y_val, y_test)):
        raise ValueError("Some classes are absent from a partition; increase min_originals.")
    X_train, y_train = increase_rare_birds(X_train, y_train, random_state=random_state)
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_data(base_path=None, random_state=12, min_originals=5):
    """Load Perch arrays and apply the corrected train-only augmentation protocol."""
    base_path = project_path(base_path) if base_path is not None else PROCESSED_DATA_DIR
    embeddings_path = base_path / "X_embeddings_perch.npy"
    labels_path = base_path / "y_labels.npy"

    if not embeddings_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Perch embeddings are missing. Run "
            "scripts/perch_embedding/extract_recording_embeddings.py first."
        )

    X = np.load(embeddings_path)
    Y = np.load(labels_path)

    return split_embeddings(X, Y, random_state=random_state, min_originals=min_originals)


def merge_manual_features(audio_features, mfcc_features):
    """Join one row per recording, rejecting missing/mismatched metadata.

    Metadata never enters the numeric feature matrix. Identical numeric vectors
    across partitions trigger a warning; historical observations are kept intact.
    """
    for name, frame in (("acoustic", audio_features), ("MFCC", mfcc_features)):
        missing = set(META_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} table is missing metadata columns: {sorted(missing)}")
        if frame[META_COLUMNS].isna().any().any():
            raise ValueError(f"{name} table contains missing metadata.")
        if frame["filepath"].duplicated().any():
            raise ValueError(f"{name} table contains duplicate recording paths (possible leakage).")
        if set(frame["split"]) != {"train", "val", "test"}:
            raise ValueError(f"{name} table must contain only train, val and test partitions.")
    overlap = (set(audio_features) & set(mfcc_features)) - set(META_COLUMNS)
    if overlap:
        raise ValueError(f"Overlapping feature names: {sorted(overlap)}")
    data = audio_features.merge(
        mfcc_features, on=META_COLUMNS, how="outer", validate="one_to_one", indicator=True,
        sort=False,
    )
    if not data["_merge"].eq("both").all():
        raise ValueError("Feature tables disagree on recordings, labels or split assignments.")
    # Preserve acoustic-table order, which also matches the committed manifest.
    data = data.drop(columns="_merge").set_index("filepath").loc[audio_features.filepath].reset_index()
    columns = [c for c in data if c not in META_COLUMNS]
    if not columns or not all(pd.api.types.is_numeric_dtype(data[c]) for c in columns):
        raise ValueError("Feature columns must be numeric and nonempty.")
    if np.isinf(data[columns].to_numpy()).any():
        raise ValueError("Infinite feature values are not supported; NaNs can be imputed.")
    known = set(data.loc[data.split == "train", "label"])
    if not set(data.label).issubset(known):
        raise ValueError("Validation/test contains classes absent from training.")
    duplicates = data[data.duplicated(columns, keep=False)]
    if not duplicates.empty and duplicates.groupby(columns, dropna=False)["split"].nunique().gt(1).any():
        warnings.warn(
            "Identical feature vectors occur across partitions; historical scores may be optimistic. "
            "Run scripts/summarize_results.py and review results/duplicate_features.csv.",
            UserWarning,
            stacklevel=2,
        )
    return data, columns


def load_data2(features_dir=None):
    """Load the fixed manual split; return X/y for train, validation and test.

    ``features_dir`` is relative to the project root unless absolute. Both CSVs
    must describe exactly the same recordings. NaNs are left to train-fitted
    imputers; identity and split inconsistencies fail before any model is fitted.
    """
    base_path = project_path(features_dir) if features_dir is not None else FEATURES_DIR
    audio_features_path = base_path / "features.csv"
    mfcc_features_path = base_path / "mfcc_features.csv"

    if not audio_features_path.exists() or not mfcc_features_path.exists():
        raise FileNotFoundError(
            "Manual feature files are missing. Run "
            "scripts/manual_embedding/extract_features.py and "
            "scripts/manual_embedding/extract_mfcc_features.py first."
        )

    audio_features = pd.read_csv(audio_features_path)
    mfcc_features = pd.read_csv(mfcc_features_path)
    data, feature_columns = merge_manual_features(audio_features, mfcc_features)

    train = data[data["split"] == "train"]
    val = data[data["split"] == "val"]
    test = data[data["split"] == "test"]

    return (
        train[feature_columns],
        train["label"],
        val[feature_columns],
        val["label"],
        test[feature_columns],
        test["label"],
    )


def save_results(modelName, path, y_test, predictions, test_accuracy, test_f1, *, configuration=None):
    """Export full-precision metrics, predictions, confusion counts and run metadata.

    Supplied accuracy/F1 are checked against predictions to catch stale arguments.
    Historical artifacts are preserved by choosing a new ``--output-dir``.
    """
    y_test, predictions = np.asarray(y_test), np.asarray(predictions)
    actual_accuracy = accuracy_score(y_test, predictions)
    actual_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    if not np.isclose(test_accuracy, actual_accuracy) or not np.isclose(test_f1, actual_f1):
        raise ValueError("Supplied metrics disagree with the predictions.")
    results_dir = project_path(path)
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        labels = list(np.unique(np.concatenate([y_test, predictions])))
        scores = classification_report(y_test, predictions, labels=labels, zero_division=0, output_dict=True)
        metrics = pd.DataFrame(
            [{"model": modelName, "test_accuracy": actual_accuracy, "test_f1_macro": actual_f1,
              "test_precision_macro": scores["macro avg"]["precision"],
              "test_recall_macro": scores["macro avg"]["recall"], "n_test": len(y_test)}]
        )
        metrics.to_csv(results_dir / "metrics.csv", index=False)

        report = classification_report(y_test, predictions, labels=labels, zero_division=0)
        (results_dir / "classification_report.txt").write_text(report, encoding="utf-8")
        (results_dir / "classification_report.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
        pd.DataFrame({"label": y_test, "prediction": predictions}).to_csv(
            results_dir / "predictions_test.csv", index=False
        )
        config = configuration or {}
        sources = {}
        for source in config.get("input_files", []):
            source_path = project_path(source)
            sources[str(source)] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        run = {"configuration": config, "input_sha256": sources, "python": platform.python_version(),
               "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn")}}
        (results_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")

        matrix = confusion_matrix(y_test, predictions, labels=labels)
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
            results_dir / "confusion_matrix.csv"
        )
    except PermissionError as error:
        raise PermissionError(
            f"Cannot write results to {results_dir}. Close any open result files "
            "or choose another directory with --output-dir."
        ) from error
