"""Recording-group splits and aligned multi-label targets for window experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def recording_groups(identifiers, *, window_ids=False):
    """Canonicalize filenames, optionally removing the final window timestamp."""
    groups = []
    for identifier in identifiers:
        value = str(identifier).replace("\\", "/").rsplit("/", 1)[-1]
        if window_ids:
            stem, separator, timestamp = value.rpartition("_")
            if not separator or not timestamp.isdigit():
                raise ValueError(f"Expected a window identifier ending in _<seconds>: {identifier}")
            value = stem
        groups.append(Path(value).stem)
    return np.asarray(groups)


def labels_for_windows(row_ids, annotations, species):
    """Align semicolon-separated soundscape labels to extracted window order.

    Repeated annotations are combined by logical OR, and unannotated windows are
    zero-filled. Column order follows the official submission vocabulary.
    """
    annotations = annotations.copy()
    end_seconds = pd.to_timedelta(annotations["end"]).dt.total_seconds().astype(int)
    annotations["row_id"] = annotations.filename.str.replace(".ogg", "", regex=False) + "_" + end_seconds.astype(str)
    targets = annotations.primary_label.str.get_dummies(sep=";")
    targets["row_id"] = annotations.row_id
    targets = targets.groupby("row_id").max()
    return targets.reindex(index=list(row_ids), columns=list(species), fill_value=0).fillna(0).to_numpy(dtype=int)


def split_grouped_data(X, Y, groups, X_unlabeled, Y_unlabeled, unlabeled_groups, *, random_state=15):
    """Keep recording groups disjoint and exclude held-out groups from pseudo-labeling.

    Uses the upstream 20% group test split followed by 18.75% validation from
    remaining groups. Fractions describe groups, not necessarily window counts.
    Independent unlabeled-only recordings remain available for pseudo-labeling.
    """
    X, Y, groups = np.asarray(X), np.asarray(Y), np.asarray(groups)
    X_unlabeled, Y_unlabeled = np.asarray(X_unlabeled), np.asarray(Y_unlabeled)
    unlabeled_groups = np.asarray(unlabeled_groups)
    if len(X) != len(Y) or len(X) != len(groups):
        raise ValueError("Labeled features, targets and recording groups must align.")
    if len(X_unlabeled) != len(Y_unlabeled) or len(X_unlabeled) != len(unlabeled_groups):
        raise ValueError("Unlabeled features, targets and recording groups must align.")
    split_test = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=random_state)
    train_val, test = next(split_test.split(X, Y, groups=groups))
    split_val = GroupShuffleSplit(n_splits=1, test_size=.1875, random_state=random_state)
    train_relative, val_relative = next(split_val.split(X[train_val], Y[train_val], groups=groups[train_val]))
    train, val = train_val[train_relative], train_val[val_relative]
    held_out = np.concatenate((groups[val], groups[test]))
    keep = ~np.isin(unlabeled_groups, held_out)
    return (X[train], Y[train], X[val], Y[val], X[test], Y[test],
            X_unlabeled[keep], Y_unlabeled[keep], unlabeled_groups[keep])
