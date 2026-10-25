"""Compute exact single-label metrics from saved confusion counts."""

import numpy as np
import pandas as pd


def metrics_from_confusion(matrix):
    """Return aggregate and per-class metrics without expanding observations.

    Rows represent true labels; columns represent predicted labels. Zero-denominator
    precision, recall and F1 are defined as zero, matching sklearn's zero_division=0.
    All listed classes participate in macro averages, including zero-support ones.
    """
    if matrix.empty or list(matrix.index) != list(matrix.columns):
        raise ValueError("Confusion matrix must have matching, ordered row/column labels.")
    if not matrix.index.is_unique:
        raise ValueError("Confusion matrix labels must be unique.")
    values = matrix.to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.equal(values, np.floor(values)).all() or values.sum() == 0):
        raise ValueError("Confusion counts must be finite nonnegative integers with nonzero total.")
    support = values.sum(axis=1)
    predicted = values.sum(axis=0)
    correct = np.diag(values)
    precision = np.divide(correct, predicted, out=np.zeros(len(correct)), where=predicted != 0)
    recall = np.divide(correct, support, out=np.zeros(len(correct)), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(correct)), where=(precision + recall) != 0)
    per_class = pd.DataFrame({"label": matrix.index, "precision": precision,
                              "recall": recall, "f1": f1, "support": support.astype(int)})
    scores = {"n_test": int(support.sum()), "n_classes": len(matrix),
              "accuracy": float(correct.sum() / support.sum()),
              "precision_macro": float(precision.mean()), "recall_macro": float(recall.mean()),
              "f1_macro": float(f1.mean()),
              "precision_weighted": float(precision @ support / support.sum()),
              "recall_weighted": float(recall @ support / support.sum()),
              "f1_weighted": float(f1 @ support / support.sum())}
    return scores, per_class


def duplicate_feature_rows(data, feature_columns):
    """List exact numeric duplicates and whether each group crosses partitions.

    This is a feature-level diagnostic, not an audio fingerprint or proof of
    waveform equality. Stable group IDs follow the input recording order.
    """
    duplicates = data[data.duplicated(feature_columns, keep=False)]
    rows = []
    for group_id, (_, group) in enumerate(
        duplicates.groupby(feature_columns, dropna=False, sort=False), start=1
    ):
        cross_partition = group["split"].nunique() > 1
        for row in group.itertuples():
            rows.append({"group": group_id, "filepath": row.filepath, "label": row.label,
                         "split": row.split, "cross_partition": cross_partition})
    return pd.DataFrame(rows, columns=["group", "filepath", "label", "split", "cross_partition"])
