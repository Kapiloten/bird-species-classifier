"""Rebuild portfolio tables and figures from archived predictions/confusion counts.

No model fitting, dataset download, audio access, or unpickling is performed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluation import duplicate_feature_rows, metrics_from_confusion
from project_paths import PROJECT_ROOT, project_path
from single_label import merge_manual_features


def markdown_table(frame):
    """Render a small DataFrame as Markdown without an extra tabulate dependency."""
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in row) + " |")
    return "\n".join(lines)


def collect_results(root):
    """Validate saved metrics/predictions against counts and retain provenance."""
    rows, class_rows, matrices, inputs = [], [], {}, set()
    source = root / "ressource/results/ml_classic"
    for metric_path in sorted(source.glob("*/*/metrics.csv")):
        family, run = metric_path.relative_to(source).parts[:2]
        cm_path = metric_path.with_name("confusion_matrix.csv")
        matrix = pd.read_csv(cm_path, index_col=0)
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str)
        scores, per_class = metrics_from_confusion(matrix)
        saved = pd.read_csv(metric_path).iloc[0]
        for old, new in (("test_accuracy", "accuracy"), ("test_f1_macro", "f1_macro")):
            if not np.isclose(float(saved[old]), scores[new], rtol=1e-10, atol=1e-12):
                raise ValueError(f"{metric_path}: {old} does not match confusion counts.")
        inputs.update((metric_path, cm_path))
        test_cm_path = cm_path.with_name("confusion_matrix_test.csv")
        if test_cm_path.exists():
            pd.testing.assert_frame_equal(matrix, pd.read_csv(test_cm_path, index_col=0), check_names=False)
            inputs.add(test_cm_path)
        prediction_path = cm_path.with_name("predictions_test.csv")
        if prediction_path.exists():
            from sklearn.metrics import confusion_matrix

            predictions = pd.read_csv(prediction_path)
            reconstructed = confusion_matrix(predictions.label, predictions.prediction, labels=matrix.index)
            if len(predictions) != scores["n_test"] or not np.array_equal(reconstructed, matrix.to_numpy()):
                raise ValueError(f"{prediction_path}: predictions disagree with confusion counts.")
            inputs.add(prediction_path)
        caveat = "identical features across train/test" if family == "manual_embedding" else "pre-split augmentation; mixed taxa"
        rows.append({"family": family, "run": run, "model": saved["model"], **scores, "caveat": caveat})
        per_class.insert(0, "run", run)
        per_class.insert(0, "family", family)
        class_rows.append(per_class)
        matrices[family, run] = matrix
    if not rows:
        raise ValueError(f"No archived results found under {source}")
    return pd.DataFrame(rows), pd.concat(class_rows, ignore_index=True), matrices, inputs


def confusion_pairs(matrix, names, limit=12):
    """Rank directed off-diagonal errors, with deterministic tie ordering."""
    rows = [{"true_label": str(actual), "predicted_label": str(predicted),
             "true_name": names.get(str(actual), str(actual)),
             "predicted_name": names.get(str(predicted), str(predicted)),
             "count": int(matrix.loc[actual, predicted])}
            for actual in matrix.index for predicted in matrix.columns
            if actual != predicted and matrix.loc[actual, predicted] > 0]
    frame = pd.DataFrame(rows, columns=["true_label", "predicted_label", "true_name", "predicted_name", "count"])
    return frame.sort_values(["count", "true_label", "predicted_label"], ascending=[False, True, True]).head(limit)


def create_figures(output, summary, per_class, matrices, names, pairs):
    """Plot separately scoped comparisons and interpretable confusion diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = output / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, family, title in zip(axes, ("manual_embedding", "perch_embedding"),
                               ("Manual: 6 bird classes / 596 test files",
                                "Perch historical: 206 taxa / 7,118 observations")):
        frame = summary[summary.family == family].sort_values("f1_macro")
        y = np.arange(len(frame))
        ax.barh(y - .18, frame.accuracy, height=.34, label="Accuracy", color="#2563a6")
        ax.barh(y + .18, frame.f1_macro, height=.34, label="Macro F1", color="#e18a23")
        ax.set_yticks(y, frame.run.str.replace("_", " "))
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Score (historical protocol)")
    axes[0].legend(loc="lower right")
    fig.suptitle("Separate evaluation scopes — both have leakage caveats", fontsize=13)
    save(fig, "model_comparison.png")

    matrix = matrices["manual_embedding", "svm_rbf"]
    normalized = matrix.div(matrix.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    labels = [names.get(label, label) for label in matrix.index]
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix.iloc[i, j]}\n{normalized.iloc[i, j]:.1%}", ha="center", va="center",
                    fontsize=9, color="white" if normalized.iloc[i, j] > .5 else "#17222e")
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("True species")
    ax.set_title("Manual SVM: counts and row-normalized recall\nHistorical split; seven duplicate-feature train/test pairs")
    fig.colorbar(im, ax=ax, label="Proportion of true class", fraction=.04)
    save(fig, "manual_svm_confusion.png")

    frame = per_class[(per_class.family == "manual_embedding") & (per_class.run == "svm_rbf")].sort_values("f1")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([names.get(label, label) for label in frame.label], frame.f1, color="#2563a6")
    for i, row in enumerate(frame.itertuples()):
        ax.text(row.f1 + .01, i, f"{row.f1:.3f} (n={row.support})", va="center")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-score")
    ax.set_title("Manual SVM: per-species performance on the historical split")
    save(fig, "manual_class_f1.png")

    frame = pairs.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(frame.true_name + " → " + frame.predicted_name, frame["count"], color="#8a5a9c")
    ax.set_xlabel("Misclassified observations (directed pairs)")
    ax.set_title("Perch soft voting: historical confusions\nPre-split augmentation; independent-test interpretation is limited")
    save(fig, "perch_confusions.png")


def generate_report(root, output):
    """Write exact tables, four figures, duplicate diagnostics and input hashes."""
    summary, per_class, matrices, inputs = collect_results(root)
    taxonomy_path = root / "ressource/raw_data/taxonomy.csv"
    taxonomy = pd.read_csv(taxonomy_path, dtype={"primary_label": str})
    names = taxonomy.set_index("primary_label").common_name.to_dict()
    inputs.add(taxonomy_path)
    paths = [root / "ressource/features" / name for name in ("features.csv", "mfcc_features.csv")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        features, columns = merge_manual_features(*(pd.read_csv(p) for p in paths))
    inputs.update(paths)
    manifest_path = root / "splits/audio_split_65_15_20.csv"
    manifest = pd.read_csv(manifest_path)
    pd.testing.assert_frame_equal(features[manifest.columns], manifest)
    inputs.add(manifest_path)
    duplicates = duplicate_feature_rows(features, columns)
    pairs = confusion_pairs(matrices["perch_embedding", "soft_voting"], names)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    per_class.to_csv(output / "per_class_metrics.csv", index=False)
    duplicates.to_csv(output / "duplicate_features.csv", index=False)
    pairs.to_csv(output / "perch_top_confusions.csv", index=False)
    split_counts = pd.crosstab(features.label, features.split).reindex(columns=["train", "val", "test"])
    split_counts.to_csv(output / "split_counts.csv")
    provenance = {"operation": "recompute metrics/plots from archived integer confusion matrices; no training",
                  "averaging": "macro over all listed labels; zero_division=0",
                  "inputs_sha256": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(inputs)}}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    text = ["# Historical result summary", "",
            "Generated with `uv run python scripts/summarize_results.py`. No model training was performed.", "",
            "Manual scores contain seven exact feature-duplicate pairs across train/test. Perch scores use "
            "pre-split synthetic augmentation and mixed taxa. These scopes are not directly comparable.", ""]
    for family in ("manual_embedding", "perch_embedding"):
        text.extend([f"## {family}", "", markdown_table(summary.loc[summary.family == family,
                     ["run", "n_classes", "n_test", "accuracy", "precision_macro", "recall_macro", "f1_macro"]]), ""])
    text.extend(["Exact values are in `summary.csv`; inputs and SHA-256 hashes are in `provenance.json`.", "",
                 "Voting variants 2/3 and LightGBM are archived experiments; their complete runnable "
                 "configurations are not preserved in the current executable code.", ""])
    (output / "summary.md").write_text("\n".join(text), encoding="utf-8")
    create_figures(output, summary, per_class, matrices, names, pairs)
    print(f"Validated {len(summary)} archived runs; wrote tables and figures to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results", help="Destination relative to the repository root.")
    args = parser.parse_args()
    generate_report(PROJECT_ROOT, project_path(args.output_dir))


if __name__ == "__main__":
    main()
