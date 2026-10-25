# Reproducible result artifacts

Rebuild from the repository root:

```bash
uv sync --locked
uv run --locked python scripts/summarize_results.py
```

This reads committed tables, predictions and confusion matrices. It does not train
models, download data, read audio or load serialized estimators.

| Artifact | Contents |
| --- | --- |
| [summary.md](summary.md) / [summary.csv](summary.csv) | Ten archived runs; exact accuracy, macro/weighted precision, recall and F1 |
| [per_class_metrics.csv](per_class_metrics.csv) | Exact class-level metrics and support for every run |
| [split_counts.csv](split_counts.csv) | Manual train/validation/test counts by bird species |
| [duplicate_features.csv](duplicate_features.csv) | 11 pairs of identical 51-dimensional manual feature vectors; seven train/test pairs |
| [perch_top_confusions.csv](perch_top_confusions.csv) | Largest directed confusions for historical Perch soft voting |
| [provenance.json](provenance.json) | SHA-256 fingerprints of all numeric inputs |
| [figures/](figures/) | Separate-scope model comparison, manual SVM confusion matrix, per-species F1 and Perch confusion pairs |

## Interpretation

All scores are internal, historical evaluations. Manual: six bird classes and
596 test files, with identical feature vectors crossing train/test. Perch:
206 mixed-taxon classes and 7,118 test observations, with augmentation performed
before splitting in the historical loader. The Perch matrices do not identify
which observations were synthetic. They cannot be turned into clean test scores
after the fact. Their ranking is descriptive, since historical augmentation was
unseeded and individual runs are not guaranteed to share exactly the same split.

Metrics are recomputed from integer counts and checked against the archived
accuracy and macro F1. Manual predictions are checked against their confusion
matrices; both copies of each manual matrix are also checked. Text reports round
to two decimals and are not the numeric source. Macro averaging includes every
matrix label; undefined per-class values are set to zero.

The generator fails on inconsistent counts or metadata. It verifies that the two
feature tables exactly match the committed manual manifest. Historical artifacts
under `ressource/results/` are read only. New training commands write to ignored
`results/runs/` by default; pass a distinct `--output-dir` to retain multiple runs.

The corrected Perch protocol and a duplicate-aware manual benchmark both need new
experiments. No corrected Perch score is reported. See [the audit](../docs/AUDIT.md)
and [reproduction guide](../docs/REPRODUCIBILITY.md).

The current multi-label workflow from `main` is preserved separately. Its PNGs
remain under `ressource/results/ml_classic/`; they are not substituted for numeric
prediction/count artifacts in this report. Source metadata moved to
`ressource/raw_data/`; input fingerprints reflect the merged layout. Serialized
models removed upstream remain accessible in Git history rather than being restored.
