# Integration with current main

The first portfolio audit was based on `42143d3b`. Pull request #1 subsequently
targeted `main` at `0c1a794d`, which introduced a different multi-label experiment
workflow and moved source metadata. Conflicts are resolved by merging both
histories, not rebasing or force-pushing.

## Resolution choices

- Current `scripts/utils.py` retains the multi-label loader API (nine return
  arrays), annotation alignment and plotting functions. GroupShuffleSplit and
  the original model/search settings are preserved.
- Audited historical utilities move to `scripts/single_label.py`; the three
  whole-recording manual baselines import this module explicitly.
- Historical Perch commands are now `extract_recording_embeddings.py` (averaged
  recording vectors) and `train_single_label.py` (corrected soft voting).
- Current `perch_embedding/embedding_perch.py` preserves frame-wise embeddings;
  `perch_embedding/main.py` preserves the multi-label logistic/pseudo-label
  workflow. Extraction is guarded, configurable and records failures. Both data
  sources can be requested explicitly.
- Upstream manual Nystroem/random-forest experiments, score inspection scripts,
  UMAP utility and existing figure files are retained. Unused imports are removed;
  command guards let CLI help work without loading a dataset or model.
- Source CSVs use `ressource/raw_data/`, matching current main. Notebook paths
  and the result generator follow that layout. Historical compact feature,
  metric and prediction tables are retained for reproducible evidence.
- Upstream deletion of serialized estimators and duplicate historical PNGs is
  retained. Their original contents remain in Git history. No raw audio or large
  newly generated arrays are added; `rapport/` stays untracked and untouched.

## Targeted correctness fixes

1. Unannotated windows of validation/test recordings are filtered out of the
   pseudo-label pool. Grouping only labeled windows had left this overlap possible.
2. Annotation timestamps support hours, duplicate annotations combine by OR, and
   target columns and extracted row order remain aligned with the submission.
3. Audio class labels come from the recording's parent directory for both POSIX
   and Windows/absolute CSV paths.
4. PR comparison shading interpolates between curves with different numbers of
   thresholds, preventing incompatible-array plotting errors. Score calculation
   and displayed PR curves are unchanged.
5. Plot output directories resolve from the project root. `joblib` is explicit;
   existing seaborn/UMAP use is declared in optional `plots` dependencies and
   imported only when needed.

## Scientific limits

The ten score rows in `results/` remain historical single-label results with their
existing caveats. Current multi-label figure images do not provide a substitute
for full-precision metrics/predictions, so no new score is invented. Large-data
multi-label training and external Perch inference are not rerun during conflict
resolution. Group-aware inner CV, final parameter selection and test-based
pseudo-label threshold diagnostics still need a dedicated experiment review.

Validation covers the historical report and synthetic current loaders, recording
group independence, held-out pseudo-pool exclusion, annotation alignment, PR
plotting and all relevant CLI help paths. See `VALIDATION.md` for execution results.
