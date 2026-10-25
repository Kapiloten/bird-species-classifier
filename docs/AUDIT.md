# Repository audit

This records the initial audit of the historical single-label checkout.
For subsequent integration with the current multi-label `main` workflow, moved
metadata and upstream artifact cleanup, see [INTEGRATION.md](INTEGRATION.md).

Inspection date: 2026-10-25. Base commit: `42143d3b`.
Working branch: `codex/portfolio-audit`.

## Initial diagnosis

### Strengths

- Two implemented audio representations: interpretable acoustic/MFCC features and frozen Perch embeddings, followed by classical supervised classifiers.
- A committed recording-level split, compact feature tables, predictions for the manual models, confusion matrices and machine-readable metrics.
- Imputation and scaling are fitted inside scikit-learn pipelines on training data only.
- Seeds already exist for the manual split (42), manual estimators (42), Perch split (12), and some Perch estimators (15).
- Useful exploratory notebooks and a local technical report already exist.

### Findings, ordered by impact

| Finding | Evidence at inspection | Action |
| --- | --- | --- |
| Perch augmentation before splitting leaks related observations | `load_data()` calls `increase_rare_birds()` before both splits; its global standard deviation also uses all observations | Split original embeddings first; omit classes with fewer than five original observations from this closed-set evaluation, then augment training only with a local seeded generator. Historical scores remain flagged. |
| Manual split contains identical feature vectors across partitions | 11 duplicate pairs across all 51 numeric features; seven pairs cross train/test | Export a reproducible duplicate report and warn when loading. Preserve the historical split and scores; a new grouped split and retraining are required for a clean benchmark. |
| Recording identity is not independence | No duplicate file paths, but 193 authors occur in both manual train and test metadata | Document source/author/site dependence; do not claim generalization to new recordists or locations. |
| Incomparable experimental scopes | Manual: six bird classes, 596 test records. Perch: 206 classes, 7,118 test observations, potentially including synthetic observations | Separate result tables and figures; do not interpret their difference as a controlled representation gain. |
| README empty, project metadata placeholder | Empty `README.md`, description `Add your description here`, generic `main.py` | Write an English technical README, add honest portfolio metadata, replace the placeholder entry point with useful command guidance. |
| Incomplete dependencies | Perch imports TensorFlow, TF Hub, Torch and tqdm without declarations; window scripts directly import SciPy and SoundFile; obsolete `pathlib` backport declared | Declare optional Perch/notebook groups, direct dependencies and development tools; update the uv lock without installing the heavy Perch stack. |
| Feature join can silently drop or multiply rows | Unvalidated inner merge on six metadata columns | Validate schemas, unique recording paths, one-to-one matching and split consistency before training. |
| Notebook path stale | `features_calssique.ipynb` reads `features/features.csv` after the directory moved | Correct this reference while preserving notebook cells and historical outputs. |
| Weak result provenance | Perch embeddings and file identities absent; voting variant configurations not saved; manual JSON refers to an older feature directory | Reconstruct metrics from integer confusion counts, validate against archived metrics, record input SHA-256 hashes; save predictions and configuration for future runs. |
| No tests or CI | No test directory or workflow | Add synthetic-data regression tests and dataset-free CI. |
| Publication hygiene incomplete | 16 GB local archive ignored, but other archives/checkpoints/caches need coverage; existing RF joblib is ~29 MB | Extend ignore rules; retain already tracked artifacts and Git history. No data or trained artifact is deleted. |

## Evidence and interpretation

`taxonomy.csv` has 234 taxa: 162 Aves, 35 Amphibia, 28 Insecta,
8 Mammalia and 1 Reptilia. `train.csv` has 35,549 recordings across 206 labels;
14 labels have fewer than five recordings. The local `train_audio/` contains
35,549 OGG files. These are BirdCLEF+ 2026 assets; no other training dataset is
evidenced in the inspected scripts or notebooks.

The manual manifest and both feature CSVs match exactly on their six metadata
columns (2,982 rows; 1,938 train / 448 validation / 596 test; 51 numeric features).
There are no missing values or duplicate paths in these tables. Identical feature
pairs have different compressed-file hashes, so exact feature equality alone does
not prove byte-identical audio. Decoded-waveform or perceptual checks and grouping
by source recording are still needed. Near-duplicates have not been exhaustively
searched. The feature report is reproducible without distributing raw audio.

The manual SVM confusion matrix gives accuracy 0.7483221476510067 and macro F1
0.7500740592378207. The archived Perch soft-voting matrix gives accuracy
0.8617589210452374 and macro F1 0.8462004135123383. These are **historical internal
results**, not competition scores or corrected independent-test estimates.
Precision and recall can be recovered exactly from the saved integer matrices;
the rounded text classification reports are not used as the numeric source.

`analyse_dataset.ipynb` selects the six most represented bird classes.
`features_calssique.ipynb` explores manual features; neither notebook contains a
trained-model evaluation beyond the separately archived artifacts. The pre-existing
`rapport/` is retained as historical material; its claims of a Perch gain must be
read with the scope and leakage caveats above.

## Change classification

- **Documentation:** English README, pipeline diagram, result interpretation,
  installation/reproduction commands, GitHub description/topics/skills proposal,
  audit and result provenance.
- **Bug fixes:** train-only Perch augmentation and deterministic randomness;
  validated feature joins; stale notebook path; fixed Perch input rate and stable
  file ordering; complete dependency declarations.
- **Small supporting additions:** reproducible result/duplicate reports, future
  prediction/configuration exports, unit tests and a dataset-free GitHub Action.

No classifier family, acoustic descriptor formula or manual model hyperparameter
is changed. The Perch evaluation protocol intentionally changes to correct
leakage; new scores will not reproduce the old table. No Perch retraining is
claimed by this audit. Validation performed after implementation is recorded in
`docs/VALIDATION.md`.

## Preservation and remaining work

The checkout already had uncommitted changes in `.gitignore`, feature extraction,
training scripts, Perch scripts and `scripts/utils.py`, plus untracked path/window
utilities and `rapport/`. Work continues on top of those changes; they are not
reverted or attributed to this audit. No commit, push, repository setting update,
dataset removal or history rewrite is part of this work.

Before making performance claims: create a versioned group-aware split, deduplicate
recordings, retain original recording identities in Perch exports, audit overlap
with pretrained model data, run both representations on the same bird-only scope,
and evaluate once on an untouched test set. Full soundscape multi-label evaluation,
cross-validation/uncertainty estimates and voting-weight selection on validation
data remain future experiments.
