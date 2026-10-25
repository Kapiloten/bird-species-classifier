# Validation performed

## Integration with main (Pull Request #1)

After merging the multi-label workflow from `main` at `0c1a794d`:

- **46 tests passed** in 50.69 seconds, with the same three upstream audioread
  deprecation warnings. This includes historical tests and CLI help plus current
  CSV/Perch fixture loading, annotation alignment, recording groups, held-out
  pseudo-label pool exclusion and different-length PR plotting.
- Ruff passed for all scripts, tests and the root command guide. `uv lock --check
  --offline` validated 185 resolved packages, including optional plotting extras.
- The report generator validated all ten historical runs after the metadata move
  to `ressource/raw_data/`; all numeric scores remain unchanged. SHA-256 provenance
  paths were regenerated for this layout.
- The 17 command-help cases run from outside the repository, including current
  multi-label entry points, without datasets, serialized models or Perch weights.
- No large multi-label experiment, optional model inference or new score claim
  was added. Existing model/search settings were preserved; protocol/plot fixes
  are described in [INTEGRATION.md](INTEGRATION.md).

The initial validation record below remains historical: its no-commit/no-push
statement describes the first audit delivery, before the user's commit and PR.

## Initial audit validation

Date: 2026-10-25. Local environment: Windows, CPython 3.11.14, uv 0.9.27.
Core packages: NumPy 2.4.4, pandas 3.0.3, scikit-learn 1.8.0,
librosa 0.11.0 and Matplotlib 3.10.9. Development tools resolved in the lock:
pytest 9.1.1 and Ruff 0.17.2.

## Completed checks

- `uv lock --check --offline`: lock matches `pyproject.toml`; 182 packages resolved
  including optional extras. Only development additions were installed locally;
  the existing notebook environment was preserved with `uv sync --inexact`.
- `ruff check scripts tests main.py`: passed. The rules cover import/name errors
  and basic syntax/style errors without reformatting historical research code.
- `pytest -q`: **35 tests passed**. A fresh dedicated workspace-local
  `--basetemp results/runs/pytest-<unique-id>` was used because the Windows sandbox
  could not create pytest's default system temp directory. Tests cover deterministic
  and disjoint splits, train-only augmentation, rare-class exclusion, schema/join
  failures, duplicate warnings, metrics against sklearn, result exports, generated
  audio features and CLI help from outside the repository. Three upstream
  `audioread` deprecation warnings were emitted on Python 3.11.
- `python scripts/summarize_results.py`: validated **10 archived runs** against
  their confusion matrices, both copies of manual confusion matrices, and all
  manual prediction files. Generated exact aggregate/per-class tables, duplicate
  diagnostics, source hashes and four figures without audio or model weights.
- All four generated PNGs were visually inspected for readable labels, unclipped
  text, consistent axes and explicit experimental-scope caveats.
- Relative Markdown links in the README and documentation resolve locally.
- `git diff --check`: passed. Ignore rules were checked for the archive, audio,
  embeddings, run outputs, uv cache, Kaggle credentials and environment files.

## Actual manual re-fits

Each current training CLI was executed against the committed 51-feature tables,
with outputs under ignored `results/runs/manual/`. **Every test prediction and the
full-precision accuracy/macro F1 match the corresponding historical artifact.**
No historical result files or serialized models were overwritten.

| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.720982 | 0.721487 | 0.724832 | 0.725346 |
| Random forest | 0.671875 | 0.671518 | 0.666107 | 0.667089 |
| RBF SVM | 0.729911 | 0.729396 | 0.748322 | 0.750074 |

These re-fits verify behavior preservation, not independence of the historical
split. The loader correctly emits the cross-partition duplicate-feature warning.

## Not performed

- No Perch dependency installation, weight download, real embedding inference or
  corrected Perch model training. The optional dependency set resolves, CLI help
  works without it, and splitting/augmentation is tested with synthetic arrays.
- No full raw-audio feature re-extraction or rerun of exploratory window exports.
- No new duplicate-aware/group-aware experiment, full audio fingerprint search,
  hyperparameter search, uncertainty estimation or competition submission.
- The GitHub Action was added and its check commands were exercised locally;
  it has not been executed on GitHub's Linux runner in this task.
- No commit, push, GitHub metadata change, license selection or history rewrite.

The initial local changes and historical artifacts were preserved. The changes
are available for review on the local branch `codex/portfolio-audit`.
