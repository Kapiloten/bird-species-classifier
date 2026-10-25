# Reproducing reports and experiments

Use Python 3.11 and the committed `uv.lock`. Commands below run from the repository
root and use no shell-specific line continuations.

## 1. Rebuild archived tables and figures

```bash
uv sync --locked --group dev
uv run --locked python scripts/summarize_results.py
```

This validates ten archived runs, reads no audio and fits no models. Outputs and
input fingerprints are described in [results/README.md](../results/README.md).
To write elsewhere: `--output-dir results/runs/report-check`.

## 2. Re-fit manual classifiers without raw audio

```bash
uv run --locked python scripts/manual_embedding/train_logistic_regression.py --output-dir results/runs/check/logistic_regression
uv run --locked python scripts/manual_embedding/train_random_forest.py --output-dir results/runs/check/random_forest
uv run --locked python scripts/manual_embedding/train_svm_rbf.py --output-dir results/runs/check/svm_rbf
```

Inputs are the two committed `ressource/features/` CSVs; their metadata exactly
matches the split manifest. Preprocessing is fit on training only. Future result
exports contain predictions and `run.json` (classifier parameters, sizes,
validation metrics, Python/package versions and SHA-256 of feature inputs).
Predictions are in manifest test-row order, so they can be joined positionally to
the versioned test manifest. These runs retain the historical duplicate-feature
limitation and are not a new clean benchmark.

## 3. Re-extract whole-recording features from the original audio

Obtain BirdCLEF+ 2026 through Kaggle with the required access and keep data local:

```text
train_audio/
  banana/*.ogg
  fepowl/*.ogg
  ...
train_soundscapes/*.ogg          # Only needed for the exploratory window scripts
```

The committed split SHA-256 at inspection is
`06273cf1ee113bab1e782acffd8de5a6c6ced1fbdd75f06a01de3b007992866b`.
Using it avoids changing file assignments. Extract to a separate directory:

```bash
uv run --locked python scripts/manual_embedding/extract_features.py --output results/runs/reextracted/features.csv
uv run --locked python scripts/manual_embedding/extract_mfcc_features.py --output results/runs/reextracted/mfcc_features.csv
uv run --locked python scripts/manual_embedding/train_svm_rbf.py --features-dir results/runs/reextracted --output-dir results/runs/reextracted/svm
```

Both extraction scripts accept `--split`, `--output` and `--limit`. A limited export
is an extraction smoke check, not a trainable complete dataset. Audio is loaded
with librosa at 32 kHz; the full-recording MFCC count is 20. These formulas and
defaults are unchanged by the audit.

To regenerate the *historical recording-level* split from the exact same audio
inventory, use the existing per-class shuffle/rounding scheme:

```bash
uv run --locked python scripts/manual_embedding/create_audio_split.py --seed 42 --output results/runs/audio_split_65_15_20.csv
```

This does not remove duplicate-source leakage. Design a new group-aware manifest
before making independent-test claims. Keep any revised split and its provenance
under a distinct filename; do not replace historical results with new-protocol
scores without identifying the protocol change.

## 4. Perch extraction and corrected evaluation

The optional stack is resolved in the lock but was not installed or exercised
end to end during the audit. Use Python 3.11 (the optional stack is restricted to
Python <3.14) and consult [TensorFlow platform instructions](https://www.tensorflow.org/install/pip)
if needed. GPU inference is not required by the script.

```bash
uv sync --locked --extra perch
uv run --locked --extra perch python scripts/perch_embedding/embedding_perch.py --audio-dir train_audio --output-dir ressource/processed_data
uv run --locked python scripts/perch_embedding/main.py --embeddings-dir ressource/processed_data --seed 12 --min-originals 5 --output-dir results/runs/perch_corrected
```

The extractor downloads the versioned model referenced by `MODEL_URL`. It saves
aligned `X_embeddings_perch.npy`, `y_labels.npy`, `filepaths.npy` (sorted relative
recording paths), and `extraction.json` with the model URL, frame/aggregation
parameters and failures. It averages embeddings of padded five-second frames.
Check failures and observed class counts before interpreting any downstream run.

The classifier uses embeddings only and does not require TensorFlow/Torch once
the arrays exist. The corrected protocol:

1. Excludes labels with fewer than five original observations, with a warning.
2. Splits original rows with stratification, test fraction 0.20, then validation
   fraction 0.1875 of the remaining rows (approximately 65/15/20 overall), seed 12.
3. Ensures all included labels occur in each partition; otherwise fails with a
   request to raise `--min-originals`.
4. Augments only training classes below five rows using Gaussian noise scaled by
   training-array standard deviation, local RNG seed 12.
5. Fits the existing 1:2:3 RF/SVM/logistic soft vote; estimator/probability seed 15.

The old loader augmented before splitting. Its unseeded synthetic rows and missing
embedding inventory prevent exact end-to-end reproduction of the historical
Perch table. New corrected scores must be reported separately. Sorted input order
and seeds aid repeatability, but do not guarantee bitwise equivalence across
platforms or eliminate duplicate/source/pretraining overlap.

## 5. Exploratory window exports

```bash
uv run --locked python scripts/manual_embedding/extract_all_5s_features.py --help
uv run --locked python scripts/manual_embedding/extract_soundscape_window_features.py --help
```

These pre-existing utilities are preserved. The all-window script uses its own
FFT/log-mel/DCT implementation; its values are not interchangeable with the
whole-recording librosa baseline. The soundscape utility exports multi-hot targets
from the supplied annotations. Neither output feeds the reported classifier
scores. If using windows for experiments, split/group by original recording before
windowing; adjacent windows must not cross training and test.

## 6. Local checks and CI

```bash
uv run --locked ruff check scripts tests main.py
uv run --locked pytest
```

Tests use synthetic tables, embeddings and generated audio in temporary folders.
They do not need Kaggle access, raw recordings, pretrained weights or serialized
models. The GitHub Action additionally validates archived metric/prediction
consistency and generates a temporary report. It installs only core + development
dependencies, with a read-only repository token.

For a Windows sandbox that cannot create its default system temporary directory,
pass a new workspace-local directory to pytest, for example
`--basetemp results/runs/pytest-local`. Pytest owns and may clear its base directory;
use a dedicated location containing no project data.
