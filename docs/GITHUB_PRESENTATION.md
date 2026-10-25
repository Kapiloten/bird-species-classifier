# GitHub portfolio presentation

## Suggested About description

BirdCLEF audio classification with acoustic features, Perch embeddings and classical ML; reproducible evaluation, error analysis and tests.

## Suggested topics

`bioacoustics`, `birdclef`, `bird-species-classification`, `audio-classification`,
`signal-processing`, `mfcc`, `transfer-learning`, `perch`, `scikit-learn`,
`machine-learning`, `python`, `reproducible-research`

These are proposals; no GitHub settings were changed.

## Skills demonstrated by the repository

| Skill | Concrete evidence |
| --- | --- |
| Audio signal processing | Mono/resampling pipeline, RMS/ZCR/spectral descriptors, MFCC statistics and five-second window utilities |
| Representation learning | Inference with a frozen pretrained Perch model and mean aggregation; no claim of training the foundation model |
| Supervised machine learning | Logistic regression, random forest, RBF SVM, class balancing and soft probability voting |
| Experimental reasoning | Separate validation/test partitions, macro metrics, class-level confusion analysis and explicit limitations |
| Research software engineering | Root-relative paths, configurable CLIs, uv lock, prediction/configuration exports, synthetic pytest cases and CI |
| Data quality analysis | Metadata/schema validation, exact feature-duplicate diagnostics and augmentation-leakage correction |

## Suggested portfolio wording

> Built an audio classification pipeline on BirdCLEF+ 2026 data, comparing
> interpretable acoustic/MFCC features with pretrained Perch embeddings and
> classical classifiers. Reconstructed archived evaluations, analyzed class-level
> errors, and improved experiment traceability, tests and data-split checks.

If citing the manual SVM's **74.83% historical accuracy**, mention the six-class
scope and duplicate-feature split limitation. If citing Perch's **86.18% historical
accuracy**, mention the 206 mixed-taxon scope and pre-split augmentation limitation.
Neither is a competition result or an independent deployment benchmark.

## Repository organization

Keep the existing paths and useful notebook filenames to preserve references and
history. Use `docs/` for methodology/audit, `results/` for small generated portfolio
artifacts, `results/runs/` for ignored local experiments, and `tests/` for synthetic
checks. Keep the pre-existing `rapport/` as historical analysis, with the audit's
evaluation caveats. Moving `ressource/` or renaming notebooks can be a separate
small migration if desired; it is unnecessary for this audit.

Upstream `main` removed serialized models; that cleanup is retained. Older trained
artifacts remain recoverable from Git history, and compact metric/prediction tables
are retained for report reproduction. Future model and embedding exports are
ignored. Do not redistribute raw audio, model weights or
third-party metadata without checking their applicable terms. Choose a source-code
license deliberately before advertising the project as open source; this audit
does not assume ownership or add a license on the author's behalf.
