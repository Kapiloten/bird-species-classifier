# Historical result summary

Generated with `uv run python scripts/summarize_results.py`. No model training was performed.

Manual scores contain seven exact feature-duplicate pairs across train/test. Perch scores use pre-split synthetic augmentation and mixed taxa. These scopes are not directly comparable.

## manual_embedding

| run | n_classes | n_test | accuracy | precision_macro | recall_macro | f1_macro |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 6 | 596 | 0.7248 | 0.7267 | 0.7247 | 0.7253 |
| random_forest | 6 | 596 | 0.6661 | 0.6701 | 0.6659 | 0.6671 |
| svm_rbf | 6 | 596 | 0.7483 | 0.7547 | 0.7482 | 0.7501 |

## perch_embedding

| run | n_classes | n_test | accuracy | precision_macro | recall_macro | f1_macro |
| --- | --- | --- | --- | --- | --- | --- |
| light_gbm | 206 | 7118 | 0.2727 | 0.1640 | 0.2881 | 0.1969 |
| log_reg | 206 | 7118 | 0.8554 | 0.8547 | 0.8275 | 0.8347 |
| random_forest | 206 | 7118 | 0.7435 | 0.7684 | 0.6837 | 0.7025 |
| soft_voting | 206 | 7118 | 0.8618 | 0.8648 | 0.8421 | 0.8462 |
| soft_voting2 | 206 | 7118 | 0.8597 | 0.8524 | 0.8360 | 0.8383 |
| soft_voting3 | 206 | 7118 | 0.8554 | 0.8498 | 0.8285 | 0.8332 |
| svm | 206 | 7118 | 0.8462 | 0.8455 | 0.7733 | 0.8009 |

Exact values are in `summary.csv`; inputs and SHA-256 hashes are in `provenance.json`.

Voting variants 2/3 and LightGBM are archived experiments; their complete runnable configurations are not preserved in the current executable code.
