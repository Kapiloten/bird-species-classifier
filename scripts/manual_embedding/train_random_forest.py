import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from project_paths import FEATURES_DIR, project_path
from utils import load_data2, save_results


DEFAULT_OUTPUT_DIR = project_path("results/runs/manual/random_forest")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a random forest on manual features.")
    parser.add_argument("--features-dir", default=str(FEATURES_DIR), help="Directory containing both feature CSVs.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where metrics and reports are saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    x_train, y_train, x_val, y_val, x_test, y_test = load_data2(args.features_dir)

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
    )

    model.fit(x_train, y_train)

    val_predictions = model.predict(x_val)
    test_predictions = model.predict(x_test)

    val_accuracy = accuracy_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions, average="macro")
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions, average="macro")

    print("Random Forest")
    print(f"Validation accuracy : {val_accuracy:.4f}")
    print(f"Validation F1 macro : {val_f1:.4f}")
    print(f"Test accuracy       : {test_accuracy:.4f}")
    print(f"Test F1 macro       : {test_f1:.4f}")

    save_results(
        "Random Forest",
        project_path(args.output_dir),
        y_test,
        test_predictions,
        test_accuracy,
        test_f1,
        configuration={
            "protocol": "manual-recording-split-65-15-20", "classifier": model[-1].get_params(),
            "n_train": len(y_train), "n_val": len(y_val), "n_features": x_train.shape[1],
            "validation_accuracy": val_accuracy, "validation_f1_macro": val_f1,
            "input_files": [str(project_path(args.features_dir) / name)
                            for name in ("features.csv", "mfcc_features.csv")],
        },
    )


if __name__ == "__main__":
    main()
