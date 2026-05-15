import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from scripts.utils import load_data2, save_results


def main():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data2()

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

    save_results("Random Forest","ressource/results/ml_classic/manual_embedding/random_forest",y_test, test_predictions, test_accuracy, test_f1)


if __name__ == "__main__":
    main()
