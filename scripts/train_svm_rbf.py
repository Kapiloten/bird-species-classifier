from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


META_COLUMNS = ["filepath", "filename", "label", "common_name", "scientific_name", "split"]
RESULTS_DIR = Path("results/ml_classique/svm_rbf")


def load_data():
    audio_features = pd.read_csv("features/features.csv")
    mfcc_features = pd.read_csv("features/mfcc_features.csv")
    data = audio_features.merge(mfcc_features, on=META_COLUMNS)

    feature_columns = [column for column in data.columns if column not in META_COLUMNS]

    train = data[data["split"] == "train"]
    val = data[data["split"] == "val"]
    test = data[data["split"] == "test"]

    return (
        train[feature_columns],
        train["label"],
        val[feature_columns],
        val["label"],
        test[feature_columns],
        test["label"],
    )


def save_results(y_test, predictions, test_accuracy, test_f1):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(
        [{"model": "svm_rbf", "test_accuracy": test_accuracy, "test_f1_macro": test_f1}]
    )
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)

    labels = sorted(y_test.unique())
    report = classification_report(y_test, predictions, labels=labels)
    (RESULTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(RESULTS_DIR / "confusion_matrix.csv")


def main():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced"),
    )

    model.fit(x_train, y_train)

    val_predictions = model.predict(x_val)
    test_predictions = model.predict(x_test)

    val_accuracy = accuracy_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions, average="macro")
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions, average="macro")

    print("SVM RBF")
    print(f"Validation accuracy : {val_accuracy:.4f}")
    print(f"Validation F1 macro : {val_f1:.4f}")
    print(f"Test accuracy       : {test_accuracy:.4f}")
    print(f"Test F1 macro       : {test_f1:.4f}")

    save_results(y_test, test_predictions, test_accuracy, test_f1)


if __name__ == "__main__":
    main()
