import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scripts.utils import load_data2, save_results

def main():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data2()

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

    save_results("SVM RBF","ressource/results/ml_classic/manual_embedding/svm_rbf2",y_test, test_predictions, test_accuracy, test_f1)


if __name__ == "__main__":
    main()
