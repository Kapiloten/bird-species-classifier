import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from project_paths import PROCESSED_DATA_DIR, project_path
from utils import load_data, save_results


DEFAULT_OUTPUT_DIR = project_path("results/runs/perch_soft_voting_corrected")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a soft-voting classifier on Perch embeddings.")
    parser.add_argument("--embeddings-dir", default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--seed", type=int, default=12, help="Seed for splitting and train-only augmentation.")
    parser.add_argument("--min-originals", type=int, default=5, help="Minimum original recordings per class.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where metrics and reports are saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(
        args.embeddings_dir, random_state=args.seed, min_originals=args.min_originals
    )

    #print("Logistic Regression algorithm: ")
    logit = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,     
            C=1.0,  
            class_weight="balanced",
            random_state=15,
        )
    )
    '''logit.fit(x_train, y_train)

    val_predictions = logit.predict(x_val)
    test_predictions = logit.predict(x_test)

    val_accuracy = accuracy_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions, average="macro")
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions, average="macro")

    print(f"Validation accuracy : {val_accuracy:.4f}")
    print(f"Validation F1 macro : {val_f1:.4f}")
    print(f"Test accuracy       : {test_accuracy:.4f}")
    print(f"Test F1 macro       : {test_f1:.4f}")

    save_results("Logistic Regression","ressource/results/ml_classic/perch_embedding/log_reg",y_test, test_predictions, test_accuracy, test_f1)'''
    

    #bagging + random embed
    #print("Random Forest algorithm: ")
    rf = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=15,
            n_jobs=-1,
        ),
    )

    '''rf.fit(x_train, y_train)
    
    val_predictions = rf.predict(x_val)
    test_predictions = rf.predict(x_test)

    val_accuracy = accuracy_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions, average="macro")
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions, average="macro")

    print(f"Validation accuracy : {val_accuracy:.4f}")
    print(f"Validation F1 macro : {val_f1:.4f}")
    print(f"Test accuracy       : {test_accuracy:.4f}")
    print(f"Test F1 macro       : {test_f1:.4f}")

    save_results("Random Forest","ressource/results/ml_classic/perch_embedding/random_forest",y_test, test_predictions, test_accuracy, test_f1)'''


    #svm with gaussian kernel
    #print("SVM RBF algorithm: ")
    svm = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", probability=True, random_state=15),
    )

    '''svm.fit(x_train, y_train)

    val_predictions = svm.predict(x_val)
    test_predictions = svm.predict(x_test)

    val_accuracy = accuracy_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions, average="macro")
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions, average="macro")

    print(f"Validation accuracy : {val_accuracy:.4f}")
    print(f"Validation F1 macro : {val_f1:.4f}")
    print(f"Test accuracy       : {test_accuracy:.4f}")
    print(f"Test F1 macro       : {test_f1:.4f}")

    save_results("SVM RBF","ressource/results/ml_classic/perch_embedding/svm",y_test, test_predictions, test_accuracy, test_f1)'''

    '''#boosting
    print("Light GBM: ")
    gbm = make_pipeline(
        SimpleImputer(strategy="median"),
        LGBMClassifier(
            n_estimators=150,
            learning_rate=0.3, 
            class_weight="balanced",
            reg_alpha=0.1,        
            reg_lambda=0.1,
            random_state=15,
            n_jobs=-1,
            verbose=-1
        )
    )
    gbm.fit(x_train, y_train)

    val_predictions = gbm.predict(x_val)
    test_predictions = gbm.predict(x_test)

    val_accuracy = accuracy_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions, average="macro")
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions, average="macro")

    print(f"Validation accuracy : {val_accuracy:.4f}")
    print(f"Validation F1 macro : {val_f1:.4f}")
    print(f"Test accuracy       : {test_accuracy:.4f}")
    print(f"Test F1 macro       : {test_f1:.4f}")

    save_results("Light GBM","ressource/results/ml_classic/perch_embedding/light_gbm",y_test, test_predictions, test_accuracy, test_f1)'''

    print("Soft voting using rf, svm and logit: ")
    soft_voting = VotingClassifier(
        estimators=[
            ('rf', rf), 
            ('svm', svm), 
            ('logit', logit),
        ],
        voting='soft',
        weights=[1, 2, 3]
    )

    soft_voting.fit(x_train, y_train)

    val_predictions = soft_voting.predict(x_val)
    test_predictions = soft_voting.predict(x_test)

    val_accuracy = accuracy_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions, average="macro")
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions, average="macro")

    print(f"Validation accuracy : {val_accuracy:.4f}")
    print(f"Validation F1 macro : {val_f1:.4f}")
    print(f"Test accuracy       : {test_accuracy:.4f}")
    print(f"Test F1 macro       : {test_f1:.4f}")
    save_results(
        "Soft voting model (rf + svm + logit)",
        project_path(args.output_dir),
        y_test,
        test_predictions,
        test_accuracy,
        test_f1,
        configuration={
            "protocol": "original-recording-split-then-training-augmentation-v1",
            "split_seed": args.seed, "min_originals": args.min_originals,
            "estimator_seed": 15, "weights": [1, 2, 3],
            "n_train_after_augmentation": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
            "n_classes": len(set(y_train)),
            "validation_accuracy": val_accuracy, "validation_f1_macro": val_f1,
            "input_files": [str(project_path(args.embeddings_dir) / name)
                            for name in ("X_embeddings_perch.npy", "y_labels.npy")],
        },
    )



    
if __name__ == "__main__":
    main()
