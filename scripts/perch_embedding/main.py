import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from scripts.utils import load_data, save_results
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier



def main():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    #print("Logistic Regression algorithm: ")
    logit = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,     
            C=1.0,  
            class_weight="balanced",
            random_state=15,
            n_jobs=-1
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
        SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", probability=True),
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
    save_results("Soft voting model (rf + svm + logit)","ressource/results/ml_classic/perch_embedding/soft_voting",y_test, test_predictions, test_accuracy, test_f1)



    
if __name__ == "__main__":
    main()
