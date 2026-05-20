import sys
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.multiclass import OneVsRestClassifier
sys.path.append(str(Path(__file__).parent.parent.parent))
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from scripts.utils import load_data, plot_learning_curve, plot_model_comparison, plot_pr_curve, plot_top_flop_f1, plot_worst_confusion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier



def main():
    X_train, Y_train, X_val, Y_val, X_test, Y_test, X_nLabeled, Y_nLabeled, Z_nLabeled = load_data()
    grid_logreg = {
        'estimator__C': [0.001, 0.01, 0.1, 1.0, 10.0]
    }
    model_logreg = GridSearchCV(
        OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=1000)),
        param_grid=grid_logreg, 
        cv=3, 
        scoring='f1_macro'
    )
    model_logreg.fit(X_train, Y_train)
    print(f"\n Best hyperparam : {model_logreg.best_params_}")


    grid_svm_rbf = {
        'estimator__C': [1.0, 10.0],
        'estimator__gamma': ['scale', 'auto'] 
    }
    
    model_svm = GridSearchCV(
        OneVsRestClassifier(SVC(kernel='rbf', class_weight='balanced', max_iter=2000)),
        param_grid=grid_svm_rbf, 
        cv=3, 
        scoring='f1_macro',
        n_jobs=-1  
    )
    model_svm.fit(X_train, Y_train)
    print(model_svm.best_params_)

    Y_val_pred_logreg = (model_logreg.best_estimator_.predict_proba(X_val) > 0.5).astype(int)
    score_logreg = f1_score(Y_val, Y_val_pred_logreg, average='macro')

    Y_val_pred_svm = (model_svm.best_estimator_.decision_function(X_val) > 0).astype(int)
    score_svm = f1_score(Y_val, Y_val_pred_svm, average='macro')

    print(f"Final score LogReg : {score_logreg:.4f}")
    print(f"Final score SVM: {score_svm:.4f}")
    plot_model_comparison(score_logreg, score_svm)


    '''model = "LogReg_Perch"


    print("\n--- Final training by merging validation set with training set ---")
    X_train_merge = np.concatenate((X_train, X_val))
    Y_train_merge = np.concatenate((Y_train, Y_val))
    
    base_model = OneVsRestClassifier(LogisticRegression(C=best_c, class_weight='balanced', max_iter=1000, n_jobs=-1))
    
    start_time = time.time()
    base_model.fit(X_train_merge, Y_train_merge)
    print(f"Model trained in {time.time() - start_time:.1f} secondes.")

    print("Evaluation on the test set...")
    Y_test_proba_base = base_model.predict_proba(X_test)
    Y_test_pred_base = (Y_test_proba_base > 0.5).astype(int)

    print("Plotting learning curve...")
    train_sizes, train_scores, val_scores = learning_curve(
        base_model, X_train, Y_train, cv=3, scoring='f1_macro', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    plot_learning_curve(train_sizes, train_scores, val_scores, model)
    
    # 2. Évaluation classique
    plot_pr_curve(Y_test, Y_test_proba_base, model)
    plot_top_flop_f1(Y_test, Y_test_pred_base, model)
    plot_worst_confusion(Y_test, Y_test_pred_base, model)'''

    
if __name__ == "__main__":
    main()
