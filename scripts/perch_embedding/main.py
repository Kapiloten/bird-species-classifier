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
from scripts.utils import load_data, plot_learning_curve, plot_model_comparison, plot_pr_comparison, plot_pr_curve, plot_pseudo_label_tradeoff, plot_top_flop_f1, plot_umap_projection, plot_worst_confusion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import Nystroem
import joblib
from sklearn.neighbors import KNeighborsClassifier




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


    '''grid_svm_rbf = {
        'estimator__svc__C': [0.001,1.0, 10.0],
        'estimator__svc__gamma': ['scale', 'auto'] 
    }
    
    svm_pipeline = make_pipeline(
        StandardScaler(), 
        SVC(kernel='rbf', class_weight='balanced', max_iter=2000)
    )

    model_svm = GridSearchCV(
        OneVsRestClassifier(svm_pipeline),
        param_grid=grid_svm_rbf, 
        cv=3, 
        scoring='f1_macro',
        n_jobs=-1  
    )
    
    model_svm.fit(X_train, Y_train)
    print(model_svm.best_params_)'''

    Y_val_pred_logreg = (model_logreg.best_estimator_.predict_proba(X_val) > 0.5).astype(int)
    score_logreg = f1_score(Y_val, Y_val_pred_logreg, average='macro')

    '''Y_val_pred_svm = (model_svm.best_estimator_.decision_function(X_val) > 0).astype(int)
    score_svm = f1_score(Y_val, Y_val_pred_svm, average='macro')'''

    print(f"Final score LogReg : {score_logreg:.4f}")
    #print(f"Final score SVM: {score_svm:.4f}")
    #plot_model_comparison(score_logreg, score_svm)



    print("\n--- Entraînement final sur Train + Val ---")
    X_train_merge = np.concatenate((X_train, X_val))
    Y_train_merge = np.concatenate((Y_train, Y_val))
    
    
    print("Entraînement LogReg Final...")
    final_logreg = OneVsRestClassifier(LogisticRegression(C=0.001, class_weight='balanced', max_iter=1000), n_jobs=-1)
    final_logreg.fit(X_train_merge, Y_train_merge)
    
    joblib.dump(final_logreg, 'ressource/results/final_logreg_model.pkl')
    

    print("\n--- Prédictions sur le set de Test ---")
    
    Y_test_proba_logreg = final_logreg.predict_proba(X_test)
    Y_test_pred_logreg = (Y_test_proba_logreg > 0.5).astype(int)
    

    score_logreg = f1_score(Y_test, Y_test_pred_logreg, average='macro')
    print(f"Final score LogReg : {score_logreg:.4f}")



    print("\n--- Génération des graphiques LogReg ---")
    nom_logreg = "LogReg_Perch"
    
    
    plot_pr_curve(Y_test, Y_test_proba_logreg, nom_logreg)
    plot_top_flop_f1(Y_test, Y_test_pred_logreg, nom_logreg)
    plot_worst_confusion(Y_test, Y_test_pred_logreg, nom_logreg)


    
    print("\n--- Début du Pseudo-Labeling ---")

    print("Prédiction sur les données non étiquetées...")
    proba_unlabeled = final_logreg.predict_proba(X_nLabeled)

    seuil_confiance = 0.9
    
    Y_pseudo = (proba_unlabeled >= seuil_confiance).astype(int)


    mask_confident = Y_pseudo.sum(axis=1) > 0
    X_pseudo_confident = X_nLabeled[mask_confident]
    Y_pseudo_confident = Y_pseudo[mask_confident]

    print(f"Nombre de nouveaux extraits d'oiseaux ajoutés : {len(X_pseudo_confident)}")

    X_train_augmented = np.concatenate((X_train_merge, X_pseudo_confident))
    Y_train_augmented = np.concatenate((Y_train_merge, Y_pseudo_confident))

    print("\n--- Entraînement du modèle augmenté ---")
    logreg_pseudo = OneVsRestClassifier(LogisticRegression(C=0.001, class_weight='balanced', max_iter=1000), n_jobs=4)
    
    start_time = time.time()
    logreg_pseudo.fit(X_train_augmented, Y_train_augmented)

    print(f"Modèle augmenté entraîné en {time.time() - start_time:.1f} secondes.")

    joblib.dump(logreg_pseudo, 'ressource/results/pseudo_logreg_model.pkl')

    print("\n--- Évaluation finale sur le set de Test ---")
    
    Y_test_proba_pseudo = logreg_pseudo.predict_proba(X_test)
    Y_test_pred_pseudo = (Y_test_proba_pseudo > 0.5).astype(int)

    score_pseudo = f1_score(Y_test, Y_test_pred_pseudo, average='macro')
    print(f"Score F1 AVANT Pseudo-Labeling : {score_logreg:.4f}")
    print(f"Score F1 APRES Pseudo-Labeling : {score_pseudo:.4f}")


    print("\n--- Génération des graphiques de Pseudo-Labeling ---")
    nom_modele = "LogReg_Perch"

    print("Génération du graphe de compromis (Tradeoff)...")
    plot_pseudo_label_tradeoff(Y_test, Y_test_proba_logreg, nom_modele)

    print("Génération du graphe de comparaison PR...")
    plot_pr_comparison(Y_test, Y_test_proba_logreg, Y_test_proba_pseudo, nom_modele)

    print("Préparation des données pour l'UMAP...")
    
    f1_scores_base = f1_score(Y_test, Y_test_pred_logreg, average=None)
    pire_espece_idx = np.argmin(f1_scores_base)
    
    mask_true = Y_train_merge[:, pire_espece_idx] == 1
    X_true_umap = X_train_merge[mask_true]

    mask_pseudo = Y_pseudo_confident[:, pire_espece_idx] == 1
    X_pseudo_umap = X_pseudo_confident[mask_pseudo]

    mask_noise = Y_train_merge[:, pire_espece_idx] == 0
    X_noise_umap = X_train_merge[mask_noise][:1000] 

    if len(X_pseudo_umap) > 0 and len(X_true_umap) > 0:
        print(f"Génération de l'UMAP pour l'espèce n°{pire_espece_idx}...")
        plot_umap_projection(X_true_umap, X_pseudo_umap, X_noise_umap)
    else:
        print(f"Impossible de faire l'UMAP : pas assez de vrais/faux labels pour l'espèce {pire_espece_idx}.")


    '''print("\n--- Évaluation avec KNN (Recherche par similarité) ---")
    knn_model = OneVsRestClassifier(
        KNeighborsClassifier(n_neighbors=3, algorithm='brute', metric='cosine', n_jobs=-1)
    )

    start_time = time.time()
    knn_model.fit(X_train_merge, Y_train_merge)
    print(f"Mémorisation terminée en {time.time() - start_time:.2f} secondes.")

    start_time = time.time()
    Y_test_proba_knn = knn_model.predict_proba(X_test)
    print(f"Prédictions terminées en {(time.time() - start_time) / 60:.1f} minutes.")

    Y_test_pred_knn = (Y_test_proba_knn > 0.5).astype(int)

    score_knn = f1_score(Y_test, Y_test_pred_knn, average='macro')
    print(f"Score F1 KNN (Few-Shot) : {score_knn:.4f}")'''
    
if __name__ == "__main__":
    main()
