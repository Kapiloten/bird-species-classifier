import json
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import Nystroem
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit

import os

from scripts.utils import load_csv_features, plot_pr_curve, plot_top_flop_f1, plot_worst_confusion


X_train, Y_train, X_val, Y_val, X_test, Y_test, X_nLabeled, Y_nLabeled, Z_nLabeled = load_csv_features()


SAVE_DIR = "ressource/results/benchmark_mfcc"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"📂 Toutes les sauvegardes seront stockées dans : {SAVE_DIR}/")

# --- CORRECTION DU LEAKAGE ET RÉCUPÉRATION DE X_VAL ---
# -1 = Données d'entraînement strictes
#  0 = Données de validation pour tester les hyperparamètres
indices_split = np.concatenate([
    np.full(len(X_train), -1),  
    np.full(len(X_val), 0)      
])
ps = PredefinedSplit(test_fold=indices_split)

X_train_full = np.concatenate((X_train, X_val))
Y_train_full = np.concatenate((Y_train, Y_val))


# Modèle 1 : Logistic Regression (Baseline Linéaire)
pipeline_logreg = make_pipeline(
    StandardScaler(), 
    OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight='balanced'))
)
param_logreg = {
    'onevsrestclassifier__estimator__C': [0.1, 1.0, 10.0]
}

resultats_scores = {}

# Modèle 2 : Random Forest (Arbres non-linéaires)
rf_model = RandomForestClassifier(n_jobs=3, random_state=15, class_weight='balanced')
param_rf = {
    'n_estimators': [50],
    'max_depth': [60],
}

# Modèle 3 : Approximation RBF (Nystroëm + LogReg)
pipeline_rbf = make_pipeline(
    StandardScaler(),
    Nystroem(random_state=15),
    OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight='balanced'))
)
param_rbf = {
    'nystroem__gamma': [0.05, 0.2, 1.0],        
    'nystroem__n_components': [100, 300],       
    'onevsrestclassifier__estimator__C': [0.1, 1.0, 10.0]
}

dictionnaire_modeles = {
    "RandomForest": (rf_model, param_rf),
    "Approximation_RBF_Nystroem": (pipeline_rbf, param_rbf)
}



for nom_modele, (modele, params) in dictionnaire_modeles.items():
    print(f"\n🚀 Lancement du GridSearch pour {nom_modele}...", flush=True)
    start_time = time.time()
    
    search = GridSearchCV(modele, params, cv=ps, scoring='f1_macro', n_jobs=1)
    
    search.fit(X_train_full, Y_train_full)
    meilleur_modele = search.best_estimator_
    
    print(f"Terminé ! Meilleurs paramètres : {search.best_params_}", flush=True)
    print(f"Temps de recherche total : {(time.time() - start_time)/60:.1f} minutes", flush=True)
    
    print(f"💾 Sauvegarde du modèle {nom_modele}...", flush=True)
    joblib.dump(search, f"{SAVE_DIR}/{nom_modele}_GridSearch_complet.pkl")
    joblib.dump(meilleur_modele, f"{SAVE_DIR}/{nom_modele}_Best_Model.pkl")
    
    if "RandomForest" in nom_modele:
        probas_brutes = meilleur_modele.predict_proba(X_test)
        Y_test_proba = np.array([p[:, 1] if p.shape[1] > 1 else np.zeros(len(X_test)) for p in probas_brutes]).T
    else:
        Y_test_proba = meilleur_modele.predict_proba(X_test)
        
    Y_test_pred = (Y_test_proba > 0.5).astype(int)
    
    np.save(f"{SAVE_DIR}/{nom_modele}_Y_test_proba.npy", Y_test_proba)
    np.save(f"{SAVE_DIR}/{nom_modele}_Y_test_pred.npy", Y_test_pred)
    
    score_f1 = f1_score(Y_test, Y_test_pred, average='macro')
    resultats_scores[nom_modele] = score_f1
    print(f"Score F1 (Macro) FINAL pour {nom_modele} : {score_f1:.4f}", flush=True)
    
    with open(f"{SAVE_DIR}/scores_en_direct.json", "w") as f:
        json.dump(resultats_scores, f, indent=4)
    
    print("Génération des graphiques...", flush=True)
    try:
        plot_pr_curve(Y_test, Y_test_proba, nom_modele)
        plot_top_flop_f1(Y_test, Y_test_pred, nom_modele)
        plot_worst_confusion(Y_test, Y_test_pred, nom_modele)
    except Exception as e:
        print(f"Erreur mineure lors des graphiques pour {nom_modele} : {e}", flush=True)


print("CLASSEMENT FINAL DES MODÈLES ", flush=True)
for nom, score in sorted(resultats_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{nom.ljust(30)} : {score:.4f} F1-Macro")

print(f"\nTout est terminé et sécurisé dans le dossier : {SAVE_DIR}/")
