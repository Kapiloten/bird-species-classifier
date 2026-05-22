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

modeles_a_tester = ["LogReg_Lineaire", "RandomForest", "Approximation_RBF_Nystroem"]

for nom in modeles_a_tester:
    chemin_modele = f"{SAVE_DIR}/{nom}_Best_Model.pkl"
    
    try:
        modele = joblib.load(chemin_modele)
        
        if "RandomForest" in nom:
            probas_brutes = modele.predict_proba(X_train)
            probas_val = modele.predict_proba(X_val)
            probas_test = modele.predict_proba(X_test)
            Y_train_proba = np.array([p[:, 1] if p.shape[1] > 1 else np.zeros(len(X_train)) for p in probas_brutes]).T
            Y_val_proba = np.array([p[:, 1] if p.shape[1] > 1 else np.zeros(len(X_val)) for p in probas_val]).T
            Y_test_proba = np.array([p[:, 1] if p.shape[1] > 1 else np.zeros(len(X_test)) for p in probas_test]).T

        else:
            Y_train_proba = modele.predict_proba(X_train)
            Y_val_proba = modele.predict_proba(X_val)
            Y_test_proba = modele.predict_proba(X_test)

            
        Y_train_pred = (Y_train_proba > 0.5).astype(int)
        Y_val_pred = (Y_val_proba > 0.5).astype(int)
        Y_test_pred = (Y_test_proba > 0.5).astype(int)
        
        # 3. On calcule le F1-Macro
        score_train = f1_score(Y_train, Y_train_pred, average='macro')
        score_val = f1_score(Y_val, Y_val_pred, average='macro')
        score_test = f1_score(Y_test, Y_test_pred, average='macro')
        
        print(f"Modèle : {nom}")
        print(f"   -> F1-Score sur le set de TRAIN : {score_train:.4f}")
        print(f"   -> F1-Score sur le set de VALIDATION : {score_val:.4f}")
        print(f"   -> F1-Score sur le set de TEST : {score_test:.4f}")

        
    except FileNotFoundError:
        print(f"Modèle {nom} introuvable dans {SAVE_DIR}/")