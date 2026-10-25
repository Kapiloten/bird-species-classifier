import sys
from pathlib import Path

import numpy as np
sys.path.append(str(Path(__file__).parent.parent.parent))
from sklearn.metrics import f1_score
import joblib

from scripts.utils import load_data



import argparse
from scripts.project_paths import project_path


def main():
    parser = argparse.ArgumentParser(description="Multi-label window experiment from the current main workflow.")
    parser.add_argument("--embeddings-dir", default="ressource/processed_data4")
    parser.add_argument("--output-dir", default="ressource/results")
    args = parser.parse_args()

    X_train, Y_train, X_val, Y_val, X_test, Y_test, X_nLabeled, Y_nLabeled, Z_nLabeled = load_data(args.embeddings_dir)

    SAVE_DIR = str(project_path(args.output_dir))

    modeles_a_tester = ["final_logreg", "pseudo_logreg"]

    for nom in modeles_a_tester:
        chemin_modele = f"{SAVE_DIR}/{nom}_model.pkl"

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

            score_train = f1_score(Y_train, Y_train_pred, average='macro')
            score_val = f1_score(Y_val, Y_val_pred, average='macro')
            score_test = f1_score(Y_test, Y_test_pred, average='macro')

            if nom == "final_logreg":
                nom = "Perch_LogReg_Lineaire"
            else:
                nom = "Perch_LogReg_Lineaire_Pseudo_Label"

            print(f"Modèle : {nom}")
            print(f"   -> F1-Score sur le set de TRAIN : {score_train:.4f}")
            print(f"   -> F1-Score sur le set de VALIDATION : {score_val:.4f}")
            print(f"   -> F1-Score sur le set de TEST : {score_test:.4f}")


        except FileNotFoundError:
            print(f"Modèle {nom} introuvable dans {SAVE_DIR}/")


if __name__ == "__main__":
    main()
