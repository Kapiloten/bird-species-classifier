from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split


#Duplicate vectors from rare classes by adding a small gaussian noise to make the split for train/test/val possible
def increase_rare_birds(X, y, min_threshold=5, noise=0.02):
    classes, counts = np.unique(y, return_counts=True)
    rare_birds = classes[counts < min_threshold]
    
    if len(rare_birds) == 0:
        return X, y
            
    new_X = []
    new_y = []
    
    global_std = np.std(X)
    noise_calc = global_std * noise

    for bird in rare_birds:
        existing_birds = X[y == bird]
        nb_existing = len(existing_birds)
        
        nb_needed = min_threshold - nb_existing
        
        for _ in range(nb_needed):
            base_vect = existing_birds[np.random.randint(0, nb_existing)]
            
            noise_vect = np.random.normal(loc=0.0, scale=   noise_calc, size=base_vect.shape)
            
            mutant = base_vect + noise_vect
            
            new_X.append(mutant)
            new_y.append(bird)
            
    X_final = np.vstack([X, np.array(new_X)])
    y_final = np.concatenate([y, np.array(new_y)])
        
    return X_final, y_final

def load_data():
    '''Only use this for pre trained embedding models, use load_data2 otherwise.'''
    base_path = Path("ressource/processed_data")
    X = np.load(base_path / "X_embeddings_perch.npy")
    Y = np.load(base_path / "y_labels.npy")

    X,Y = increase_rare_birds(X,Y)


    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X, Y, 
        test_size=0.20, 
        stratify=Y,   
        random_state=12 
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, Y_temp, 
        test_size=0.1875, 
        stratify=Y_temp, 
        random_state=12
    )

    return (
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test
    )

def load_data2():
    '''Only use this for manual embedding models, use load_data2 otherwise.'''
    META_COLUMNS = ["filepath", "filename", "label", "common_name", "scientific_name", "split"]
    audio_features = pd.read_csv("ressource/features/features.csv")
    mfcc_features = pd.read_csv("ressource/features/mfcc_features.csv")
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


def save_results(modelName,path,y_test, predictions, test_accuracy, test_f1):
    RESULTS_DIR = Path(path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(
        [{"model": modelName, "test_accuracy": test_accuracy, "test_f1_macro": test_f1}]
    )
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)

    labels = list(np.unique(y_test))
    report = classification_report(y_test, predictions, labels=labels, zero_division=0)
    (RESULTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(RESULTS_DIR / "confusion_matrix.csv")
