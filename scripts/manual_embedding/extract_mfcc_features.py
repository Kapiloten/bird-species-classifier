from pathlib import Path

import librosa
import numpy as np
import pandas as pd


SPLIT_CSV = Path("splits/audio_split_65_15_20.csv")
OUTPUT_CSV = Path("ressource/features/mfcc_features.csv")

SAMPLE_RATE = 32000
N_MFCC = 20

LIMIT = None


def extract_mfcc(audio_path):
    signal, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=N_MFCC)

    features = {}
    for i in range(N_MFCC):
        features[f"mfcc_{i + 1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i + 1}_std"] = np.std(mfcc[i])

    return features


def main():
    split_df = pd.read_csv(SPLIT_CSV)

    if LIMIT is not None:
        split_df = split_df.head(LIMIT)

    rows = []

    for index, audio in split_df.iterrows():
        audio_path = Path(audio["filepath"])
        mfcc_features = extract_mfcc(audio_path)

        row = {
            "filepath": audio["filepath"],
            "filename": audio["filename"],
            "label": audio["label"],
            "common_name": audio["common_name"],
            "scientific_name": audio["scientific_name"],
            "split": audio["split"],
        }
        row.update(mfcc_features)
        rows.append(row)

        if index == 0 or (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(split_df)} audios traites")

    features_df = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Fichier cree : {OUTPUT_CSV}")
    print(f"Nombre d'audios : {len(features_df)}")
    print(f"Nombre de features MFCC : {N_MFCC * 2}")


if __name__ == "__main__":
    main()
