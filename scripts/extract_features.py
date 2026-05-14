from pathlib import Path

import librosa
import numpy as np
import pandas as pd


SPLIT_CSV = Path("splits/audio_split_65_15_20.csv")
OUTPUT_CSV = Path("features/features.csv")

SAMPLE_RATE = 32000

# Mets une valeur comme 10 pour tester rapidement, ou None pour tout traiter.
LIMIT = None


def add_mean_std(features, name, values):
    features[f"{name}_mean"] = np.mean(values)
    features[f"{name}_std"] = np.std(values)


def extract_features(audio_path):
    signal, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    features = {}
    features["duration"] = librosa.get_duration(y=signal, sr=sample_rate)

    rms = librosa.feature.rms(y=signal)[0]
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate)[0]

    add_mean_std(features, "rms", rms)
    add_mean_std(features, "zcr", zcr)
    add_mean_std(features, "spectral_centroid", centroid)
    add_mean_std(features, "spectral_bandwidth", bandwidth)
    add_mean_std(features, "spectral_rolloff", rolloff)

    return features


def main():
    split_df = pd.read_csv(SPLIT_CSV)

    if LIMIT is not None:
        split_df = split_df.head(LIMIT)

    rows = []

    for index, audio in split_df.iterrows():
        audio_path = Path(audio["filepath"])
        audio_features = extract_features(audio_path)

        row = {
            "filepath": audio["filepath"],
            "filename": audio["filename"],
            "label": audio["label"],
            "common_name": audio["common_name"],
            "scientific_name": audio["scientific_name"],
            "split": audio["split"],
        }
        row.update(audio_features)
        rows.append(row)

        if index == 0 or (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(split_df)} audios traites")

    features_df = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Fichier cree : {OUTPUT_CSV}")
    print(f"Nombre d'audios : {len(features_df)}")
    print(f"Nombre de features audio : {len(features_df.columns) - 6}")


if __name__ == "__main__":
    main()
