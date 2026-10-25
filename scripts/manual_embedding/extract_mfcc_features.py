import argparse
from pathlib import Path
import sys

import librosa
import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_paths import FEATURES_DIR, SPLITS_DIR, project_path


SPLIT_CSV = SPLITS_DIR / "audio_split_65_15_20.csv"
OUTPUT_CSV = FEATURES_DIR / "mfcc_features.csv"

SAMPLE_RATE = 32000
N_MFCC = 20

LIMIT = None


def extract_mfcc(audio_path):
    """Return mean/std of 20 librosa MFCCs for a complete mono 32 kHz recording."""
    signal, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=N_MFCC)

    features = {}
    for i in range(N_MFCC):
        features[f"mfcc_{i + 1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i + 1}_std"] = np.std(mfcc[i])

    return features


def main():
    parser = argparse.ArgumentParser(description="Extract whole-recording MFCC summaries at 32 kHz.")
    parser.add_argument("--split", default=str(SPLIT_CSV), help="Recording split CSV.")
    parser.add_argument("--output", default=str(OUTPUT_CSV), help="Output feature CSV.")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Process only the first N manifest rows.")
    args = parser.parse_args()
    output_path = project_path(args.output)
    split_df = pd.read_csv(project_path(args.split))

    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        split_df = split_df.head(args.limit)

    rows = []

    for index, audio in split_df.iterrows():
        audio_path = project_path(audio["filepath"])
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    print(f"Fichier cree : {output_path}")
    print(f"Nombre d'audios : {len(features_df)}")
    print(f"Nombre de features MFCC : {N_MFCC * 2}")


if __name__ == "__main__":
    main()
