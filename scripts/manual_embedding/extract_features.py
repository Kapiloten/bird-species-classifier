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
OUTPUT_CSV = FEATURES_DIR / "features.csv"

SAMPLE_RATE = 32000

# Mets une valeur comme 10 pour tester rapidement, ou None pour tout traiter.
LIMIT = None


def add_mean_std(features, name, values):
    """Add population mean/std summaries of a frame-level acoustic descriptor."""
    features[f"{name}_mean"] = np.mean(values)
    features[f"{name}_std"] = np.std(values)


def extract_features(audio_path):
    """Load mono 32 kHz audio and return duration plus ten acoustic summaries."""
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
    parser = argparse.ArgumentParser(description="Extract whole-recording acoustic features at 32 kHz.")
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    print(f"Fichier cree : {output_path}")
    print(f"Nombre d'audios : {len(features_df)}")
    print(f"Nombre de features audio : {len(features_df.columns) - 6}")


if __name__ == "__main__":
    main()
