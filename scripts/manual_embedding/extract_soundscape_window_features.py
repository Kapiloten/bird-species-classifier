from __future__ import annotations

import argparse
from pathlib import Path
import sys

import librosa
import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_paths import FEATURES_DIR, project_path


DEFAULT_LABELS_CSV = "train_soundscapes_labels.csv"
DEFAULT_AUDIO_DIR = "train_soundscapes"
DEFAULT_SAMPLE_SUBMISSION = "sample_submission.csv"
DEFAULT_OUTPUT_CSV = FEATURES_DIR / "soundscape_5s_features.csv"

SAMPLE_RATE = 32000
N_MFCC = 20
TARGET_PREFIX = "target_"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract manual acoustic features on 5-second train soundscape windows "
            "and add a binary multi-label target vector."
        )
    )
    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS_CSV,
        help="CSV containing filename, start, end and semicolon-separated labels.",
    )
    parser.add_argument(
        "--audio-dir",
        default=DEFAULT_AUDIO_DIR,
        help="Directory containing train soundscape audio files.",
    )
    parser.add_argument(
        "--sample-submission",
        default=DEFAULT_SAMPLE_SUBMISSION,
        help="CSV used to recover the complete BirdCLEF class order.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        help="Audio sample rate used by librosa.",
    )
    parser.add_argument(
        "--n-mfcc",
        type=int,
        default=N_MFCC,
        help="Number of MFCC coefficients to compute.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of windows to process for a quick test.",
    )
    return parser.parse_args()


def time_to_seconds(value) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    parts = str(value).strip().split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)

    return float(parts[0])


def format_seconds(seconds: float) -> str:
    if float(seconds).is_integer():
        return str(int(seconds))
    return f"{seconds:g}"


def add_mean_std(features: dict[str, float], name: str, values: np.ndarray) -> None:
    features[f"{name}_mean"] = float(np.mean(values))
    features[f"{name}_std"] = float(np.std(values))


def load_class_labels(sample_submission_path: Path, labels_df: pd.DataFrame) -> list[str]:
    if sample_submission_path.exists():
        columns = pd.read_csv(sample_submission_path, nrows=0).columns.tolist()
        return [str(column) for column in columns if column != "row_id"]

    labels = set()
    for raw_labels in labels_df["primary_label"]:
        labels.update(split_labels(raw_labels))
    return sorted(labels)


def split_labels(raw_labels) -> list[str]:
    if pd.isna(raw_labels):
        return []
    return [
        label.strip()
        for label in str(raw_labels).split(";")
        if label.strip()
    ]


def load_audio_window(
    audio_path: Path,
    start_seconds: float,
    end_seconds: float,
    sample_rate: int,
) -> np.ndarray:
    duration = end_seconds - start_seconds
    if duration <= 0:
        raise ValueError(f"Invalid window duration for {audio_path}: {duration}")

    signal, _ = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True,
        offset=start_seconds,
        duration=duration,
    )

    target_samples = int(round(duration * sample_rate))
    if signal.size < target_samples:
        signal = np.pad(signal, (0, target_samples - signal.size))
    elif signal.size > target_samples:
        signal = signal[:target_samples]

    if signal.size == 0:
        raise ValueError(f"Empty audio window for {audio_path}")

    return signal


def extract_window_features(
    audio_path: Path,
    start_seconds: float,
    end_seconds: float,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
) -> dict[str, float]:
    signal = load_audio_window(audio_path, start_seconds, end_seconds, sample_rate)

    features: dict[str, float] = {
        "duration": float(end_seconds - start_seconds),
    }

    rms = librosa.feature.rms(y=signal)[0]
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate)[0]
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc)

    add_mean_std(features, "rms", rms)
    add_mean_std(features, "zcr", zcr)
    add_mean_std(features, "spectral_centroid", centroid)
    add_mean_std(features, "spectral_bandwidth", bandwidth)
    add_mean_std(features, "spectral_rolloff", rolloff)

    for index in range(n_mfcc):
        add_mean_std(features, f"mfcc_{index + 1}", mfcc[index])

    return features


def make_target_vector(active_labels: set[str], class_labels: list[str]) -> dict[str, int]:
    return {
        f"{TARGET_PREFIX}{class_label}": int(class_label in active_labels)
        for class_label in class_labels
    }


def main():
    args = parse_args()

    labels_path = project_path(args.labels)
    audio_dir = project_path(args.audio_dir)
    sample_submission_path = project_path(args.sample_submission)
    output_path = project_path(args.output)

    labels_df = pd.read_csv(labels_path)
    if args.limit is not None:
        labels_df = labels_df.head(args.limit)

    class_labels = load_class_labels(sample_submission_path, labels_df)
    class_set = set(class_labels)

    rows = []
    unknown_labels = set()

    for index, row in labels_df.iterrows():
        filename = str(row["filename"])
        audio_path = audio_dir / filename
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio file: {audio_path}")

        start_seconds = time_to_seconds(row["start"])
        end_seconds = time_to_seconds(row["end"])
        active_labels = set(split_labels(row["primary_label"]))
        unknown_labels.update(active_labels - class_set)

        features = extract_window_features(
            audio_path,
            start_seconds,
            end_seconds,
            sample_rate=args.sample_rate,
            n_mfcc=args.n_mfcc,
        )

        output_row = {
            "row_id": f"{Path(filename).stem}_{format_seconds(end_seconds)}",
            "filename": filename,
            "start": row["start"],
            "end": row["end"],
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "label_list": ";".join(label for label in class_labels if label in active_labels),
        }
        output_row.update(features)
        output_row.update(make_target_vector(active_labels, class_labels))
        rows.append(output_row)

        if index == 0 or (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(labels_df)} fenetres traitees")

    features_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    n_manual_features = 1 + 5 * 2 + args.n_mfcc * 2
    print(f"Fichier cree : {output_path}")
    print(f"Nombre de fenetres : {len(features_df)}")
    print(f"Nombre de features audio : {n_manual_features}")
    print(f"Nombre de classes multi-label : {len(class_labels)}")

    if unknown_labels:
        labels = ", ".join(sorted(unknown_labels))
        print(f"Labels ignores car absents du sample submission : {labels}")


if __name__ == "__main__":
    main()
