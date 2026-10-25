from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_paths import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TRAIN_AUDIO_DIR,
    first_existing_path,
    project_path,
)


MODEL_URL = (
    "https://www.kaggle.com/models/google/bird-vocalization-classifier/"
    "TensorFlow2/bird-vocalization-classifier/8"
)


def load_perch_dependencies():
    """Import the optional extraction stack only when extraction is requested."""
    if sys.version_info >= (3, 14):
        raise SystemExit("Perch extraction requires Python 3.11-3.13; use uv sync --python 3.11 --extra perch.")
    missing = []

    try:
        import librosa
    except ModuleNotFoundError:
        missing.append("librosa")
        librosa = None

    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        missing.append("tensorflow")
        tf = None

    try:
        import tensorflow_hub as hub
    except ModuleNotFoundError:
        missing.append("tensorflow-hub")
        hub = None

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ModuleNotFoundError:
        missing.append("torch")
        torch = None
        DataLoader = None
        Dataset = object

    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        missing.append("tqdm")
        tqdm = None

    if missing:
        package_list = " ".join(sorted(set(missing)))
        raise SystemExit(
            "Missing Perch dependencies. Install them before running this script:\n"
            f"uv sync --extra perch\nMissing packages: {package_list}"
        )

    return librosa, tf, hub, torch, DataLoader, Dataset, tqdm


def enable_tensorflow_gpu_growth(tf):
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow GPU memory growth enabled")
    except RuntimeError as error:
        print(error)


def create_dataset_class(librosa, torch, Dataset):
    """Build the existing Torch loader with deterministic recording order."""
    class BirdDataset(Dataset):
        def __init__(self, root_dir, target_sample_rate=32000):
            self.filepaths = []
            self.labels = []
            self.target_sample_rate = target_sample_rate

            root_path = Path(root_dir)
            for file_path in sorted(root_path.rglob("*.ogg")):
                self.filepaths.append(str(file_path))
                self.labels.append(file_path.parent.name)

        def __len__(self):
            return len(self.filepaths)

        def __getitem__(self, idx):
            path = self.filepaths[idx]
            label = self.labels[idx]

            np_waveform, _ = librosa.load(path, sr=self.target_sample_rate, mono=True)
            waveform = torch.from_numpy(np_waveform)

            return waveform, label, path

    return BirdDataset


def parse_args():
    default_audio_dir = first_existing_path(
        TRAIN_AUDIO_DIR,
        RAW_DATA_DIR / "train_audio",
    )

    parser = argparse.ArgumentParser(
        description="Extract Perch embeddings from BirdCLEF audio files."
    )
    parser.add_argument(
        "--audio-dir",
        default=str(default_audio_dir),
        help="Directory containing one folder per class.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROCESSED_DATA_DIR),
        help="Directory where X_embeddings_perch.npy and y_labels.npy are saved.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=[32000],
        default=32000,
        help="Target sample rate used by librosa.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    audio_dir = project_path(args.audio_dir)
    output_dir = project_path(args.output_dir)

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    librosa, tf, hub, torch, DataLoader, Dataset, tqdm = load_perch_dependencies()
    enable_tensorflow_gpu_growth(tf)

    model = hub.load(MODEL_URL)
    BirdDataset = create_dataset_class(librosa, torch, Dataset)
    dataset = BirdDataset(root_dir=audio_dir, target_sample_rate=args.sample_rate)

    if len(dataset) == 0:
        raise FileNotFoundError(f"No .ogg files found in: {audio_dir}")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    embeddings = []
    labels = []
    filepaths = []
    failures = []

    for batch_waveforms, batch_labels, batch_paths in tqdm(
        dataloader,
        desc="Extraction",
        unit="file",
    ):
        try:
            audio_flat = batch_waveforms[0].numpy()
            tf_audio = tf.convert_to_tensor(audio_flat, dtype=tf.float32)
            frames = tf.signal.frame(
                tf_audio,
                frame_length=160000,
                frame_step=160000,
                pad_end=True,
            )

            file_embeddings = []
            for i in range(frames.shape[0]):
                frame_input = tf.expand_dims(frames[i], 0)
                outputs = model.infer_tf(frame_input)

                if "embeddings" in outputs:
                    embedding = outputs["embeddings"].numpy()
                elif "embedding" in outputs:
                    embedding = outputs["embedding"].numpy()
                else:
                    raise KeyError("Perch output does not contain an embedding tensor")

                file_embeddings.append(embedding)

            mean_vector = np.mean(file_embeddings, axis=0)
            embeddings.append(np.squeeze(mean_vector))
            labels.extend(batch_labels)
            filepaths.append(str(Path(batch_paths[0]).relative_to(audio_dir).as_posix()))

        except Exception as error:
            print(f"\nError on {batch_paths[0]}: {error}")
            failures.append({"filepath": str(batch_paths[0]), "error": str(error)})
            continue

    if not embeddings:
        raise RuntimeError("No embeddings were extracted.")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_embeddings_perch.npy", np.vstack(embeddings))
    np.save(output_dir / "y_labels.npy", np.array(labels))
    np.save(output_dir / "filepaths.npy", np.array(filepaths))
    import json

    metadata = {"model_url": MODEL_URL, "sample_rate": args.sample_rate,
                "frame_samples": 160000, "aggregation": "mean", "n_recordings": len(labels),
                "n_failed": len(failures), "failures": failures}
    (output_dir / "extraction.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Data saved: {output_dir}")


if __name__ == "__main__":
    main()
