"""Extract Perch embeddings per five-second window for multi-label experiments."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from perch_embedding.extract_recording_embeddings import (
    MODEL_URL, create_dataset_class, enable_tensorflow_gpu_growth, load_perch_dependencies,
)
from project_paths import PROJECT_ROOT, RAW_DATA_DIR, first_existing_path, project_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", default=str(RAW_DATA_DIR))
    parser.add_argument("--sources", nargs="+", choices=["train_audio", "train_soundscapes"], default=["train_audio"])
    parser.add_argument("--output-dir", default="ressource/processed_data4")
    return parser.parse_args()


def main():
    args = parse_args()
    librosa, tf, hub, torch, DataLoader, Dataset, tqdm = load_perch_dependencies()
    enable_tensorflow_gpu_growth(tf)
    model = hub.load(MODEL_URL)
    BirdDataset = create_dataset_class(librosa, torch, Dataset)
    output = project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for source in args.sources:
        audio_dir = first_existing_path(project_path(args.raw_data_dir) / source, PROJECT_ROOT / source)
        dataset = BirdDataset(audio_dir)
        if not len(dataset):
            raise FileNotFoundError(f"No .ogg recordings found under {audio_dir}")
        embeddings, labels, row_ids, failures = [], [], [], []
        for waveforms, batch_labels, paths in tqdm(DataLoader(dataset, batch_size=1, shuffle=False), desc=source):
            try:
                frames = tf.signal.frame(tf.convert_to_tensor(waveforms[0].numpy(), dtype=tf.float32),
                                         frame_length=160000, frame_step=160000, pad_end=True)
                file_vectors, file_ids = [], []
                for index in range(frames.shape[0]):
                    outputs = model.infer_tf(tf.expand_dims(frames[index], 0))
                    key = "embeddings" if "embeddings" in outputs else "embedding"
                    file_vectors.append(np.squeeze(outputs[key].numpy()))
                    file_ids.append(f"{Path(paths[0]).stem}_{(index + 1) * 5}")
                embeddings.extend(file_vectors)
                row_ids.extend(file_ids)
                if source == "train_audio":
                    labels.extend([batch_labels[0]] * len(file_vectors))
            except Exception as error:
                failures.append({"filepath": str(paths[0]), "error": str(error)})
                print(f"Error on {paths[0]}: {error}")
        if not embeddings:
            raise RuntimeError(f"No embeddings extracted for {source}")
        np.save(output / f"X_embeddings_perch_{source}.npy", np.vstack(embeddings))
        np.save(output / f"row_ids_{source}.npy", np.asarray(row_ids))
        if source == "train_audio":
            np.save(output / f"y_labels_{source}.npy", np.asarray(labels))
        metadata = {"model_url": MODEL_URL, "sample_rate": 32000, "frame_samples": 160000,
                    "aggregation": "none; one row per window", "n_windows": len(row_ids), "failures": failures}
        (output / f"extraction_{source}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved {len(row_ids)} {source} windows to {output}")


if __name__ == "__main__":
    main()
