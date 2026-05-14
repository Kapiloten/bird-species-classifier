from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def read_classes(path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def assign_splits(files, seed):
    shuffled_files = files[:]
    random.Random(seed).shuffle(shuffled_files)

    n_files = len(shuffled_files)
    n_val = round(n_files * 0.15)
    n_test = round(n_files * 0.20)
    n_train = n_files - n_val - n_test

    rows: list[tuple[Path, str]] = []
    rows.extend((path, "train") for path in shuffled_files[:n_train])
    rows.extend((path, "val") for path in shuffled_files[n_train : n_train + n_val])
    rows.extend((path, "test") for path in shuffled_files[n_train + n_val :])
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Create a stratified 65/15/20 split for the selected bird audio classes."
    )
    parser.add_argument("--classes", default="classes.txt", help="Selected class labels file.")
    parser.add_argument("--metadata", default="selected_classes.csv", help="Selected classes metadata CSV.")
    parser.add_argument("--audio-dir", default="train_audio", help="Directory containing one folder per class.")
    parser.add_argument("--output", default="splits/audio_split_65_15_20.csv", help="Output split CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    args = parser.parse_args()

    project_dir = Path.cwd()
    classes_path = project_dir / args.classes
    metadata_path = project_dir / args.metadata
    audio_dir = project_dir / args.audio_dir
    output_path = project_dir / args.output

    selected_labels = read_classes(classes_path)
    metadata = pd.read_csv(metadata_path).set_index("primary_label")

    rows = []
    for label in selected_labels:
        class_dir = audio_dir / label
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing audio directory for class: {class_dir}")

        files = sorted(class_dir.glob("*.ogg"))
        if not files:
            raise FileNotFoundError(f"No .ogg files found for class: {class_dir}")

        if label not in metadata.index:
            raise ValueError(f"Class {label!r} is missing from {metadata_path}")

        common_name = metadata.loc[label, "common_name"]
        scientific_name = metadata.loc[label, "scientific_name"]

        for filepath, split in assign_splits(files, seed=args.seed):
            rows.append(
                {
                    "filepath": filepath.relative_to(project_dir).as_posix(),
                    "filename": filepath.name,
                    "label": label,
                    "common_name": common_name,
                    "scientific_name": scientific_name,
                    "split": split,
                }
            )

    split_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)

    summary = (
        split_df.groupby(["label", "common_name", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    print(f"Split saved to: {output_path}")
    print(f"Seed: {args.seed}")
    print()
    print(summary.to_string(index=False))
    print()
    print("Total by split:")
    print(split_df["split"].value_counts().reindex(["train", "val", "test"]).to_string())


if __name__ == "__main__":
    main()
