import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


from scripts.utils import load_csv_features, plot_umap_features_justification



import argparse


def main():
    parser = argparse.ArgumentParser(description="Multi-label window experiment from the current main workflow.")
    parser.add_argument("--features-csv", default="ressource/features/all_5s_features.csv")
    parser.add_argument("--labels-csv", default="ressource/raw_data/train_soundscapes_labels.csv")
    parser.add_argument("--output-dir", default="ressource/results/benchmark_mfcc")
    args = parser.parse_args()

    X_train, Y_train, X_val, Y_val, X_test, Y_test, X_nLabeled, Y_nLabeled, Z_nLabeled = load_csv_features(args.features_csv, args.labels_csv)


    plot_umap_features_justification(X_train, Y_train, top_k=6)


if __name__ == "__main__":
    main()
