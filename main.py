def main():
    print(
        "Bird Species Classifier\n\n"
        "Rebuild archived result tables/figures (no audio required):\n"
        "  uv run python scripts/summarize_results.py\n\n"
        "Train the manual SVM from committed features:\n"
        "  uv run python scripts/manual_embedding/train_svm_rbf.py\n\n"
        "See README.md for datasets, Perch setup and evaluation limitations."
    )


if __name__ == "__main__":
    main()
