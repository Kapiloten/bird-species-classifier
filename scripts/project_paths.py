"""Resolve project assets independently of the shell's working directory."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESSOURCE_DIR = PROJECT_ROOT / "ressource"
FEATURES_DIR = RESSOURCE_DIR / "features"
PROCESSED_DATA_DIR = RESSOURCE_DIR / "processed_data"
RESULTS_DIR = RESSOURCE_DIR / "results"
SPLITS_DIR = PROJECT_ROOT / "splits"
TRAIN_AUDIO_DIR = PROJECT_ROOT / "train_audio"
RAW_DATA_DIR = RESSOURCE_DIR / "raw_data"


def project_path(path: str | Path) -> Path:
    """Return an absolute path, interpreting relative paths from the repository."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def first_existing_path(*paths: str | Path) -> Path:
    """Find the first existing candidate, or return the first as a useful default."""
    if not paths:
        raise ValueError("At least one candidate path is required.")
    for path in paths:
        candidate = project_path(path)
        if candidate.exists():
            return candidate

    return project_path(paths[0])
