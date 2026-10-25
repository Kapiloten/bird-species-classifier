from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from scripts.manual_embedding.extract_features import extract_features
from scripts.manual_embedding.extract_mfcc_features import extract_mfcc


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_audio_produces_finite_acoustic_and_mfcc_features(tmp_path):
    sample_rate = 32000
    signal = .8 * np.sin(2 * np.pi * 1000 * np.arange(sample_rate) / sample_rate)
    audio = tmp_path / "tone.wav"
    sf.write(audio, signal, sample_rate)
    features = {**extract_features(audio), **extract_mfcc(audio)}
    assert len(features) == 51
    assert np.isfinite(list(features.values())).all()
    assert features["duration"] == pytest.approx(1)
    assert features["spectral_centroid_mean"] == pytest.approx(1000, rel=.1)


@pytest.mark.parametrize("script", [
    "manual_embedding/create_audio_split.py",
    "manual_embedding/extract_features.py",
    "manual_embedding/extract_mfcc_features.py",
    "manual_embedding/train_logistic_regression.py",
    "manual_embedding/train_random_forest.py",
    "manual_embedding/train_svm_rbf.py",
    "manual_embedding/extract_all_5s_features.py",
    "manual_embedding/extract_soundscape_window_features.py",
    "perch_embedding/embedding_perch.py",
    "perch_embedding/extract_recording_embeddings.py",
    "perch_embedding/train_single_label.py",
    "perch_embedding/main.py",
    "perch_embedding/train_val_test.py",
    "manual_embedding/main.py",
    "manual_embedding/train_val_test_score.py",
    "manual_embedding/aa.py",
    "summarize_results.py",
])
def test_cli_help_works_outside_repository_without_datasets(script, tmp_path):
    completed = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"],
                               cwd=tmp_path, capture_output=True, text=True, timeout=45)
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
