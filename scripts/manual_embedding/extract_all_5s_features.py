from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import lru_cache
import math
from pathlib import Path
import sys

import librosa
import numpy as np
from scipy.fft import dct
import soundfile as sf

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_paths import FEATURES_DIR, PROJECT_ROOT, project_path


DEFAULT_OUTPUT_CSV = FEATURES_DIR / "all_5s_features.csv"
DEFAULT_SOUNDSCAPE_LABELS_CSV = "train_soundscapes_labels.csv"

SAMPLE_RATE = 32000
WINDOW_SECONDS = 5
N_MFCC = 20
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
EPSILON = np.finfo(np.float32).eps


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract manual acoustic features for every 5-second window in "
            "train_audio and train_soundscapes."
        )
    )
    parser.add_argument(
        "--train-audio-dir",
        default="train_audio",
        help="Directory containing class subfolders with train audio files.",
    )
    parser.add_argument(
        "--train-soundscapes-dir",
        default="train_soundscapes",
        help="Directory containing train soundscape audio files.",
    )
    parser.add_argument(
        "--soundscape-labels",
        default=DEFAULT_SOUNDSCAPE_LABELS_CSV,
        help=(
            "CSV used to select train_soundscapes windows. By default only the "
            "unique windows present in train_soundscapes_labels.csv are extracted."
        ),
    )
    parser.add_argument(
        "--all-soundscape-windows",
        action="store_true",
        help="Extract every 5-second window from every train_soundscapes file.",
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
        "--window-seconds",
        type=int,
        default=WINDOW_SECONDS,
        help="Window size in seconds.",
    )
    parser.add_argument(
        "--n-mfcc",
        type=int,
        default=N_MFCC,
        help="Number of MFCC coefficients to compute.",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=N_FFT,
        help="FFT window size used by librosa features.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=HOP_LENGTH,
        help="Hop length used by librosa features.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Optional number of audio files to process for a quick test.",
    )
    parser.add_argument(
        "--start-file-index",
        type=int,
        default=1,
        help="1-based index of the first audio file to process. Useful for resuming.",
    )
    parser.add_argument(
        "--limit-windows",
        type=int,
        default=None,
        help="Optional number of windows to process for a quick test.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip unreadable audio files instead of stopping the extraction.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker threads to use.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="Number of files sent to each worker batch when --jobs > 1.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path where progress messages are also written.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print progress messages to stdout.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows to an existing output CSV instead of overwriting it.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Log progress every N processed files.",
    )
    return parser.parse_args()


def format_seconds(seconds: float) -> str:
    if float(seconds).is_integer():
        return str(int(seconds))
    return f"{seconds:g}"


def time_to_seconds(value) -> float:
    parts = str(value).strip().split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(parts[0])


def format_timestamp(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def add_mean_std(features: dict[str, float], name: str, values: np.ndarray) -> None:
    features[f"{name}_mean"] = float(np.mean(values))
    features[f"{name}_std"] = float(np.std(values))


def feature_columns(n_mfcc: int) -> list[str]:
    columns = [
        "duration",
        "rms_mean",
        "rms_std",
        "zcr_mean",
        "zcr_std",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
    ]
    for index in range(n_mfcc):
        columns.extend([f"mfcc_{index + 1}_mean", f"mfcc_{index + 1}_std"])
    return columns


def load_audio(audio_path: Path, sample_rate: int) -> np.ndarray:
    signal, native_sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)

    if native_sample_rate != sample_rate:
        signal = librosa.resample(
            signal,
            orig_sr=native_sample_rate,
            target_sr=sample_rate,
        )

    return np.asarray(signal, dtype=np.float32)


@lru_cache(maxsize=16)
def mel_filter_bank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    return librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
    ).astype(np.float32)


def frame_signal(signal: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    if signal.size < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.size))

    n_frames = 1 + (signal.size - n_fft) // hop_length
    shape = (n_frames, n_fft)
    strides = (signal.strides[0] * hop_length, signal.strides[0])
    return np.lib.stride_tricks.as_strided(
        signal,
        shape=shape,
        strides=strides,
        writeable=False,
    )


def extract_signal_features(
    signal: np.ndarray,
    sample_rate: int,
    window_seconds: int,
    n_mfcc: int,
) -> dict[str, float]:
    features: dict[str, float] = {
        "duration": float(window_seconds),
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


def feature_frame_matrix(
    signal: np.ndarray,
    sample_rate: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
) -> dict[str, np.ndarray]:
    frames = frame_signal(signal, n_fft=n_fft, hop_length=hop_length)
    window = np.hanning(n_fft).astype(np.float32)
    windowed_frames = frames * window

    spectrum = np.fft.rfft(windowed_frames, axis=1)
    power = np.abs(spectrum).astype(np.float32) ** 2
    magnitude = np.sqrt(power)
    magnitude_sum = magnitude.sum(axis=1) + EPSILON
    frequencies = np.fft.rfftfreq(n_fft, d=1 / sample_rate).astype(np.float32)

    rms = np.sqrt(np.mean(frames * frames, axis=1))
    zcr = np.mean(np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]), axis=1)
    centroid = (magnitude @ frequencies) / magnitude_sum
    bandwidth = np.sqrt(
        (magnitude * (frequencies[None, :] - centroid[:, None]) ** 2).sum(axis=1)
        / magnitude_sum
    )

    cumulative_magnitude = np.cumsum(magnitude, axis=1)
    rolloff_threshold = 0.85 * magnitude_sum
    rolloff_index = (cumulative_magnitude >= rolloff_threshold[:, None]).argmax(axis=1)
    rolloff = frequencies[rolloff_index]

    mel_basis = mel_filter_bank(sample_rate, n_fft, N_MELS)
    mel_power = power @ mel_basis.T
    log_mel_power = np.log(np.maximum(mel_power, EPSILON))
    mfcc = dct(log_mel_power, type=2, norm="ortho", axis=1)[:, :n_mfcc].T

    frame_features = {
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
    }
    for index in range(n_mfcc):
        frame_features[f"mfcc_{index + 1}"] = mfcc[index]

    return frame_features


def aggregate_frame_features(
    frame_features: dict[str, np.ndarray],
    window_index: int,
    window_samples: int,
    hop_length: int,
) -> dict[str, float]:
    start_sample = window_index * window_samples
    end_sample = start_sample + window_samples
    start_frame = math.ceil(start_sample / hop_length)
    end_frame = math.ceil(end_sample / hop_length)

    features: dict[str, float] = {}
    for name, values in frame_features.items():
        window_values = values[start_frame:end_frame]
        if window_values.size == 0:
            window_values = values[max(0, start_frame - 1) : start_frame]
        add_mean_std(features, name, window_values)

    return features


def load_soundscape_windows(
    labels_path: Path,
    window_seconds: int,
) -> dict[str, list[int]]:
    selected_windows: dict[str, set[int]] = {}
    with labels_path.open("r", encoding="utf-8", newline="") as labels_file:
        reader = csv.DictReader(labels_file)
        for row in reader:
            filename = row["filename"]
            end_seconds = time_to_seconds(row["end"])
            window_index = int(round(end_seconds / window_seconds)) - 1
            selected_windows.setdefault(filename, set()).add(window_index)

    return {
        filename: sorted(indices)
        for filename, indices in selected_windows.items()
    }


def iter_audio_files(
    train_audio_dir: Path,
    train_soundscapes_dir: Path,
    soundscape_labels_path: Path,
    all_soundscape_windows: bool,
    window_seconds: int,
):
    for path in sorted(train_audio_dir.rglob("*.ogg")):
        yield "train_audio", train_audio_dir, path, None

    if all_soundscape_windows:
        for path in sorted(train_soundscapes_dir.glob("*.ogg")):
            yield "train_soundscapes", train_soundscapes_dir, path, None
        return

    selected_windows = load_soundscape_windows(
        soundscape_labels_path,
        window_seconds=window_seconds,
    )
    for filename, window_indices in sorted(selected_windows.items()):
        yield "train_soundscapes", train_soundscapes_dir, train_soundscapes_dir / filename, window_indices


def make_row_stem(base_dir: Path, audio_path: Path) -> str:
    relative_stem = audio_path.relative_to(base_dir).with_suffix("")
    return relative_stem.as_posix().replace("/", "_")


def count_csv_data_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        line_count = sum(1 for _ in csv_file)
    return max(0, line_count - 1)


def make_rows_for_file(
    source: str,
    base_dir: Path,
    audio_path: Path,
    sample_rate: int,
    window_seconds: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
    selected_window_indices: list[int] | None,
):
    signal = load_audio(audio_path, sample_rate)
    if signal.size == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

    window_samples = int(round(window_seconds * sample_rate))
    n_windows = math.ceil(signal.size / window_samples)
    target_samples = n_windows * window_samples
    if signal.size < target_samples:
        signal = np.pad(signal, (0, target_samples - signal.size))

    frame_features = feature_frame_matrix(
        signal=signal,
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    row_stem = make_row_stem(base_dir, audio_path)
    relative_path = audio_path.relative_to(PROJECT_ROOT).as_posix()

    window_indices = range(n_windows) if selected_window_indices is None else selected_window_indices
    for window_index in window_indices:
        if window_index < 0 or window_index >= n_windows:
            raise ValueError(f"Invalid window index {window_index} for {audio_path}")

        start_seconds = window_index * window_seconds
        end_seconds = (window_index + 1) * window_seconds
        features = {
            "duration": float(window_seconds),
        }
        features.update(
            aggregate_frame_features(
                frame_features=frame_features,
                window_index=window_index,
                window_samples=window_samples,
                hop_length=hop_length,
            )
        )

        row = {
            "row_id": f"{row_stem}_{format_seconds(end_seconds)}",
            "source": source,
            "filepath": relative_path,
            "filename": audio_path.name,
            "start": format_timestamp(start_seconds),
            "end": format_timestamp(end_seconds),
            "start_seconds": float(start_seconds),
            "end_seconds": float(end_seconds),
        }
        row.update(features)
        yield row


def process_file_task(task):
    (
        source,
        base_dir,
        audio_path,
        selected_window_indices,
        sample_rate,
        window_seconds,
        n_mfcc,
        n_fft,
        hop_length,
    ) = task

    source = str(source)
    base_dir = Path(base_dir)
    audio_path = Path(audio_path)

    try:
        rows = list(
            make_rows_for_file(
                source=source,
                base_dir=base_dir,
                audio_path=audio_path,
                sample_rate=sample_rate,
                window_seconds=window_seconds,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length,
                selected_window_indices=selected_window_indices,
            )
        )
    except Exception as error:
        return source, str(audio_path), [], repr(error)

    return source, str(audio_path), rows, None


def main():
    args = parse_args()

    train_audio_dir = project_path(args.train_audio_dir)
    train_soundscapes_dir = project_path(args.train_soundscapes_dir)
    soundscape_labels_path = project_path(args.soundscape_labels)
    output_path = project_path(args.output)
    log_path = project_path(args.log_file) if args.log_file else None
    if log_path is not None and not args.append:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")

    def emit(message: str) -> None:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"{message}\n")
        if not args.quiet:
            print(message, flush=True)

    jobs = args.jobs
    if args.limit_windows is not None and jobs > 1:
        emit("--limit-windows actif : utilisation de --jobs 1 pour arreter au bon moment.")
        jobs = 1

    files = list(
        iter_audio_files(
            train_audio_dir=train_audio_dir,
            train_soundscapes_dir=train_soundscapes_dir,
            soundscape_labels_path=soundscape_labels_path,
            all_soundscape_windows=args.all_soundscape_windows,
            window_seconds=args.window_seconds,
        )
    )
    total_file_count = len(files)
    if args.start_file_index < 1:
        raise ValueError("--start-file-index must be >= 1")
    if args.start_file_index > 1:
        files = files[args.start_file_index - 1 :]
    if args.limit_files is not None:
        files = files[: args.limit_files]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "row_id",
        "source",
        "filepath",
        "filename",
        "start",
        "end",
        "start_seconds",
        "end_seconds",
        *feature_columns(args.n_mfcc),
    ]

    total_windows = count_csv_data_rows(output_path) if args.append else 0
    skipped_files = 0
    tasks = [
        (
            source,
            str(base_dir),
            str(audio_path),
            selected_window_indices,
            args.sample_rate,
            args.window_seconds,
            args.n_mfcc,
            args.n_fft,
            args.hop_length,
        )
        for source, base_dir, audio_path, selected_window_indices in files
    ]

    output_mode = "a" if args.append else "w"
    write_header = not args.append or not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open(output_mode, encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        if write_header:
            writer.writeheader()

        if jobs <= 1:
            for file_index, (source, base_dir, audio_path, selected_window_indices) in enumerate(
                files,
                start=args.start_file_index,
            ):
                if args.limit_windows is not None and total_windows >= args.limit_windows:
                    break

                try:
                    rows = make_rows_for_file(
                        source=source,
                        base_dir=base_dir,
                        audio_path=audio_path,
                        sample_rate=args.sample_rate,
                        window_seconds=args.window_seconds,
                        n_mfcc=args.n_mfcc,
                        n_fft=args.n_fft,
                        hop_length=args.hop_length,
                        selected_window_indices=selected_window_indices,
                    )

                    for row in rows:
                        if args.limit_windows is not None and total_windows >= args.limit_windows:
                            break

                        writer.writerow(row)
                        total_windows += 1

                except Exception as error:
                    if not args.skip_errors:
                        raise
                    skipped_files += 1
                    emit(f"Fichier ignore ({error}): {audio_path}")

                if file_index == args.start_file_index or file_index % args.progress_every == 0:
                    emit(
                        f"{file_index}/{total_file_count} fichiers, "
                        f"{total_windows} fenetres extraites"
                    )
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                results = executor.map(process_file_task, tasks, chunksize=args.chunksize)
                for file_index, (source, audio_path, rows, error) in enumerate(
                    results,
                    start=args.start_file_index,
                ):
                    if args.limit_windows is not None and total_windows >= args.limit_windows:
                        break

                    if error is not None:
                        if not args.skip_errors:
                            raise RuntimeError(f"{audio_path}: {error}")
                        skipped_files += 1
                        emit(f"Fichier ignore ({error}): {audio_path}")
                        continue

                    for row in rows:
                        if args.limit_windows is not None and total_windows >= args.limit_windows:
                            break

                        writer.writerow(row)
                        total_windows += 1

                    if file_index == args.start_file_index or file_index % args.progress_every == 0:
                        emit(
                            f"{file_index}/{total_file_count} fichiers, "
                            f"{total_windows} fenetres extraites"
                        )

    emit(f"Fichier cree : {output_path}")
    emit(f"Nombre de fichiers ignores : {skipped_files}")
    emit(f"Nombre de fenetres : {total_windows}")
    emit(f"Nombre de features audio : {len(feature_columns(args.n_mfcc))}")


if __name__ == "__main__":
    main()
