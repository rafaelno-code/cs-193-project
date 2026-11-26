from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 16000


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No audio file at {path}")

    audio, _ = librosa.load(path, sr=sr, mono=True)
    if audio.size == 0:
        raise ValueError(f"{path} contains no audio")

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)
