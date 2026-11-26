from dataclasses import dataclass

import librosa
import numpy as np
import torch
import torchcrepe

from .audio import SAMPLE_RATE

HOP_SECONDS = 0.01 
FMIN = 65.0  
FMAX = 1975.0  


@dataclass
class PitchTrack:

    times: np.ndarray 
    frequency: np.ndarray  
    confidence: np.ndarray 
    loudness_db: np.ndarray 
    hop_seconds: float


def track_pitch(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    hop_seconds: float = HOP_SECONDS,
    model: str = "full",
    device: str | None = None,
) -> PitchTrack:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    hop_length = int(round(sr * hop_seconds))

    frequency, confidence = torchcrepe.predict(
        torch.from_numpy(audio).unsqueeze(0),
        sr,
        hop_length,
        FMIN,
        FMAX,
        model=model,
        return_periodicity=True,
        batch_size=512,
        device=device,
        pad=True,
    )
    frequency = frequency.squeeze(0).cpu().numpy()
    confidence = confidence.squeeze(0).cpu().numpy()

    rms = librosa.feature.rms(y=audio, frame_length=4 * hop_length, hop_length=hop_length)[0]
    loudness_db = librosa.amplitude_to_db(rms, ref=np.max)
    loudness_db = _match_length(loudness_db, len(frequency), fill=loudness_db.min())

    times = np.arange(len(frequency)) * hop_seconds
    return PitchTrack(times, frequency, confidence, loudness_db, hop_seconds)


def _match_length(x: np.ndarray, n: int, fill: float) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.full(n - len(x), fill)])
