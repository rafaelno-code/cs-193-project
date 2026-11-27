from dataclasses import dataclass, replace

import librosa
import numpy as np

from .notes import Note

MIN_BPM, MAX_BPM = 70.0, 160.0 

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)
PITCH_CLASSES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


@dataclass
class Key:
    tonic: int 
    mode: str 
    strength: float 

    def __str__(self) -> str:
        return f"{PITCH_CLASSES[self.tonic]} {self.mode}"

    @property
    def scale(self) -> set[int]:
        steps = MAJOR_SCALE if self.mode == "major" else MINOR_SCALE
        return {(self.tonic + s) % 12 for s in steps}

def estimate_tempo(notes: list[Note], hop_seconds: float = 0.01) -> float:
    if len(notes) < 3:
        return 120.0
    starts = np.array([n.start for n in notes]) - notes[0].start
    envelope = np.zeros(int(starts[-1] / hop_seconds) + 50)
    envelope[np.round(starts / hop_seconds).astype(int)] = 1.0
    envelope = np.convolve(envelope, np.hanning(7), mode="same")

    sr = 16000
    hop_length = int(round(sr * hop_seconds))
    bpm = float(librosa.feature.tempo(onset_envelope=envelope, sr=sr, hop_length=hop_length)[0])
    return _refine_tempo(starts, _fold_tempo(bpm))


def _refine_tempo(starts: np.ndarray, bpm: float, search: float = 0.08) -> float:
    candidates = bpm * np.linspace(1 - search, 1 + search, 161)
    steps = starts[None, :] * candidates[:, None] / 60.0 * 2 
    error = np.mean((steps - np.round(steps)) ** 2, axis=1)
    return float(candidates[np.argmin(error)])


def _fold_tempo(bpm: float) -> float:
    while bpm < MIN_BPM:
        bpm *= 2
    while bpm > MAX_BPM:
        bpm /= 2
    return bpm

def quantize_timing(notes: list[Note], bpm: float, steps_per_beat: int = 4) -> list[Note]:
    if not notes:
        return []
    step = 60.0 / bpm / steps_per_beat
    anchor = notes[0].start

    snapped = []
    for n in notes:
        start = round((n.start - anchor) / step)
        end = max(start + 1, round((n.end - anchor) / step))
        snapped.append((start, end, n))

    by_start: dict[int, tuple[int, int, Note]] = {}
    for s, e, n in snapped:
        if s not in by_start or n.duration > by_start[s][2].duration:
            by_start[s] = (s, e, n)
    ordered = [by_start[s] for s in sorted(by_start)]

    out = []
    for i, (s, e, n) in enumerate(ordered):
        if i + 1 < len(ordered):
            e = min(e, ordered[i + 1][0])  # no overlaps in a monophonic line
        out.append(replace(n, start=s * step, end=e * step))
    return out

def detect_key(notes: list[Note]) -> Key:
    histogram = np.zeros(12)
    for n in notes:
        histogram[n.pitch % 12] += n.duration
    if not histogram.any():
        return Key(0, "major", 0.0)

    best = Key(0, "major", -np.inf)
    for tonic in range(12):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            r = float(np.corrcoef(histogram, np.roll(profile, tonic))[0, 1])
            if r > best.strength:
                best = Key(tonic, mode, r)
    return best


def snap_to_key(notes: list[Note], key: Key) -> list[Note]:
    scale = key.scale
    out = []
    for i, n in enumerate(notes):
        if n.pitch % 12 in scale:
            out.append(n)
            continue
        candidates = [p for p in (n.pitch - 1, n.pitch + 1) if p % 12 in scale]
        previous = out[-1].pitch if out else n.pitch
        pitch = min(candidates, key=lambda p: abs(p - previous))
        out.append(replace(n, pitch=pitch))
    return out
