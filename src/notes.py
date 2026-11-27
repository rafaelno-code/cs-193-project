from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .pitch import PitchTrack


@dataclass
class Note:
    pitch: int  
    start: float  
    end: float  
    velocity: int = 100

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SegmentSettings:
    confidence_threshold: float = 0.5  
    silence_db: float = -35.0  
    smoothing_frames: int = 7  
    min_note_seconds: float = 0.08 
    max_gap_seconds: float = 0.03  
    rearticulation_db: float = 6.0


def hz_to_midi(frequency: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 69 + 12 * np.log2(frequency / 440.0)


def median_smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x.copy()
    pad = width // 2
    padded = np.pad(x, pad, constant_values=np.nan)
    windows = sliding_window_view(padded, width)
    out = np.full_like(x, np.nan)
    voiced = ~np.isnan(x)
    if voiced.any():
        out[voiced] = np.nanmedian(windows[voiced], axis=1)
    return out


def estimate_tuning_offset(midi: np.ndarray) -> float:
    midi = midi[~np.isnan(midi)]
    if midi.size == 0:
        return 0.0
    angle = 2 * np.pi * (midi - np.round(midi)) 
    return float(np.angle(np.mean(np.exp(1j * angle))) / (2 * np.pi))


def segment_notes(track: PitchTrack, settings: SegmentSettings | None = None) -> list[Note]:
    s = settings or SegmentSettings()
    hop = track.hop_seconds

    voiced = (
        np.isfinite(track.frequency)
        & (track.frequency > 0)
        & (track.confidence >= s.confidence_threshold)
        & (track.loudness_db >= s.silence_db)
    )
    midi = np.where(voiced, hz_to_midi(track.frequency), np.nan)
    midi = median_smooth(midi, s.smoothing_frames)
    midi -= estimate_tuning_offset(midi)

    min_frames = max(1, int(round(s.min_note_seconds / hop)))
    max_gap = int(round(s.max_gap_seconds / hop))

    notes = []
    regions = _voiced_regions(~np.isnan(midi), max_gap)
    regions = _split_at_dips(regions, track.loudness_db, s.rearticulation_db)
    for first, last in regions:
        pitched = np.flatnonzero(~np.isnan(midi[first : last + 1]))
        if pitched.size == 0:
            continue
        first, last = first + pitched[0], first + pitched[-1]
        region = midi[first : last + 1]
        semitones = np.round(_fill_forward(region)).astype(int)
        for run_start, run_end, pitch in _stable_runs(semitones, min_frames):
            if run_end - run_start + 1 < min_frames:
                continue
            a, b = first + run_start, first + run_end
            loudness = float(np.mean(track.loudness_db[a : b + 1]))
            notes.append(Note(pitch, float(track.times[a]), float(track.times[b] + hop), _velocity(loudness)))
    return notes


def _voiced_regions(voiced: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(voiced)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > max_gap + 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def _split_at_dips(
    regions: list[tuple[int, int]], loudness_db: np.ndarray, depth_db: float, window: int = 10
) -> list[tuple[int, int]]:
    loud = np.convolve(loudness_db, np.ones(3) / 3, mode="same")
    out = []
    for first, last in regions:
        start = first
        for i in range(first + window, last - window + 1):
            if loud[i] > loud[i - 1] or loud[i] > loud[i + 1]:
                continue 
            before = loud[i - window : i].max()
            after = loud[i + 1 : i + 1 + window].max()
            if min(before, after) - loud[i] >= depth_db and i - start >= window:
                out.append((start, i - 1))
                start = i + 1
        out.append((start, last))
    return out


def _fill_forward(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    for i in range(1, len(x)):
        if np.isnan(x[i]):
            x[i] = x[i - 1]
    return x


def _stable_runs(semitones: np.ndarray, min_frames: int) -> list[tuple[int, int, int]]:
    change = np.flatnonzero(np.diff(semitones)) + 1
    bounds = np.concatenate([[0], change, [len(semitones)]])
    runs = [[bounds[i], bounds[i + 1] - 1, int(semitones[bounds[i]])] for i in range(len(bounds) - 1)]

    while len(runs) > 1:
        lengths = [r[1] - r[0] + 1 for r in runs]
        i = int(np.argmin(lengths))
        if lengths[i] >= min_frames:
            break
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]
        j = min(neighbours, key=lambda k: (abs(runs[k][2] - runs[i][2]), -lengths[k]))
        runs[j][0], runs[j][1] = min(runs[i][0], runs[j][0]), max(runs[i][1], runs[j][1])
        del runs[i]
        runs = _merge_equal_neighbours(runs)

    return [tuple(r) for r in _merge_equal_neighbours(runs)]


def _merge_equal_neighbours(runs: list[list[int]]) -> list[list[int]]:
    merged = [runs[0]]
    for r in runs[1:]:
        if r[2] == merged[-1][2]:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return merged


def _velocity(loudness_db: float) -> int:
    return int(np.clip(np.interp(loudness_db, [-40, 0], [40, 120]), 1, 127))