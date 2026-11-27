from dataclasses import dataclass
from pathlib import Path

from .audio import load_audio
from .notes import Note, SegmentSettings, segment_notes
from .pitch import track_pitch
from .quantize import Key, detect_key, estimate_tempo, quantize_timing, snap_to_key


@dataclass
class Transcription:
    notes: list[Note]
    bpm: float
    key: Key


def transcribe(
    path: str | Path,
    settings: SegmentSettings | None = None,
    model: str = "full",
    bpm: float | None = None,
    steps_per_beat: int = 4,
    quantize: bool = True,
    snap_key: bool = False,
) -> Transcription:
    audio = load_audio(path)
    track = track_pitch(audio, model=model)
    notes = segment_notes(track, settings)

    key = detect_key(notes)
    tempo = bpm or estimate_tempo(notes, track.hop_seconds)
    if snap_key:
        notes = snap_to_key(notes, key)
    if quantize:
        notes = quantize_timing(notes, tempo, steps_per_beat)
    return Transcription(notes, tempo, key)
