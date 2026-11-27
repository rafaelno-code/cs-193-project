from pathlib import Path

from .audio import load_audio
from .notes import Note, SegmentSettings, segment_notes
from .pitch import track_pitch


def transcribe(
    path: str | Path,
    settings: SegmentSettings | None = None,
    model: str = "full",
) -> list[Note]:
    audio = load_audio(path)
    track = track_pitch(audio, model=model)
    return segment_notes(track, settings)
