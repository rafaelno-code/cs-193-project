from pathlib import Path

import pretty_midi

from .notes import Note


def notes_to_midi(
    notes: list[Note],
    path: str | Path,
    bpm: float = 120.0,
    program: int = 0,
    trim_leading_silence: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    offset = notes[0].start if (notes and trim_leading_silence) else 0.0

    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    track = pretty_midi.Instrument(program=program, name="Hummed melody")
    for n in notes:
        track.notes.append(
            pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start - offset, end=n.end - offset)
        )
    midi.instruments.append(track)
    midi.write(str(path))
    return path
