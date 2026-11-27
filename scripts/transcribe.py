import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pretty_midi

from hum_to_midi.pipeline import transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="WAV file of a hummed melody")
    parser.add_argument("--model", default="full", choices=["full", "tiny"])
    args = parser.parse_args()

    notes = transcribe(args.input, model=args.model)

    print(f"{'note':>5}  {'start':>6}  {'end':>6}  {'vel':>3}")
    for n in notes:
        name = pretty_midi.note_number_to_name(n.pitch)
        print(f"{name:>5}  {n.start:6.2f}  {n.end:6.2f}  {n.velocity:3d}")
    print(f"{len(notes)} notes")


if __name__ == "__main__":
    main()
