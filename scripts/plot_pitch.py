import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hum_to_midi.audio import load_audio
from hum_to_midi.pitch import track_pitch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="WAV file of a hummed melody")
    parser.add_argument("--model", default="full", choices=["full", "tiny"])
    parser.add_argument("-o", "--output", type=Path, help="where to save the PNG")
    args = parser.parse_args()

    track = track_pitch(load_audio(args.input), model=args.model)
    midi = 69 + 12 * np.log2(track.frequency / 440.0)

    fig, ax = plt.subplots(figsize=(12, 4))
    points = ax.scatter(track.times, midi, c=track.confidence, s=4, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(points, ax=ax, label="confidence")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("MIDI pitch")
    ax.set_title(f"Pitch curve: {args.input.name}")
    ax.grid(alpha=0.3)

    out = args.output or Path("output") / f"{args.input.stem}_pitch.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"{len(track.times)} frames, {track.times[-1]:.2f}s -> saved {out}")


if __name__ == "__main__":
    main()
