"""DoD P1.1: imprime meta + cuenta de frames muestreados de un .mp4.

Uso: python -m tests.check_video [path_video] [sample_every_n]
Default: primer .mp4 encontrado en data/, N=10.
"""

import sys
from pathlib import Path

from engine.video import get_video_meta, read_frames


def main() -> None:
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
    else:
        candidates = sorted(Path("data").rglob("*.mp4"))
        if not candidates:
            sys.exit("No hay .mp4 en data/")
        video = candidates[0]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print(f"video: {video}")
    meta = get_video_meta(video)
    for k, v in meta.items():
        print(f"  {k}: {v}")

    count = 0
    last_ts = 0.0
    for frame_idx, ts, frame in read_frames(video, sample_every_n=n):
        count += 1
        last_ts = ts
        if count == 1:
            print(f"  primer frame: idx={frame_idx} ts={ts:.3f}s shape={frame.shape}")
    print(f"frames muestreados (1/{n}): {count}")
    print(f"ultimo timestamp: {last_ts:.2f}s de {meta['duration_seconds']:.2f}s")


if __name__ == "__main__":
    main()
