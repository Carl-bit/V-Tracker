"""Cachea la trayectoria del balon de una ventana a JSON (detect+track es caro en CPU).

Permite iterar engine.events sin re-detectar. NO es parte del producto, es utilidad
de desarrollo.

Uso: python -m scripts.dump_trajectory <out.json> [sample_every_n] [start_seg] [end_seg]
"""

import json
import sys
from pathlib import Path

from engine.detect import Detector
from engine.track import BallTracker
from engine.video import get_video_meta, read_frames


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/traj_cache.json")
    sample_every_n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    start_seg = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    end_seg = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0

    video = sorted(Path("data").rglob("*.mp4"))[0]
    meta = get_video_meta(video)
    det = Detector(ball_model="ball_best.pt")
    print(f"video={video.name} {meta['width']}x{meta['height']} device={det.device}")

    ball = BallTracker()
    n = 0
    for frame_idx, ts, frame in read_frames(video, sample_every_n=sample_every_n):
        if ts < start_seg:
            continue
        if ts > end_seg:
            break
        ball.update(det.detect(frame), frame_idx, ts)
        n += 1
        if n % 50 == 0:
            print(f"  ...{n} frames (ts={ts:.1f}s)")

    traj = ball.trajectory()
    out.write_text(json.dumps({"meta": meta, "trajectory": traj}), encoding="utf-8")
    real = sum(1 for p in traj if not p["interpolated"])
    print(f"guardado {out}: {len(traj)} puntos ({real} reales) de {n} frames")


if __name__ == "__main__":
    main()
