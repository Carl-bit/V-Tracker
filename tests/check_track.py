"""DoD track.py: trayectoria de balon continua (con huecos interpolados) +
tracks de jugadores con id, sobre el .mp4 de prueba.

Uso: python -m tests.check_track [path_video] [sample_every_n] [start_seg] [end_seg]
Default: primer .mp4 en data/, 1 de cada 3 frames, ventana 10s-20s (rally con balon).
Balon disperso => muestrear denso sobre una ventana de juego real, no 1/10.
"""

import sys
from pathlib import Path

from engine.detect import Detector
from engine.track import BallTracker, PlayerTracker
from engine.video import get_video_meta, read_frames


def main() -> None:
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
    else:
        candidates = sorted(Path("data").rglob("*.mp4"))
        if not candidates:
            sys.exit("No hay .mp4 en data/")
        video = candidates[0]
    sample_every_n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    start_seg = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    end_seg = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0

    meta = get_video_meta(video)
    print(f"video={video.name} fps={meta['fps']:.1f} {meta['width']}x{meta['height']}")

    det = Detector(ball_model="ball_best.pt")
    print(f"device={det.device} half={det.half} especialista_balon=ON")

    players = PlayerTracker()
    ball = BallTracker()
    seen_ids: set[int] = set()
    frames = 0
    ball_hits = 0
    for frame_idx, ts, frame in read_frames(video, sample_every_n=sample_every_n):
        if ts < start_seg:
            continue
        if ts > end_seg:
            break
        dets = det.detect(frame)
        for box in players.update(dets, frame_idx, ts):
            seen_ids.add(box["id"])
        ball.update(dets, frame_idx, ts)
        if any(d["cls"] == "balon" for d in dets):
            ball_hits += 1
        frames += 1

    traj = ball.trajectory()
    real = [p for p in traj if not p["interpolated"]]
    interp = [p for p in traj if p["interpolated"]]

    print(f"\nframes procesados: {frames}")
    print(f"jugadores: {len(seen_ids)} ids distintos -> {sorted(seen_ids)}")
    print(f"balon detectado en {ball_hits}/{frames} frames")
    print(f"trayectoria balon: {len(traj)} puntos ({len(real)} reales + {len(interp)} interpolados)")
    print("\nprimeros 12 puntos de trayectoria (px):")
    for p in traj[:12]:
        tag = "interp" if p["interpolated"] else "real  "
        print(f"  t={p['timestamp']:5.2f}s [{tag}] x={p['x']:7.1f} y={p['y']:7.1f}")


if __name__ == "__main__":
    main()
