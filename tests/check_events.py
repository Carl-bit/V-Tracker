"""DoD events.py: sobre el .mp4 de prueba devuelve al menos saque + remate
con timestamp plausible.

Pipeline: video -> detect (especialista balon) -> BallTracker.trajectory() ->
detect_events. Mismo muestreo denso sobre ventana de rally que check_track.

Uso: python -m tests.check_events [path_video] [sample_every_n] [start_seg] [end_seg]
Default: primer .mp4 en data/, 1 de cada 3 frames, ventana 25s-40s (rally con saque).
"""

import sys
from pathlib import Path

from engine.detect import Detector
from engine.events import detect_events
from engine.track import BallTracker
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
    start_seg = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
    end_seg = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0

    meta = get_video_meta(video)
    print(f"video={video.name} fps={meta['fps']:.1f} {meta['width']}x{meta['height']}")

    det = Detector(ball_model="ball_best.pt")
    print(f"device={det.device} half={det.half} especialista_balon=ON")

    ball = BallTracker()
    for frame_idx, ts, frame in read_frames(video, sample_every_n=sample_every_n):
        if ts < start_seg:
            continue
        if ts > end_seg:
            break
        ball.update(det.detect(frame), frame_idx, ts)

    traj = ball.trajectory()
    events = detect_events(traj, meta)

    print(f"\ntrayectoria: {len(traj)} puntos")
    print(f"eventos: {len(events)}")
    for e in events:
        print(
            f"  {e.event_id} t={e.timestamp:5.2f}s {e.type:<9} {e.team} "
            f"conf={e.confidence:.2f} xy=({e.x:.0f},{e.y:.0f})"
        )

    types = {e.type for e in events}
    ok = "saque" in types and "remate" in types
    print(f"\nDoD (saque + remate presentes): {'OK' if ok else 'FALLA'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
