"""DoD P1.2: corre Detector sobre 1 frame de data/ y lista detecciones con conf.

Uso: python -m tests.check_detect [path_video] [timestamp_aprox_seg]
Default: primer .mp4 en data/, frame en el segundo 30 (evita intro sin juego).
"""

import sys
from pathlib import Path

from engine.detect import Detector
from engine.video import get_video_meta, read_frames


def main() -> None:
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
    else:
        candidates = sorted(Path("data").rglob("*.mp4"))
        if not candidates:
            sys.exit("No hay .mp4 en data/")
        video = candidates[0]
    target_seg = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

    meta = get_video_meta(video)
    target_idx = int(target_seg * meta["fps"])
    frame = None
    for frame_idx, ts, f in read_frames(video, sample_every_n=1):
        if frame_idx >= target_idx:
            frame = f
            print(f"frame: idx={frame_idx} ts={ts:.2f}s shape={f.shape}")
            break
    if frame is None:
        sys.exit("Video mas corto que el timestamp pedido")

    det = Detector()
    print(f"device={det.device} half={det.half} imgsz={det.imgsz}")
    detections = det.detect(frame)
    print(f"detecciones: {len(detections)}")
    for d in detections:
        x1, y1, x2, y2 = d["xyxy"]
        print(
            f"  {d['cls']:<8} conf={d['conf']:.3f} "
            f"xyxy=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
        )


if __name__ == "__main__":
    main()
