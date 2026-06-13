"""P1.3 checkpoint: medir deteccion del balon de YOLO26 COCO sobre video real.

Solo mide y reporta. No implementa TrackNet ni tiling (ver INVESTIGACION.md sec 2).

Sin ground truth, "recall" se aproxima: detection_rate sobre un rango --start/--end
de juego activo continuo (balon visible casi siempre) ~ recall. Sobre el video
completo (highlights con replays/graficos) es solo cota inferior burda.

Uso:
  python -m tests.validate_ball                          # video completo
  python -m tests.validate_ball --start 30 --end 90      # rango de juego activo
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

from engine.detect import Detector
from engine.video import get_video_meta, read_frames

RECALL_MIN = 60.0  # % bajo esto -> fine-tune o TrackNet

FRAMES_DIR = Path("data/validate_ball_frames")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--start", type=float, default=0.0, help="segundo inicial")
    p.add_argument("--end", type=float, default=None, help="segundo final")
    p.add_argument("--sample", type=int, default=10, help="1 de cada N frames")
    p.add_argument("--conf", type=float, default=0.15, help="umbral conf (medir, no filtrar)")
    p.add_argument("--model", default="yolo26n.pt")
    p.add_argument("--ball-model", default=None,
                   help="pesos especialista de balon (P1.3b); si se pasa, balon sale de ahi")
    p.add_argument("--ball-conf", type=float, default=0.25, help="umbral conf del especialista")
    p.add_argument("--save-frames", type=int, default=8, help="K frames anotados a disco")
    return p.parse_args()


def annotate(frame, detections):
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d["xyxy"])
        color = (0, 0, 255) if d["cls"] == "balon" else (0, 200, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{d['cls']} {d['conf']:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def main() -> None:
    args = parse_args()
    video = args.video
    if video is None:
        candidates = sorted(Path("data").rglob("*.mp4"))
        if not candidates:
            sys.exit("No hay .mp4 en data/")
        video = candidates[0]

    meta = get_video_meta(video)
    end = args.end if args.end is not None else meta["duration_seconds"]
    print(f"video: {video}")
    print(f"rango: {args.start:.1f}s - {end:.1f}s | sample 1/{args.sample} | conf >= {args.conf}")

    det = Detector(model=args.model, conf=args.conf,
                   ball_model=args.ball_model, ball_conf=args.ball_conf)
    print(f"device={det.device} half={det.half} imgsz={det.imgsz} model={args.model}")
    if args.ball_model:
        print(f"especialista balon: {args.ball_model} conf>={args.ball_conf} imgsz={det.ball_imgsz}")

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    with_ball = 0
    confs: list[float] = []
    gap_start = None  # ts donde empezo la racha actual sin balon
    max_gap = 0.0
    saved = 0
    t0 = time.time()

    for frame_idx, ts, frame in read_frames(video, sample_every_n=args.sample):
        if ts < args.start:
            continue
        if ts > end:
            break
        total += 1
        detections = det.detect(frame)
        balls = [d for d in detections if d["cls"] == "balon"]
        if balls:
            with_ball += 1
            confs.extend(d["conf"] for d in balls)
            if gap_start is not None:
                max_gap = max(max_gap, ts - gap_start)
                gap_start = None
            if saved < args.save_frames:
                cv2.imwrite(str(FRAMES_DIR / f"ball_{ts:07.1f}s.jpg"), annotate(frame, detections))
                saved += 1
        elif gap_start is None:
            gap_start = ts
        if total % 50 == 0:
            rate = total / (time.time() - t0)
            print(f"  {total} frames | balon en {with_ball} | {rate:.1f} fps", file=sys.stderr)

    if gap_start is not None:
        max_gap = max(max_gap, end - gap_start)
    if total == 0:
        sys.exit("Rango sin frames muestreados")

    recall = 100.0 * with_ball / total
    print()
    print(f"frames muestreados:     {total}")
    print(f"frames con balon:       {with_ball}")
    print(f"recall aproximado:      {recall:.1f}%")
    if confs:
        print(f"conf balon prom/max:    {sum(confs) / len(confs):.3f} / {max(confs):.3f}")
    print(f"racha max sin balon:    {max_gap:.1f}s")
    print(f"frames anotados en:     {FRAMES_DIR}/")
    print()
    if recall >= RECALL_MIN:
        print(f"VEREDICTO: recall >= {RECALL_MIN:.0f}%. OK para MVP, seguir a P1.4.")
    else:
        print(
            f"VEREDICTO: recall < {RECALL_MIN:.0f}%. COCO no alcanza (regla 1). "
            "-> P1.3b: fine-tune YOLO26 con dataset voleibol (Roboflow) o TrackNet. "
            "Ver INVESTIGACION.md sec 2. NO seguir a P1.4 sin decidir."
        )


if __name__ == "__main__":
    main()
