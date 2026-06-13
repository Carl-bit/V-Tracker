"""Recall del especialista de balon SOLO (sin COCO en el proceso).

Workaround ROCm: cargar 2 modelos YOLO en un proceso corrompe la inferencia en
gfx1201. Aca solo el ball model. Mide recall aproximado = % de frames con >=1
balon en un rango de juego activo. Compara vs baseline COCO 16% (P1.3).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from engine.video import get_video_meta, read_frames

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/ball_best.pt")
ap.add_argument("--start", type=float, default=30.0)
ap.add_argument("--end", type=float, default=90.0)
ap.add_argument("--sample", type=int, default=10)
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--imgsz", type=int, default=1280)
args = ap.parse_args()

video = sorted(Path("data").rglob("*.mp4"))[0]
meta = get_video_meta(video)
end = args.end if args.end is not None else meta["duration_seconds"]
print(f"video: {video} | rango {args.start}-{end}s | sample 1/{args.sample} | conf>={args.conf} | imgsz={args.imgsz}")

m = YOLO(args.model)
m.to("cuda:0")

total = with_ball = 0
confs = []
t0 = time.time()
for idx, ts, frame in read_frames(video, sample_every_n=args.sample):
    if ts < args.start:
        continue
    if ts > end:
        break
    total += 1
    r = m.predict(frame, imgsz=args.imgsz, conf=args.conf, half=True, verbose=False)[0]
    if len(r.boxes):
        with_ball += 1
        confs.extend(float(c) for c in r.boxes.conf.tolist())
    if total % 50 == 0:
        print(f"  {total} frames | balon en {with_ball} | {total/(time.time()-t0):.1f} fps", file=sys.stderr)

if total == 0:
    sys.exit("rango sin frames")
recall = 100.0 * with_ball / total
print(f"\nframes: {total} | con balon: {with_ball} | recall aprox: {recall:.1f}%")
if confs:
    print(f"conf prom/max: {sum(confs)/len(confs):.2f} / {max(confs):.2f}")
print(f"baseline COCO P1.3: ~16%  | umbral: 60%")
