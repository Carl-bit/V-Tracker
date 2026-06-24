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
import torch
from ultralytics import YOLO
from engine.video import get_video_meta, read_frames

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/ball_best.pt")
ap.add_argument("--video", default=None, help="ruta .mp4; default = primer mp4 bajo data/")
ap.add_argument("--start", type=float, default=30.0)
ap.add_argument("--end", type=float, default=90.0)
ap.add_argument("--sample", type=int, default=10)
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--dump", default=None, help="dir: guarda cada frame con deteccion + caja/conf para revisar FPs")
ap.add_argument("--dump-min-conf", type=float, default=0.0, help="solo volcar detecciones >= este conf")
args = ap.parse_args()

video = Path(args.video) if args.video else sorted(Path("data").rglob("*.mp4"))[0]
meta = get_video_meta(video)
end = args.end if args.end is not None else meta["duration_seconds"]

# FP16 roto en MIOpen (ROCm gfx1201): da 0 cajas. half solo en CUDA NVIDIA.
is_rocm = getattr(torch.version, "hip", None) is not None
use_half = torch.cuda.is_available() and not is_rocm
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"video: {video} | rango {args.start}-{end}s | sample 1/{args.sample} | conf>={args.conf} | imgsz={args.imgsz} | half={use_half}")

m = YOLO(args.model)
m.to(device)

dump_dir = manifest_fh = None
if args.dump:
    import cv2
    import json
    dump_dir = Path(args.dump)
    (dump_dir / "clean").mkdir(parents=True, exist_ok=True)  # frame sin caja (para dataset)
    manifest_fh = (dump_dir / "manifest.jsonl").open("w", encoding="utf-8")
    print(f"dump: frames con deteccion -> {dump_dir} (anotado + clean/ + manifest.jsonl)")

total = with_ball = dumped = 0
confs = []
t0 = time.time()
for idx, ts, frame in read_frames(video, sample_every_n=args.sample):
    if ts < args.start:
        continue
    if ts > end:
        break
    total += 1
    r = m.predict(frame, imgsz=args.imgsz, conf=args.conf, half=use_half, verbose=False)[0]
    if len(r.boxes):
        with_ball += 1
        confs.extend(float(c) for c in r.boxes.conf.tolist())
        if dump_dir is not None:
            keep = [(b, c) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())
                    if c >= args.dump_min_conf]
            if keep:
                h_px, w_px = frame.shape[:2]
                cmax = max(c for _, c in keep)
                name = f"c{cmax:.2f}_t{ts:07.1f}_n{len(keep)}"
                cv2.imwrite(str(dump_dir / "clean" / f"{name}.jpg"), frame)  # limpio
                vis = frame.copy()
                for (x1, y1, x2, y2), c in keep:
                    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                    cv2.rectangle(vis, p1, p2, (0, 255, 255), 2)
                    cv2.putText(vis, f"{c:.2f}", (p1[0], max(0, p1[1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imwrite(str(dump_dir / f"{name}.jpg"), vis)  # anotado (browse rapido)
                manifest_fh.write(json.dumps({
                    "name": name, "t": round(ts, 2), "w": w_px, "h": h_px,
                    "boxes": [[round(x, 1) for x in b] + [round(c, 3)] for b, c in keep],
                }) + "\n")
                dumped += 1
    if total % 50 == 0:
        print(f"  {total} frames | balon en {with_ball} | {total/(time.time()-t0):.1f} fps", file=sys.stderr)

if total == 0:
    sys.exit("rango sin frames")
recall = 100.0 * with_ball / total
print(f"\nframes: {total} | con balon: {with_ball} | recall aprox: {recall:.1f}%")
if confs:
    print(f"conf prom/max: {sum(confs)/len(confs):.2f} / {max(confs):.2f}")
if manifest_fh is not None:
    manifest_fh.close()
    print(f"dump: {dumped} frames -> {dump_dir}")
print(f"baseline COCO P1.3: ~16%  | umbral: 60%")
