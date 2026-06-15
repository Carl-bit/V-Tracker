"""Iteracion 2: extrae frames DENSOS dentro de rallies + pre-etiqueta (P1.3c).

El muestreo uniforme cada 10s agarro muchos frames muertos (timeout) -> pocos
positivos -> recall bajo. Aca:
  Pase A (scan grueso): corre ball_frontal por todo el video, marca donde HAY balon.
  -> arma "rallies" (intervalos de juego) uniendo detecciones cercanas.
  Pase B (denso): muestrea frames dentro de los rallies, repartidos por TODO el
  partido (variedad de luz/posicion), y los pre-etiqueta con el mismo modelo.

Frames sin balon dentro del rally (el modelo lo perdio) se guardan SIN label: son
justo los casos dificiles que suben el recall al anotarlos a mano.

NO pisa lo ya anotado: si la imagen ya existe, la saltea (no toca su label).

Uso:
  python -m scripts.extract_rally_frames "data/sample/VODME VS VEC A3.mp4" "data/sample/VODME A3.mp4" --max 300
"""

import argparse
import re
from pathlib import Path


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")


def _rallies(times: list[float], merge_gap: float, pad: float) -> list[tuple[float, float]]:
    if not times:
        return []
    times.sort()
    ivs = [[times[0] - pad, times[0] + pad]]
    for t in times[1:]:
        if t - ivs[-1][1] <= merge_gap:
            ivs[-1][1] = t + pad
        else:
            ivs.append([t - pad, t + pad])
    return [(a, b) for a, b in ivs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="data/dataset/vly_frontal/images")
    ap.add_argument("--ball-model", default="ball_frontal.pt")
    ap.add_argument("--scan-every", type=float, default=2.0, help="seg entre frames del scan grueso")
    ap.add_argument("--dense-every", type=float, default=0.5, help="seg entre frames dentro de rally")
    ap.add_argument("--pad", type=float, default=1.5, help="seg de margen alrededor de cada deteccion")
    ap.add_argument("--merge-gap", type=float, default=3.0, help="seg para unir detecciones en un rally")
    ap.add_argument("--max", type=int, default=300, help="frames a guardar por video")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--sky-frac", type=float, default=0.12)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, default=None)
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    from engine.video import get_video_meta, read_frames

    out = Path(args.out)
    ds = out.parent
    (ds / "labels").mkdir(parents=True, exist_ok=True)
    (ds / "_preview").mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(f"models/{args.ball_model}")

    def detect(frame):
        b = model.predict(frame, imgsz=args.imgsz, conf=args.conf, half=False, verbose=False)[0].boxes
        return [(xywhn, xyxy, cf) for xywhn, xyxy, cf in
                zip(b.xywhn.tolist(), b.xyxy.tolist(), b.conf.tolist())
                if args.sky_frac <= 0 or xywhn[1] >= args.sky_frac]

    grand_total = 0
    for video in args.videos:
        video = Path(video)
        meta = get_video_meta(video)
        fps = meta["fps"]
        stem = _slug(video.stem)

        # --- Pase A: scan grueso, juntar timestamps con balon ---
        det_ts = []
        scan_step = max(1, int(fps * args.scan_every))
        end_sec = args.end_sec if args.end_sec is not None else meta["duration_seconds"]
        for _, ts, frame in read_frames(video, scan_step):
            if ts < args.start_sec:
                continue
            if ts > end_sec:
                break
            if detect(frame):
                det_ts.append(ts)
        ivs = _rallies(det_ts, args.merge_gap, args.pad)
        play_s = sum(b - a for a, b in ivs)
        print(f"{video.name}: {len(det_ts)} detecciones scan -> {len(ivs)} rallies (~{play_s:.0f}s de juego)")

        # --- candidatos densos dentro de rallies, repartidos ---
        cands = []
        dense_step = max(1, int(fps * args.dense_every))
        for iv_a, iv_b in ivs:
            t = iv_a
            while t <= iv_b:
                cands.append(int(round(t * fps)))
                t += args.dense_every
        # clamp a la ventana: el pad de un rally puede cruzar start/end-sec
        lo = int(args.start_sec * fps)
        hi = int(end_sec * fps)
        cands = sorted({c for c in cands if lo <= c <= hi})
        if len(cands) > args.max:  # submuestreo uniforme para cubrir todo el partido
            step = len(cands) / args.max
            cands = [cands[int(i * step)] for i in range(args.max)]

        # --- Pase B: decodificar esos frames, guardar + pre-etiquetar ---
        cap = cv2.VideoCapture(str(video))
        saved = with_label = 0
        for idx in cands:
            name = f"{stem}_{idx:07d}"
            img_path = out / f"{name}.jpg"
            if img_path.exists():
                continue  # ya estaba (no pisar anotacion previa)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imwrite(str(img_path), frame)
            saved += 1
            kept = detect(frame)
            if not kept:
                continue  # miss dentro del rally: queda sin label para anotar a mano
            lines = [f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for (xc, yc, w, h), _, _ in kept]
            (ds / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            prev = frame.copy()
            for _, (x1, y1, x2, y2), cf in kept:
                cv2.rectangle(prev, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(prev, f"balon {cf:.2f}", (int(x1), max(0, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(ds / "_preview" / f"{name}.jpg"), prev)
            with_label += 1
        cap.release()
        print(f"  guardados {saved} frames nuevos ({with_label} con pre-label, {saved-with_label} a anotar)")
        grand_total += saved

    print(f"\nTOTAL nuevos: {grand_total} -> {out}")
    print("Anotar en labelImg, luego: split_dataset + train_ball")


if __name__ == "__main__":
    main()
