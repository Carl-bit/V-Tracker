"""Pre-etiqueta el balon en frames extraidos: propone cajas YOLO para CORREGIR.

P1.3c. Corre el especialista actual (ball_best.pt) sobre cada imagen y escribe un
label YOLO (clase 0 = balon) por cada deteccion sobre el umbral. En el dominio
frontal con balon amarillo, ball_best acierta con alta confianza pero recall bajo:
=> las propuestas son confiables (pocos falsos positivos), el humano agrega las que
faltan. Frames sin deteccion quedan SIN label (el anotador los marca a mano).

Salida (formato YOLOv8, listo para Roboflow/CVAT/LabelImg):
  <ds>/images/*.jpg        (ya estan)
  <ds>/labels/*.txt        (clase 0 + xc yc w h normalizado)
  <ds>/data.yaml           (nc=1, names=[balon])
  <ds>/classes.txt

Uso:
  python -m scripts.prelabel_ball
  python -m scripts.prelabel_ball --ds data/dataset/vly_frontal --conf 0.25
"""

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="data/dataset/vly_frontal", help="dir del dataset")
    ap.add_argument("--ball-model", default="ball_best.pt")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="umbral de propuesta (alto=menos FP, mas trabajo manual)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--sky-frac", type=float, default=0.0,
                    help="descarta propuestas cuyo centro este en la franja superior "
                         "(y_centro < sky-frac): mata FPs de luces/techo. 0=off")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    ds = Path(args.ds)
    images = sorted((ds / "images").glob("*.jpg"))
    if not images:
        raise SystemExit(f"No hay imagenes en {ds/'images'} (corre extract_frames primero)")
    labels_dir = ds / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = ds / "_preview"  # frames con cajas dibujadas (para ojo humano, NO se anota aca)
    preview_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(f"models/{args.ball_model}")
    print(f"pre-etiquetando {len(images)} imagenes (conf>={args.conf})...", flush=True)
    with_label = 0
    boxes_total = 0
    for i, img in enumerate(images, 1):
        frame = cv2.imread(str(img))  # array BGR (pasar path crashea predict en ROCm)
        if frame is None:
            continue
        b = model.predict(frame, imgsz=args.imgsz, conf=args.conf, half=False, verbose=False)[0].boxes
        if i % 50 == 0:
            print(f"  {i}/{len(images)}", flush=True)
        # filtro de cielo: descarta cajas con centro en la franja superior (luces/techo)
        kept = [(xywhn, xyxy, cf) for xywhn, xyxy, cf in
                zip(b.xywhn.tolist(), b.xyxy.tolist(), b.conf.tolist())
                if args.sky_frac <= 0 or xywhn[1] >= args.sky_frac]
        if not kept:
            continue  # sin propuesta: el anotador lo etiqueta a mano
        lines = [f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for (xc, yc, w, h), _, _ in kept]
        (labels_dir / f"{img.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        # preview: caja roja + conf, para revisar la propuesta de un vistazo
        prev = frame.copy()
        for _, (x1, y1, x2, y2), cf in kept:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(prev, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(prev, f"balon {cf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(str(preview_dir / f"{img.stem}.jpg"), prev)
        with_label += 1
        boxes_total += len(kept)

    (ds / "classes.txt").write_text("balon\n", encoding="utf-8")
    (ds / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnc: 1\nnames: ['balon']\n", encoding="utf-8"
    )
    print(f"imagenes: {len(images)}")
    print(f"con pre-label: {with_label} ({boxes_total} cajas propuestas)")
    print(f"sin deteccion (anotar a mano): {len(images) - with_label}")
    print(f"\nListo en {ds}. Subir a Roboflow/CVAT, CORREGIR, exportar YOLOv8, y:")
    print("  (GPU) python scripts/train_ball.py --data <export>/data.yaml --device 0")


if __name__ == "__main__":
    main()
