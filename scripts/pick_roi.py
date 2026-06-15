"""Marca el poligono de cancha (ROI) clickeando sobre un frame. Camara fija.

Abre un frame del video y deja clickear los vertices de la ZONA DE JUEGO. Guarda
{"polygon": [[x_norm,y_norm],...]} para usar con engine.run --roi roi.json.

Dibujar GENEROSO: cancha + margen + el aire por encima de la red (el balon sube
alto). EXCLUIR la banda lateral / tribuna donde descansa el balon extra. Un balon
jugado que cae apenas fuera de las lineas debe quedar DENTRO; solo banda afuera.
Tipico: 4-6 puntos en sentido horario empezando arriba-izquierda.

Uso:
  python -m scripts.pick_roi --video "data\\sample\\VODME VS VEC A3.mp4" --at 700 --out roi.json

Controles:
  click izq = agrega vertice | u = deshace ultimo | r = reinicia
  Enter/s   = guarda y sale  | Esc/q = cancela sin guardar
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(prog="scripts.pick_roi")
    ap.add_argument("--video", required=True, help="el .mp4 (mismo que procesa el motor)")
    ap.add_argument("--at", type=float, default=700.0, help="segundo del frame a mostrar")
    ap.add_argument("--out", default="roi.json", help="json de salida")
    ap.add_argument("--max-w", type=int, default=1400, help="ancho max de la ventana")
    args = ap.parse_args()

    import cv2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"OpenCV no pudo abrir {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.at * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"no pude leer un frame en t={args.at}s")

    h, w = frame.shape[:2]
    scale = min(1.0, args.max_w / w)
    disp_w, disp_h = int(w * scale), int(h * scale)
    base = cv2.resize(frame, (disp_w, disp_h))

    pts: list[tuple[int, int]] = []  # en px de la imagen mostrada

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    win = "ROI cancha - click vertices | u undo | r reset | Enter guarda | Esc cancela"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        img = base.copy()
        for i, p in enumerate(pts):
            cv2.circle(img, p, 4, (0, 0, 255), -1)
            if i > 0:
                cv2.line(img, pts[i - 1], p, (0, 255, 255), 2)
        if len(pts) >= 3:
            cv2.line(img, pts[-1], pts[0], (0, 255, 0), 1)  # cierre tentativo
        cv2.putText(img, f"{len(pts)} vertices", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(win, img)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, ord("s")):       # Enter / s
            break
        if k in (27, ord("q")):       # Esc / q
            cv2.destroyAllWindows()
            raise SystemExit("cancelado, no se guardo nada")
        if k == ord("u") and pts:
            pts.pop()
        if k == ord("r"):
            pts.clear()

    cv2.destroyAllWindows()
    if len(pts) < 3:
        raise SystemExit(f"poligono necesita >=3 vertices, hay {len(pts)}")

    polygon = [[round(x / disp_w, 5), round(y / disp_h, 5)] for x, y in pts]
    Path(args.out).write_text(json.dumps({"polygon": polygon}, indent=2), encoding="utf-8")
    print(f"OK {len(polygon)} vertices -> {args.out}")
    print("usar: .\\cpu.cmd -m engine.run ... --roi " + args.out)


if __name__ == "__main__":
    main()
