"""Pinta el resultado del motor sobre el video: balon + estela + eventos.

Util de DEBUG (fase 2), NO es el front (fase 4) ni parte de engine/. Lee lo que
el motor ya dejo en disco (out.json + out_ball_track.json), no re-infiere => sin
GPU, rapido. Sirve para juzgar con el ojo lo que la consola no dice: si la pelota
se sigue bien y si los eventos (saque/recepcion/armado/remate) caen donde deben.

No pinta cajas de jugador: el motor no las guarda por-frame (solo zonas promedio).

Uso:
  python -m scripts.overlay --video "data\\sample\\VODME VS VEC A3.mp4" \
      --json out.json --out overlay.mp4
  # acotar una ventana (recomendado: revisar tramos, no la hora entera):
  python -m scripts.overlay --video X.mp4 --json out.json --start 600 --end 660 --out v.mp4

Salida: un .mp4 con la pelota (circulo lleno=detectada, hueco=interpolada), su
estela de los ultimos --trail seg, y un banner cuando ocurre un evento.
"""

import argparse
import json
from pathlib import Path


# color BGR por tipo de evento (ASCII, sin depender de nada externo)
EVENT_COLOR = {
    "saque": (0, 215, 255),      # ambar
    "recepcion": (0, 255, 0),    # verde
    "armado": (255, 200, 0),     # cyan-azul
    "remate": (0, 0, 255),       # rojo
}


def _load_track(json_out: Path) -> list[dict]:
    """Trayectoria densa del balon desde <out>_ball_track.json (puntos norm)."""
    track_path = json_out.with_name(f"{json_out.stem}_ball_track.json")
    if not track_path.is_file():
        raise SystemExit(f"no encuentro {track_path} (lo genera engine.run junto al --out)")
    pts = json.loads(track_path.read_text(encoding="utf-8")).get("points", [])
    return sorted(pts, key=lambda p: p["timestamp"])


def _load_events(json_out: Path) -> list[dict]:
    """timeline_events desde el JSON principal."""
    if not json_out.is_file():
        raise SystemExit(f"no encuentro {json_out}")
    return json.loads(json_out.read_text(encoding="utf-8")).get("timeline_events", [])


def main() -> None:
    ap = argparse.ArgumentParser(prog="scripts.overlay")
    ap.add_argument("--video", required=True, help="el .mp4 original que proceso el motor")
    ap.add_argument("--json", default="out.json", help="JSON principal (de engine.run --out)")
    ap.add_argument("--out", default="overlay.mp4", help="video anotado de salida")
    ap.add_argument("--start", type=float, default=0.0, help="segundo inicial")
    ap.add_argument("--end", type=float, default=None, help="segundo final")
    ap.add_argument("--stride", type=int, default=1, help="1 de cada N frames (acelera/aligera)")
    ap.add_argument("--trail", type=float, default=1.0, help="seg de estela del balon")
    ap.add_argument("--event-hold", type=float, default=0.6, help="seg que el banner queda visible")
    ap.add_argument("--roi", default=None, help="dibuja el poligono de cancha (mismo json de engine.run)")
    args = ap.parse_args()

    import cv2

    roi_poly = None
    if args.roi:
        roi_poly = json.loads(Path(args.roi).read_text(encoding="utf-8"))["polygon"]

    json_out = Path(args.json)
    track = _load_track(json_out)
    events = _load_events(json_out)
    print(f"track: {len(track)} pts | eventos: {len(events)}")

    video = Path(args.video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"OpenCV no pudo abrir {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_fps = max(1.0, fps / args.stride)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
    if not writer.isOpened():
        raise SystemExit("VideoWriter no abrio (codec mp4v no disponible?)")

    ti = 0  # cursor sobre track (avanza monotono con el tiempo)
    frame_idx, written = 0, 0
    while True:
        if not cap.grab():
            break
        t = frame_idx / fps
        frame_idx += 1
        if t < args.start:
            continue
        if args.end is not None and t > args.end:
            break
        if (frame_idx - 1) % args.stride != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue

        # ROI de cancha (linea azul): lo que queda afuera se descarta para el balon
        if roi_poly is not None:
            import numpy as np
            poly_px = np.array([[int(x * w), int(y * h)] for x, y in roi_poly], dtype=np.int32)
            cv2.polylines(frame, [poly_px], True, (255, 120, 0), 2)

        # estela: puntos del balon en (t-trail, t]
        while ti < len(track) and track[ti]["timestamp"] < t - args.trail:
            ti += 1
        last_px = None
        cur = None
        for p in track[ti:]:
            if p["timestamp"] > t:
                break
            px = (int(p["x_norm"] * w), int(p["y_norm"] * h))
            if last_px is not None:
                cv2.line(frame, last_px, px, (0, 255, 255), 2)
            last_px = px
            cur = p
        # punto actual: lleno=detectado, hueco=interpolado
        if cur is not None:
            px = (int(cur["x_norm"] * w), int(cur["y_norm"] * h))
            if cur.get("interpolated"):
                cv2.circle(frame, px, 9, (0, 255, 255), 2)
            else:
                cv2.circle(frame, px, 7, (0, 255, 255), -1)
                cv2.circle(frame, px, 9, (0, 0, 0), 2)

        # eventos activos: banner + marca en la coord del balon
        active = [e for e in events if 0 <= t - e["timestamp"] <= args.event_hold]
        for k, e in enumerate(active):
            c = EVENT_COLOR.get(e["type"], (255, 255, 255))
            bc = e.get("ball_coordinates") or {}
            if "x_norm" in bc:
                ex, ey = int(bc["x_norm"] * w), int(bc["y_norm"] * h)
                cv2.drawMarker(frame, (ex, ey), c, cv2.MARKER_TILTED_CROSS, 28, 3)
            cv2.putText(frame, f"{e['type']} t={e['timestamp']:.1f}", (20, 70 + 34 * k),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)

        cv2.putText(frame, f"t={t:6.1f}s", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.write(frame)
        written += 1
        if written % 200 == 0:
            print(f"  {written} frames escritos (t={t:.1f}s)", flush=True)

    cap.release()
    writer.release()
    print(f"OK {written} frames -> {args.out}")


if __name__ == "__main__":
    main()
