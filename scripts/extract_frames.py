"""Extrae frames de videos para armar dataset de balon (dominio frontal/amateur).

P1.3c: los videos del equipo son vista FRONTAL (no panoramica) con balon amarillo
FIVB. El especialista actual (broadcast) no generaliza -> hay que re-entrenar.
Este script saca frames repartidos a lo largo del video (cobertura de la crudeza
real: distintos rallies, luces, posiciones) para anotar.

Salida: <out>/images/<videostem>_<frameidx>.jpg  (listo para pre-etiquetar y subir
a Roboflow/CVAT). NO parte de engine/.

Uso:
  python -m scripts.extract_frames "data/sample/VODME VS VEC A3.mp4" "data/sample/VODME A3.mp4"
  python -m scripts.extract_frames VID --every 5 --max 800        # densidad/tope
  python -m scripts.extract_frames VID --start-frac 0 --end-frac 0.5   # primer 50%
"""

import argparse
import re
from pathlib import Path

from engine.video import get_video_meta, read_frames


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="rutas .mp4")
    ap.add_argument("--out", default="data/dataset/vly_frontal/images")
    ap.add_argument("--every", type=float, default=5.0, help="segundos entre frames")
    ap.add_argument("--max", type=int, default=0, help="tope de frames por video (0=sin tope)")
    ap.add_argument("--start-frac", type=float, default=0.0, help="fraccion inicial del video")
    ap.add_argument("--end-frac", type=float, default=1.0, help="fraccion final del video")
    ap.add_argument("--start-sec", type=float, default=None, help="segundo inicial (pisa start-frac)")
    ap.add_argument("--end-sec", type=float, default=None, help="segundo final (pisa end-frac)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for video in args.videos:
        video = Path(video)
        meta = get_video_meta(video)
        dur = meta["duration_seconds"]
        t0 = args.start_sec if args.start_sec is not None else args.start_frac * dur
        t1 = args.end_sec if args.end_sec is not None else args.end_frac * dur
        step = max(1, int(meta["fps"] * args.every))
        stem = _slug(video.stem)
        n = 0
        for frame_idx, ts, frame in read_frames(video, sample_every_n=step):
            if ts < t0:
                continue
            if ts > t1:
                break
            import cv2

            cv2.imwrite(str(out / f"{stem}_{frame_idx:07d}.jpg"), frame)
            n += 1
            if args.max and n >= args.max:
                break
        print(f"{video.name}: {n} frames (cada {args.every}s, ventana {t0:.0f}-{t1:.0f}s)")
        total += n
    print(f"\nTOTAL: {total} frames -> {out}")
    print("Siguiente: python -m scripts.prelabel_ball")


if __name__ == "__main__":
    main()
