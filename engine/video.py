"""Lectura y muestreo de frames de un .mp4 (regla 7: 1/N, no 30fps completos).

Standalone: sin FastAPI, sin ARQ, sin YOLO.
"""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class VideoOpenError(Exception):
    """No se pudo abrir el video (path malo, codec no soportado, archivo corrupto)."""


def _open_capture(path: str | Path) -> cv2.VideoCapture:
    path = Path(path)
    if not path.is_file():
        raise VideoOpenError(f"Archivo no existe: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoOpenError(f"OpenCV no pudo abrir el video: {path}")
    return cap


def get_video_meta(path: str | Path) -> dict:
    """Metadata del video: {duration_seconds, fps, width, height}."""
    cap = _open_capture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0:
            raise VideoOpenError(f"FPS invalido ({fps}) en: {path}")
        return {
            "duration_seconds": frame_count / fps,
            "fps": fps,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def read_frames(
    path: str | Path, sample_every_n: int = 10
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yielda (frame_idx, timestamp_seg, frame_bgr) muestreando 1 de cada N.

    frame_idx = indice en el video original (no en la secuencia muestreada).
    timestamp_seg = frame_idx / fps del original.
    """
    if sample_every_n < 1:
        raise ValueError(f"sample_every_n debe ser >= 1, vino {sample_every_n}")
    cap = _open_capture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise VideoOpenError(f"FPS invalido ({fps}) en: {path}")
        frame_idx = 0
        while True:
            # grab() decodifica menos que read(); retrieve() solo en los muestreados
            if not cap.grab():
                break
            if frame_idx % sample_every_n == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield frame_idx, frame_idx / fps, frame
            frame_idx += 1
    finally:
        cap.release()
