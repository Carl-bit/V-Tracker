"""Deteccion de cortes de camara (cambios de escena) por histograma de color.

Standalone: sin FastAPI, sin ARQ, sin YOLO. Solo opencv+numpy (regla 9, cero dep nueva).

Un corte DURO (cambio de toma, cambio de lado entre sets) cambia el histograma de
color de golpe. Un paneo o el movimiento de jugadores/balon NO: el color global del
frame se conserva. Se mide la correlacion del histograma HSV (H,S) entre frames
muestreados consecutivos; si cae por debajo del umbral -> frontera de escena.

Las fronteras (timestamps donde arranca un shot nuevo) sirven para:
  - resetear el estado del BallTracker por shot (balon estatico, continuidad),
  - cortar segmentos en detect_events (no stitchear trayectorias de dos tomas).

Se alimenta en la pasada gruesa de engine.run, que ya recorre esos frames en orden:
cero pasada de video extra.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Bins H,S del histograma HSV. Grueso a proposito: robusto a jitter/iluminacion.
_H_BINS, _S_BINS = 50, 60


def _hist(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [_H_BINS, _S_BINS], [0, 180, 0, 256])
    cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
    return h


class SceneCutDetector:
    """Detecta cortes de camara comparando histogramas de frames consecutivos.

    corr_threshold: correlacion HSV bajo la cual se declara corte. Alto (~0.6) marca
    SOLO cortes duros; paneos y accion intra-shot quedan por encima y no cortan
    (el paneo lo absorbe el filtro de balon estatico del tracker, no es un corte).
    """

    def __init__(self, corr_threshold: float = 0.6) -> None:
        self.corr_threshold = corr_threshold
        self._prev: np.ndarray | None = None
        self._cuts: list[float] = []

    def update(self, frame_bgr: np.ndarray, ts: float) -> None:
        h = _hist(frame_bgr)
        if self._prev is not None:
            corr = cv2.compareHist(self._prev, h, cv2.HISTCMP_CORREL)
            if corr < self.corr_threshold:
                self._cuts.append(float(ts))
        self._prev = h

    def cuts(self) -> list[float]:
        """Timestamps (s) donde arranca un shot nuevo. Orden ascendente."""
        return list(self._cuts)
