"""Deteccion YOLO26 sobre frames BGR. Sin tracking, sin normalizar (eso va en export).

Standalone: sin FastAPI, sin ARQ.
Regla 1 (CLAUDE.md): el balon con COCO out-of-the-box probablemente NO alcanza.
Esta clase es la base para medirlo (P1.3), no la solucion final.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# COCO ids -> nombre interno
RELEVANT_CLASSES = {0: "persona", 32: "balon"}


class Detector:
    """Carga YOLO26 una vez y detecta persona + balon por frame.

    model: 'yolo26n.pt' o 'yolo26s.pt'. Se guarda/descarga en models/.
    half=True solo aplica en GPU; en CPU cae con warning (regla 6).
    """

    def __init__(
        self,
        model: str = "yolo26n.pt",
        imgsz: int = 640,
        conf: float = 0.25,
    ) -> None:
        import torch
        from ultralytics import YOLO

        self.imgsz = imgsz
        self.conf = conf
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.half = self.device != "cpu"
        if not self.half:
            logger.warning("Sin GPU: inferencia en CPU sin half. Sera lento.")
        # path bajo models/ -> ultralytics descarga ahi si no existe
        self.model = YOLO(f"models/{model}")
        self.model.to(self.device)

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """1 frame BGR -> [{cls, conf, xyxy}] en pixeles, solo persona/balon."""
        results = self.model.predict(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            half=self.half,
            classes=list(RELEVANT_CLASSES),
            verbose=False,
        )
        boxes = results[0].boxes
        return [
            {
                "cls": RELEVANT_CLASSES[int(cls_id)],
                "conf": float(conf),
                "xyxy": tuple(float(v) for v in xyxy),
            }
            for cls_id, conf, xyxy in zip(
                boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
            )
        ]
