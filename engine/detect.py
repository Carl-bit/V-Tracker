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
# Sin especialista: COCO da persona + balon. Con especialista: COCO solo persona.
PERSON_ONLY = {0: "persona"}


class Detector:
    """Carga YOLO26 una vez y detecta persona + balon por frame.

    model: 'yolo26n.pt' o 'yolo26s.pt'. Se guarda/descarga en models/.
    half=True solo aplica en GPU; en CPU cae con warning (regla 6).

    ball_model: pesos del especialista de balon (P1.3b, fine-tune). Si se pasa,
    el balon sale de una 2da pasada con ese modelo (su clase 0 -> "balon") y se
    ignora el balon de COCO (clase 32, recall ~16%). Drop-in para TrackNet (fase B).
    """

    def __init__(
        self,
        model: str = "yolo26n.pt",
        imgsz: int = 640,
        conf: float = 0.25,
        ball_model: str | None = None,
        ball_imgsz: int = 1280,
        ball_conf: float = 0.25,
        device: str = "auto",
        half: bool | None = None,
    ) -> None:
        import torch
        from ultralytics import YOLO

        self.imgsz = imgsz
        self.conf = conf
        # device: "auto" detecta GPU (cuda/ROCm); "cpu" o "cuda" fuerzan (benchmark).
        if device == "cpu":
            self.device = "cpu"
        elif device == "cuda":
            self.device = "cuda:0"
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # half (FP16): regla 6 dice half=True en GPU. PERO el FP16 de MIOpen en ROCm
        # 7.2.1 Windows (preview) esta roto: la 1a inferencia ok y el resto devuelve
        # 0 cajas. En FP32 es estable y correcto. Por eso half se auto-desactiva en
        # ROCm (torch.version.hip != None) y queda activo solo en CUDA NVIDIA / VPS.
        is_rocm = getattr(torch.version, "hip", None) is not None
        if self.device == "cpu":
            self.half = False
        elif half is not None:
            self.half = half
        else:
            self.half = not is_rocm
        if self.device == "cpu":
            logger.warning("Sin GPU: inferencia en CPU sin half. Sera lento.")
        elif is_rocm and self.half:
            logger.warning("half=True forzado en ROCm: FP16 inestable (MIOpen), puede dar 0 cajas.")
        elif is_rocm:
            logger.info("ROCm detectado: half=False (FP16 inestable en MIOpen Windows).")
        # path bajo models/ -> ultralytics descarga ahi si no existe
        self.model = YOLO(f"models/{model}")
        self.model.to(self.device)

        self.ball = None
        if ball_model:
            self.ball_imgsz = ball_imgsz
            self.ball_conf = ball_conf
            self.ball = YOLO(f"models/{ball_model}")
            self.ball.to(self.device)
            logger.info("Especialista de balon activo: %s (imgsz=%d)", ball_model, ball_imgsz)

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """1 frame BGR -> [{cls, conf, xyxy}] en pixeles, solo persona/balon."""
        coco_classes = PERSON_ONLY if self.ball else RELEVANT_CLASSES
        results = self.model.predict(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            half=self.half,
            classes=list(coco_classes),
            verbose=False,
        )
        boxes = results[0].boxes
        dets = [
            {
                "cls": coco_classes[int(cls_id)],
                "conf": float(conf),
                "xyxy": tuple(float(v) for v in xyxy),
            }
            for cls_id, conf, xyxy in zip(
                boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
            )
        ]
        if self.ball:
            dets.extend(self._detect_ball(frame_bgr))
        return dets

    def _detect_ball(self, frame_bgr: np.ndarray) -> list[dict]:
        """2da pasada con el especialista. Toda deteccion -> 'balon'."""
        results = self.ball.predict(
            frame_bgr,
            imgsz=self.ball_imgsz,
            conf=self.ball_conf,
            half=self.half,
            verbose=False,
        )
        boxes = results[0].boxes
        return [
            {"cls": "balon", "conf": float(conf), "xyxy": tuple(float(v) for v in xyxy)}
            for conf, xyxy in zip(boxes.conf.tolist(), boxes.xyxy.tolist())
        ]
