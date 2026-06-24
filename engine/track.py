"""Tracking: jugadores (ByteTrack) + trayectoria de balon (asociacion + interpolacion).

Standalone: sin FastAPI, sin ARQ. Consume detecciones de engine.detect.Detector
(lista de {cls, conf, xyxy} en pixeles por frame).

Coordenadas SIEMPRE en PIXELES aqui. Normalizar 0.0-1.0 va en export (regla 4).

- Jugadores: ByteTrack (el que trae ultralytics, sin dep nueva - regla 9) sobre
  detecciones 'persona' -> ids estables por frame.
- Balon: una sola trayectoria por asociacion frame-a-frame (cercania al ultimo
  punto aceptado) + interpolacion lineal de huecos cortos (balon perdido pocos
  frames muestreados).
"""

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


class _Dets:
    """Wrapper minimo con la interfaz que espera BYTETracker.update().

    Necesita: .conf (N,), .cls (N,), .xywh (N,4), len(), e indexado booleano.
    No define .xywhr para que parse_bboxes use xywh (caja recta).
    """

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_Dets":
        return _Dets(self.xywh[idx], self.conf[idx], self.cls[idx])


def _xyxy_to_xywh(xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)


def _bytetrack_args(track_buffer: int):
    """Carga defaults de bytetrack.yaml como namespace que espera BYTETracker."""
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    cfg = YAML.load(check_yaml("bytetrack.yaml"))
    cfg["track_buffer"] = track_buffer
    return IterableSimpleNamespace(**cfg)


class PlayerTracker:
    """ByteTrack sobre detecciones 'persona'. Devuelve cajas con id estable.

    track_buffer: frames (muestreados) que un track sobrevive perdido antes de
    morir. Default 30 = el del yaml. Subir aguanta mas oclusion, arriesga id-switch.
    """

    def __init__(self, track_buffer: int = 30) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.tracker = BYTETracker(_bytetrack_args(track_buffer))

    def update(self, detections: list[dict], frame_idx: int, timestamp: float) -> list[dict]:
        """1 frame de detecciones -> [{id, xyxy, conf, frame_idx, timestamp}] de jugadores."""
        persons = [d for d in detections if d["cls"] == "persona"]
        if persons:
            xywh = np.array([_xyxy_to_xywh(d["xyxy"]) for d in persons], dtype=np.float32)
            conf = np.array([d["conf"] for d in persons], dtype=np.float32)
            cls = np.zeros(len(persons), dtype=np.float32)
        else:
            xywh = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.float32)

        out = self.tracker.update(_Dets(xywh, conf, cls))
        # out: filas [x1, y1, x2, y2, track_id, score, cls, idx]
        return [
            {
                "id": int(row[4]),
                "xyxy": (float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                "conf": float(row[5]),
                "frame_idx": frame_idx,
                "timestamp": float(timestamp),
            }
            for row in out
        ]


class BallTracker:
    """Trayectoria unica del balon por asociacion frame-a-frame + interpolacion.

    Asociacion: entre los candidatos 'balon' de un frame elige el mas cercano al
    ultimo punto aceptado (el balon es uno solo). Sin punto previo: el de mayor conf.

    Gate de distancia: un candidato que implica velocidad > max_speed_pxs respecto
    al ultimo punto es un teleport (FP del especialista) -> se descarta ese frame.
    Tras max_gap frames perdidos se re-adquiere por mayor conf (rally/escena nueva).

    max_gap: maximo de huecos (frames muestreados consecutivos sin balon) que se
    rellenan por interpolacion lineal. Hueco mas largo = se corta el segmento.
    max_speed_pxs: velocidad maxima plausible del balon (px/s). Depende de la
    resolucion; el default es para ~1280 de ancho y se escala con frame_width.

    Balon ESTATICO (2o balon quieto en banda): un balon nitido e inmovil tiene conf
    alta y se roba la re-adquisicion (que elige por mayor conf). Antes de asociar se
    detectan las celdas donde un balon persiste muchos frames casi sin moverse y se
    descartan: solo el balon de JUEGO (que se mueve) sobrevive. frame_width escala
    el gate de velocidad y el tamano de celda con la resolucion real del video.
    """

    def __init__(
        self,
        max_gap: int = 5,
        max_speed_pxs: float = 3000.0,
        frame_width: float | None = None,
        static_cell_px: float | None = None,
        static_min_span_s: float = 5.0,
        static_min_occupancy: float = 0.5,
    ) -> None:
        ref_w = 1280.0
        scale = (frame_width / ref_w) if frame_width else 1.0
        self.max_gap = max_gap
        self.max_speed_pxs = max_speed_pxs * scale
        # celda ~ 4% del ancho (un balon entra de sobra y tolera jitter del estatico)
        self.static_cell_px = static_cell_px or max(48.0, (frame_width or ref_w) * 0.04)
        self.static_min_span_s = static_min_span_s
        self.static_min_occupancy = static_min_occupancy
        self._frames: list[tuple[int, float, list[dict]]] = []

    def update(self, detections: list[dict], frame_idx: int, timestamp: float) -> None:
        """Acumula los candidatos 'balon' de un frame. La trayectoria se arma al final."""
        balls = [d for d in detections if d["cls"] == "balon"]
        self._frames.append((frame_idx, float(timestamp), balls))

    def _static_cells(self, frames: list[tuple[int, float, list[dict]]]) -> set[tuple[int, int]]:
        """Celdas (grilla static_cell_px) donde un balon persiste inmovil = balon de banda.

        Una celda es estatica si tiene detecciones que abarcan >= static_min_span_s
        segundos Y ocupan >= static_min_occupancy de los frames de ese tramo (el balon
        de juego cruza una celda en 1-2 frames; el quieto esta casi siempre presente).

        Se computa por SHOT: tras un corte de camara el balon de banda salta de lugar,
        asi que una celda estatica solo tiene sentido dentro de una misma toma.
        """
        cell = self.static_cell_px
        ords_by_cell: dict[tuple[int, int], list[int]] = {}
        for ordi, (_fidx, _ts, balls) in enumerate(frames):
            for b in balls:
                cx, cy = _center(b)
                key = (int(cx // cell), int(cy // cell))
                ords_by_cell.setdefault(key, []).append(ordi)
        ts_of = [ts for _f, ts, _b in frames]
        static: set[tuple[int, int]] = set()
        for key, ords in ords_by_cell.items():
            omin, omax = min(ords), max(ords)
            span_ord = omax - omin + 1
            if span_ord < 3:
                continue
            occ = len(set(ords)) / span_ord
            span_s = ts_of[omax] - ts_of[omin]
            if span_s >= self.static_min_span_s and occ >= self.static_min_occupancy:
                static.add(key)
        if static:
            logger.info("balon estatico: %d celda(s) descartada(s) del tracking", len(static))
        return static

    def _split_by_cuts(
        self, scene_cuts: list[float]
    ) -> list[list[tuple[int, float, list[dict]]]]:
        """Parte self._frames en shots: un frame con ts >= proximo corte abre shot nuevo."""
        cuts = sorted(scene_cuts)
        if not cuts:
            return [self._frames]
        shots: list[list[tuple[int, float, list[dict]]]] = [[]]
        ci = 0
        for fr in self._frames:
            while ci < len(cuts) and fr[1] >= cuts[ci]:
                shots.append([])
                ci += 1
            shots[-1].append(fr)
        return [s for s in shots if s]

    def trajectory(self, scene_cuts: list[float] | None = None) -> list[dict]:
        """Trayectoria temporal: [{frame_idx, timestamp, x, y, interpolated}] en PIXELES.

        scene_cuts: timestamps de cortes de camara (engine.scenes). La asociacion, el
        gate de velocidad y el filtro de balon estatico se reinician en cada shot: tras
        un corte no hay continuidad de movimiento ni la misma posicion de balon de banda.
        """
        out: list[dict] = []
        for shot in self._split_by_cuts(scene_cuts or []):
            out.extend(self._shot_trajectory(shot))
        return out

    def _shot_trajectory(self, frames: list[tuple[int, float, list[dict]]]) -> list[dict]:
        """Trayectoria de UN shot (sin cortes de camara dentro)."""
        static = self._static_cells(frames)
        cell = self.static_cell_px

        # 1) asociacion: 1 punto aceptado por frame (o None si no hay balon)
        accepted: list[tuple[int, float, float | None, float | None, float]] = []
        last_xy: tuple[float, float] | None = None
        last_ts = 0.0
        misses = 0  # frames consecutivos sin punto aceptado
        for frame_idx, ts, balls in frames:
            # descarta candidatos en celdas de balon estatico (banda)
            if static:
                balls = [b for b in balls
                         if (int(_center(b)[0] // cell), int(_center(b)[1] // cell)) not in static]
            if not balls:
                accepted.append((frame_idx, ts, None, None, 0.0))
                misses += 1
                continue
            if last_xy is None or misses > self.max_gap:
                # re-adquisicion: sin referencia confiable, el de mayor conf
                best = max(balls, key=lambda d: d["conf"])
                cx, cy = _center(best)
            else:
                lx, ly = last_xy
                best = min(
                    balls,
                    key=lambda d: (_center(d)[0] - lx) ** 2 + (_center(d)[1] - ly) ** 2,
                )
                cx, cy = _center(best)
                dt = max(ts - last_ts, 1e-3)
                if math.hypot(cx - lx, cy - ly) / dt > self.max_speed_pxs:
                    # gate: salto imposible -> teleport/FP, este frame queda sin balon
                    accepted.append((frame_idx, ts, None, None, 0.0))
                    misses += 1
                    continue
            accepted.append((frame_idx, ts, cx, cy, best["conf"]))
            last_xy = (cx, cy)
            last_ts = ts
            misses = 0

        # 2) puntos reales detectados
        traj: list[dict] = []
        for frame_idx, ts, cx, cy, _conf in accepted:
            if cx is not None:
                traj.append(
                    {"frame_idx": frame_idx, "timestamp": ts, "x": cx, "y": cy, "interpolated": False}
                )

        if len(traj) < 2:
            return traj

        # 3) interpolacion lineal de huecos cortos entre puntos reales consecutivos
        filled: list[dict] = []
        idx_by_frame = {f[0]: i for i, f in enumerate(frames)}
        for a, b in zip(traj, traj[1:]):
            filled.append(a)
            # frames muestreados estrictamente entre a y b
            ia, ib = idx_by_frame[a["frame_idx"]], idx_by_frame[b["frame_idx"]]
            gap = ib - ia - 1
            if 0 < gap <= self.max_gap:
                for k in range(1, gap + 1):
                    t = k / (gap + 1)
                    g_idx, g_ts, _balls = frames[ia + k]
                    filled.append(
                        {
                            "frame_idx": g_idx,
                            "timestamp": g_ts,
                            "x": a["x"] + (b["x"] - a["x"]) * t,
                            "y": a["y"] + (b["y"] - a["y"]) * t,
                            "interpolated": True,
                        }
                    )
        filled.append(traj[-1])
        return filled


def _center(det: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = det["xyxy"]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def track_video(
    video_path,
    detector,
    sample_every_n: int = 10,
    track_buffer: int = 30,
    max_gap: int = 5,
) -> dict:
    """Orquesta: video -> detecciones -> tracks jugadores + trayectoria balon.

    Retorna {"players": {id: [cajas...]}, "ball": [puntos...], "frames": n}.
    Helper de conveniencia; engine no importa FastAPI/ARQ (regla estructura).
    """
    from engine.video import read_frames

    players_tracker = PlayerTracker(track_buffer=track_buffer)
    ball_tracker = BallTracker(max_gap=max_gap)
    players: dict[int, list[dict]] = {}
    n = 0
    for frame_idx, ts, frame in read_frames(video_path, sample_every_n=sample_every_n):
        dets = detector.detect(frame)
        for box in players_tracker.update(dets, frame_idx, ts):
            players.setdefault(box["id"], []).append(box)
        ball_tracker.update(dets, frame_idx, ts)
        n += 1

    return {"players": players, "ball": ball_tracker.trajectory(), "frames": n}
