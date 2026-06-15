"""Eventos de voley por heuristicas (NO ML) sobre la trayectoria del balon.

Standalone: sin FastAPI, sin ARQ, sin pydantic del contrato (engine independiente
de la web). Devuelve TimelineEvent dataclass en PIXELES; normalizar 0-1 y validar
contra api.schemas.TimelineEvent va en export (regla 4).

Idea:
- cambio brusco de direccion de la trayectoria = toque (alguien la toco).
- pico de velocidad hacia el campo contrario = lanzamiento fuerte (saque o remate).
- altura del balon + zona de cancha desambiguan el tipo:
  saque   -> lanzamiento fuerte desde zona de fondo, inicia el rally del segmento.
  remate  -> lanzamiento fuerte cerca de red / balon alto, hacia campo contrario.
  armado  -> toque con balon alto y poca velocidad horizontal (colocacion).
  recepcion -> toque bajo controlando un balon entrante (resto).

Vista asumida: broadcast lateral. La red ~ centro horizontal de la imagen; cada
lado es un equipo (team_a izquierda, team_b derecha). Sin homografia, sin player_id
real (backlog). team se infiere por lado de cancha.
"""

import math
from dataclasses import dataclass

# --- Umbrales ajustables (px = pixeles, s = segundos) -----------------------
# Los umbrales de velocidad estan en px/s y dependen de la RESOLUCION: estos valores
# son para ancho REF_WIDTH_PX (broadcast 1280). En detect_events se escalan por el
# ancho real (1920 frontal => ~1.5x), si no, una colocacion lenta se lee como remate.
REF_WIDTH_PX = 1280.0
SEGMENT_GAP_S = 0.6          # hueco temporal mayor = corta el rally/escena (trayectoria nueva)
MIN_SPEED_PXS = 120.0        # px/s @1280; debajo = ruido o balon casi quieto, se ignora
DIR_CHANGE_DEG = 55.0        # giro de direccion mayor a esto = toque (angulo, no escala)
SPEED_PEAK_PXS = 650.0       # px/s @1280; velocidad de salida sobre esto = lanzamiento fuerte
MAX_SPEED_PXS = 3000.0       # px/s @1280; por encima = teleport de tracking (FP), no evento real
ACCEL_RATIO = 1.15           # salida/entrada para considerar que hubo golpe (acelero)
NET_X_FRAC = 0.5             # linea de red ~ centro horizontal (vista lateral)
HIGH_BALL_FRAC = 0.45        # y_norm < esto = balon "alto" (top de imagen es y=0)
BASELINE_MARGIN_FRAC = 0.22  # x dentro de este margen del borde = zona de fondo (saque)
MERGE_WINDOW_S = 0.35        # eventos mas cercanos que esto = el mismo contacto (dedup)


@dataclass
class _Thr:
    """Umbrales de velocidad (px/s) ya escalados al ancho real del video."""

    min_speed: float
    speed_peak: float
    max_speed: float


@dataclass
class TimelineEvent:
    """Evento detectado. Coordenadas en PIXELES (x,y). Normalizar va en export."""

    event_id: str
    timestamp: float
    type: str       # saque | recepcion | armado | remate
    team: str       # team_a | team_b (por lado de cancha)
    player_id: str  # placeholder: no hay player_id real aun (backlog)
    confidence: float  # heuristico 0.0-1.0
    x: float        # centro del balon en el evento (px)
    y: float        # (px)


def detect_events(ball_trajectory: list[dict], video_meta: dict) -> list[TimelineEvent]:
    """Trayectoria de balon ({timestamp,x,y,...} en px) + meta -> eventos.

    video_meta requiere width y height (px) para zona/altura. Trayectoria asumida
    ordenada por timestamp ascendente (la que da engine.track.BallTracker).
    """
    width = float(video_meta["width"])
    height = float(video_meta["height"])
    scale = width / REF_WIDTH_PX  # umbrales px/s al ancho real (1920 frontal ~1.5x)
    thr = _Thr(MIN_SPEED_PXS * scale, SPEED_PEAK_PXS * scale, MAX_SPEED_PXS * scale)
    traj = [p for p in ball_trajectory if p.get("x") is not None]

    events: list[TimelineEvent] = []
    for seg in _split_segments(traj):
        events.extend(_events_in_segment(seg, width, height, thr))

    events = _merge(events)
    for i, e in enumerate(events, 1):
        e.event_id = f"evt_{i:03d}"
    return events


def _split_segments(traj: list[dict]) -> list[list[dict]]:
    """Corta la trayectoria donde hay un hueco temporal grande (rally/escena distinta)."""
    if not traj:
        return []
    segments: list[list[dict]] = [[traj[0]]]
    for prev, cur in zip(traj, traj[1:]):
        if cur["timestamp"] - prev["timestamp"] > SEGMENT_GAP_S:
            segments.append([])
        segments[-1].append(cur)
    return segments


def _events_in_segment(seg: list[dict], width: float, height: float, thr: _Thr) -> list[TimelineEvent]:
    if len(seg) < 3:
        return []
    out: list[TimelineEvent] = []
    served = False  # 1 saque por rally/segmento; el 1er lanzamiento de fondo es el saque

    # El saque inicia el rally: el balon aparece (arranca el segmento) y sale fuerte
    # desde la zona de fondo. El loop interior no ve el 1er punto (no tiene v_in),
    # asi que se evalua aparte. Solo se emite si clasifica saque (condicion estricta:
    # fondo+bajo+rapido+hacia campo contrario), no para cualquier balon que aparece.
    first, second = seg[0], seg[1]
    dt0 = max(second["timestamp"] - first["timestamp"], 1e-3)
    v0 = (second["x"] - first["x"], second["y"] - first["y"])
    s0 = math.hypot(*v0) / dt0
    if thr.speed_peak < s0 <= thr.max_speed:
        ev0 = _classify(first, v0, s0, 180.0, width, height, served, thr)
        if ev0.type == "saque":
            out.append(ev0)
            served = True

    for i in range(1, len(seg) - 1):
        prev, cur, nxt = seg[i - 1], seg[i], seg[i + 1]
        v_in = (cur["x"] - prev["x"], cur["y"] - prev["y"])
        v_out = (nxt["x"] - cur["x"], nxt["y"] - cur["y"])
        dt_in = max(cur["timestamp"] - prev["timestamp"], 1e-3)
        dt_out = max(nxt["timestamp"] - cur["timestamp"], 1e-3)
        s_in = math.hypot(*v_in) / dt_in
        s_out = math.hypot(*v_out) / dt_out
        angle = _angle_between(v_in, v_out)

        speed = max(s_in, s_out)
        if speed > thr.max_speed:  # teleport de tracking: no es contacto real
            continue
        is_touch = angle > DIR_CHANGE_DEG and speed > thr.min_speed
        is_launch = s_out > thr.speed_peak and s_out > s_in * ACCEL_RATIO
        if not (is_touch or is_launch):
            continue

        ev = _classify(cur, v_out, s_out, angle, width, height, served, thr)
        if ev.type == "saque":
            served = True
        out.append(ev)
    return out


def _classify(
    p: dict,
    v_out: tuple[float, float],
    s_out: float,
    angle: float,
    width: float,
    height: float,
    served: bool,
    thr: _Thr,
) -> TimelineEvent:
    x, y = p["x"], p["y"]
    x_frac, y_frac = x / width, y / height
    side = "team_a" if x_frac < NET_X_FRAC else "team_b"
    dx, _dy = v_out
    toward_opposite = (side == "team_a" and dx > 0) or (side == "team_b" and dx < 0)
    high = y_frac < HIGH_BALL_FRAC
    baseline = x_frac < BASELINE_MARGIN_FRAC or x_frac > 1 - BASELINE_MARGIN_FRAC
    fast = s_out > thr.speed_peak

    if fast and toward_opposite and baseline and not served:
        etype = "saque"
    elif fast and toward_opposite:
        etype = "remate"
    elif high:
        etype = "armado"
    else:
        etype = "recepcion"

    return TimelineEvent(
        event_id="",
        timestamp=round(p["timestamp"], 2),
        type=etype,
        team=side,
        player_id="unknown",
        confidence=_confidence(etype, s_out, angle, thr),
        x=x,
        y=y,
    )


def _confidence(etype: str, s_out: float, angle: float, thr: _Thr) -> float:
    """Heuristico 0-1: lanzamientos por exceso de velocidad, toques por exceso de giro."""
    if etype in ("saque", "remate"):
        c = 0.5 + 0.5 * (s_out - thr.speed_peak) / thr.speed_peak
    else:
        c = 0.4 + 0.5 * (angle - DIR_CHANGE_DEG) / (180.0 - DIR_CHANGE_DEG)
    return round(max(0.0, min(1.0, c)), 2)


def _merge(events: list[TimelineEvent]) -> list[TimelineEvent]:
    """Dedup: contactos en frames contiguos = 1 evento. Gana el de mayor confidence."""
    events = sorted(events, key=lambda e: e.timestamp)
    merged: list[TimelineEvent] = []
    for e in events:
        if merged and e.timestamp - merged[-1].timestamp < MERGE_WINDOW_S:
            if e.confidence > merged[-1].confidence:
                merged[-1] = e
            continue
        merged.append(e)
    return merged


def _angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))
