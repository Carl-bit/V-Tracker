"""Eventos de voley por heuristicas (NO ML) sobre la trayectoria del balon.

Standalone: sin FastAPI, sin ARQ, sin pydantic del contrato (engine independiente
de la web). Devuelve TimelineEvent dataclass en PIXELES; normalizar 0-1 y validar
contra api.schemas.TimelineEvent va en export (regla 4).

Idea (fisica, no umbral magico): entre dos contactos el balon esta en VUELO LIBRE,
una parabola. La gravedad solo:
  - conserva la velocidad horizontal vx (no la cambia nunca),
  - invierte vy de subir->bajar (el apex de la parabola).
Por lo tanto NO puede:
  - cambiar vx de golpe   -> un salto en vx = alguien redirigio (toque),
  - invertir vy bajar->subir -> el balon caia y ahora sube = golpe hacia arriba (toque),
  - acelerar el balon por encima de su rapidez de entrada -> salto de rapidez = golpe fuerte.
Asi el apex de un saque/globo (vx constante, vy subir->bajar) deja de marcarse como
toque, que era el falso positivo de las "velocidades lentas que no son pases".

Desambiguacion del tipo (altura + zona + direccion), igual que antes:
  saque   -> golpe fuerte desde zona de fondo, inicia el rally del segmento.
  remate  -> golpe fuerte hacia campo contrario.
  armado  -> toque con balon alto y poca velocidad horizontal (colocacion).
  recepcion -> toque bajo (resto / dig).

Vista asumida: broadcast lateral. Red ~ centro horizontal; cada lado un equipo
(team_a izquierda, team_b derecha). Sin homografia, sin player_id real (backlog).
y crece hacia ABAJO (convencion imagen): vy>0 = balon bajando, vy<0 = subiendo.
"""

import math
from dataclasses import dataclass

# --- Umbrales ajustables (px = pixeles, s = segundos) -----------------------
# Velocidades en px/s para ancho REF_WIDTH_PX (broadcast 1280). En detect_events
# se escalan por el ancho real (1920 => ~1.5x); sin esto una colocacion lenta se
# leeria como remate. Los angulos NO se escalan.
REF_WIDTH_PX = 1280.0
SEGMENT_GAP_S = 0.6          # hueco temporal mayor = corta el rally/escena (trayectoria nueva)
MIN_SPEED_PXS = 120.0        # px/s @1280; debajo = ruido o balon casi quieto, se ignora
SPEED_PEAK_PXS = 650.0       # px/s @1280; rapidez de salida sobre esto = golpe fuerte
MAX_SPEED_PXS = 3000.0       # px/s @1280; por encima = teleport de tracking (FP), no evento real
VX_CHANGE_PXS = 180.0        # px/s @1280; salto de vx que la gravedad NO puede causar -> redirect
VFLIP_PXS = 130.0            # px/s @1280; vy minimo para contar un golpe hacia arriba (bajaba->sube)
ACCEL_RATIO = 1.15           # salida/entrada para considerar que hubo golpe (acelero)
DIR_CHANGE_DEG = 45.0        # giro de direccion; solo para puntuar confianza del toque
NET_X_FRAC = 0.5             # linea de red ~ centro horizontal (vista lateral)
HIGH_BALL_FRAC = 0.45        # y_norm < esto = balon "alto" (top de imagen es y=0)
BASELINE_MARGIN_FRAC = 0.22  # x dentro de este margen del borde = zona de fondo (saque)
MERGE_WINDOW_S = 0.35        # eventos mas cercanos que esto = el mismo contacto (dedup)
SMOOTH_WIN = 1               # +-N puntos para suavizar posicion antes de medir velocidad


@dataclass
class _Thr:
    """Umbrales de velocidad (px/s) ya escalados al ancho real del video."""

    min_speed: float
    speed_peak: float
    max_speed: float
    vx_change: float
    vflip: float


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


def detect_events(
    ball_trajectory: list[dict], video_meta: dict, scene_cuts: list[float] | None = None
) -> list[TimelineEvent]:
    """Trayectoria de balon ({timestamp,x,y,interpolated} en px) + meta -> eventos.

    video_meta requiere width y height (px) para zona/altura. Trayectoria asumida
    ordenada por timestamp ascendente (la que da engine.track.BallTracker).
    scene_cuts: timestamps de cortes de camara; cortan segmento aunque no haya hueco
    temporal (evita stitchear dos tomas y fabricar un redirect/recepcion falso).
    """
    width = float(video_meta["width"])
    height = float(video_meta["height"])
    scale = width / REF_WIDTH_PX  # umbrales px/s al ancho real (1920 frontal ~1.5x)
    thr = _Thr(
        MIN_SPEED_PXS * scale,
        SPEED_PEAK_PXS * scale,
        MAX_SPEED_PXS * scale,
        VX_CHANGE_PXS * scale,
        VFLIP_PXS * scale,
    )
    traj = [p for p in ball_trajectory if p.get("x") is not None]

    events: list[TimelineEvent] = []
    for seg in _split_segments(traj, scene_cuts):
        events.extend(_events_in_segment(seg, width, height, thr))

    events = _merge(events)
    for i, e in enumerate(events, 1):
        e.event_id = f"evt_{i:03d}"
    return events


def _split_segments(traj: list[dict], scene_cuts: list[float] | None = None) -> list[list[dict]]:
    """Corta la trayectoria por hueco temporal grande O por corte de camara entre puntos."""
    if not traj:
        return []
    cuts = sorted(scene_cuts or [])
    segments: list[list[dict]] = [[traj[0]]]
    for prev, cur in zip(traj, traj[1:]):
        gap = cur["timestamp"] - prev["timestamp"] > SEGMENT_GAP_S
        crossed = any(prev["timestamp"] < c <= cur["timestamp"] for c in cuts)
        if gap or crossed:
            segments.append([])
        segments[-1].append(cur)
    return segments


def _smooth(vals: list[float], win: int) -> list[float]:
    """Media movil +-win (kills jitter del centro del bbox).

    Los extremos quedan SIN promediar: encoger la ventana ahi deforma la velocidad
    del borde del segmento (justo donde el balon aparece/desaparece) y fabricaria un
    salto de vx falso. Con extremo crudo, el vecino interior sigue exacto.
    """
    if win <= 0:
        return list(vals)
    n = len(vals)
    out = list(vals)
    for i in range(win, n - win):
        out[i] = sum(vals[i - win:i + win + 1]) / (2 * win + 1)
    return out


def _events_in_segment(seg: list[dict], width: float, height: float, thr: _Thr) -> list[TimelineEvent]:
    n = len(seg)
    if n < 3:
        return []

    # posiciones suavizadas para velocidad (el centro del bbox tiembla frame a frame)
    xs = _smooth([p["x"] for p in seg], SMOOTH_WIN)
    ys = _smooth([p["y"] for p in seg], SMOOTH_WIN)
    ts = [p["timestamp"] for p in seg]
    interp = [bool(p.get("interpolated")) for p in seg]

    # velocidad de entrada (i-1 -> i) y salida (i -> i+1) por punto
    vin: list[tuple[float, float] | None] = [None] * n
    vout: list[tuple[float, float] | None] = [None] * n
    s_in = [0.0] * n
    s_out = [0.0] * n
    for i in range(n):
        if i > 0:
            dt = max(ts[i] - ts[i - 1], 1e-3)
            vin[i] = ((xs[i] - xs[i - 1]) / dt, (ys[i] - ys[i - 1]) / dt)
            s_in[i] = math.hypot(*vin[i])
        if i < n - 1:
            dt = max(ts[i + 1] - ts[i], 1e-3)
            vout[i] = ((xs[i + 1] - xs[i]) / dt, (ys[i + 1] - ys[i]) / dt)
            s_out[i] = math.hypot(*vout[i])

    out: list[TimelineEvent] = []
    served = False  # 1 saque por rally/segmento; el 1er golpe fuerte de fondo es el saque

    # El saque inicia el rally: el balon aparece (arranca el segmento) y sale fuerte
    # desde la zona de fondo. El 1er punto no tiene v_in; se evalua aparte. Solo se
    # emite si clasifica saque (fondo+bajo+rapido+hacia campo contrario).
    if not interp[0] and not interp[1] and thr.speed_peak < s_out[0] <= thr.max_speed:
        ev0 = _classify(seg[0], vout[0], s_out[0], 180.0, width, height, served, thr)
        if ev0.type == "saque":
            out.append(ev0)
            served = True

    for i in range(1, n - 1):
        # un contacto en un tramo interpolado no tiene evidencia real (linea recta inventada)
        if interp[i] or interp[i - 1] or interp[i + 1]:
            continue
        vi, vo = vin[i], vout[i]
        speed = max(s_in[i], s_out[i])
        if speed < thr.min_speed or speed > thr.max_speed:
            continue  # ruido/quieto, o teleport de tracking

        dvx = vo[0] - vi[0]                 # gravedad conserva vx; salto = redirect horizontal
        falling_to_rising = vi[1] > thr.vflip and vo[1] < -thr.vflip  # bajaba y ahora sube = golpe arriba
        redirect = abs(dvx) > thr.vx_change
        launch = s_out[i] > thr.speed_peak and s_out[i] > s_in[i] * ACCEL_RATIO

        if not (redirect or falling_to_rising or launch):
            continue  # cambio explicable por gravedad (p.ej. apex): no es contacto

        angle = _angle_between(vi, vo)
        ev = _classify(seg[i], vo, s_out[i], angle, width, height, served, thr)
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
    """Heuristico 0-1: golpes por exceso de velocidad, toques por exceso de giro."""
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
