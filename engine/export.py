"""Ensambla el JSON de salida y lo valida contra el contrato (api.schemas).

Entrada en PIXELES (lo que dan engine.track / engine.events); aqui se normaliza
TODO a 0.0-1.0 (regla 4) y se valida con AnalysisResult (regla 5). No se redefine
el schema: se importa.

La trayectoria DENSA del balon NO va en el payload principal (INVESTIGACION sec 6):
se escribe a un archivo aparte (<out>_ball_track.json) como array normalizado.

Sin homografia (MVP): la velocidad en km/h es aproximada. Se asume que el ancho de
imagen abarca COURT_VIEW_M metros. Constante ajustable, no es medicion real.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from api.schemas import SCHEMA_VERSION, AnalysisResult

# --- Constantes ajustables ---------------------------------------------------
HEATMAP_BINS = 12          # grilla NxN para el heatmap del balon
MIN_PLAYER_DETS = 3        # un track con menos detecciones no entra en impact zones
COURT_VIEW_M = 18.0        # metros que se asume abarca el ancho de imagen (px->m, rough)
NET_X_FRAC = 0.5           # linea de red ~ centro horizontal (posesion por lado)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _center_norm(xyxy: tuple[float, float, float, float], w: float, h: float) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return _clamp01((x1 + x2) / 2.0 / w), _clamp01((y1 + y2) / 2.0 / h)


def build_result(
    meta: dict,
    ball_traj: list[dict],
    player_tracks: dict[int, list[dict]],
    events: list,
    job_id: str = "vly_local",
    sampled_fps: float | None = None,
    out_path: str | Path | None = None,
) -> AnalysisResult:
    """meta + trayectoria balon + tracks jugadores + eventos -> AnalysisResult validado.

    meta: {duration_seconds, fps, width, height} (de engine.video.get_video_meta).
    ball_traj: [{frame_idx, timestamp, x, y, interpolated}] en px (engine.track).
    player_tracks: {id: [ {xyxy, frame_idx, timestamp, ...} ]} en px (engine.track).
    events: [engine.events.TimelineEvent] en px.
    sampled_fps: fps efectivo procesado. Si None se infiere de meta o cae a fps.
    out_path: si se da, escribe el JSON principal ahi y la trayectoria densa aparte.
    """
    w = float(meta["width"])
    h = float(meta["height"])
    fps = float(meta["fps"])
    if sampled_fps is None:
        sampled_fps = float(meta.get("sampled_fps", fps))
    m_per_px = COURT_VIEW_M / w

    speed_timeline = _ball_speed_timeline(ball_traj, m_per_px)
    max_speed_kmh = max((p["speed"] for p in speed_timeline), default=0.0)

    doc = {
        "match_metadata": {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "status": "completado",
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "sampled_fps": sampled_fps,
            "video": {
                "duration_seconds": float(meta["duration_seconds"]),
                "fps_processed": fps,
                "original_resolution": f"{int(w)}x{int(h)}",
            },
            "statistics_summary": _statistics(events, ball_traj, max_speed_kmh, sampled_fps),
        },
        "charts_data": {
            "ball_speed_timeline": speed_timeline,
            "team_possession_percentage": _possession(ball_traj, w),
        },
        "spatial_data": {
            "ball_heat_map": _heat_map(ball_traj, w, h),
            "player_impact_zones": _impact_zones(player_tracks, w, h),
        },
        "timeline_events": _timeline_events(events, w, h),
    }

    result = AnalysisResult.model_validate(doc)  # regla 5: el contrato manda

    if out_path is not None:
        out_path = Path(out_path)
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        _write_ball_track(out_path, ball_traj, w, h, job_id, sampled_fps)
    return result


def _ball_speed_timeline(ball_traj: list[dict], m_per_px: float) -> list[dict]:
    """Array plano [{timestamp, speed}] en km/h (aprox, sin homografia)."""
    out: list[dict] = []
    for a, b in zip(ball_traj, ball_traj[1:]):
        dt = b["timestamp"] - a["timestamp"]
        if dt <= 0:
            continue
        px_s = math.hypot(b["x"] - a["x"], b["y"] - a["y"]) / dt
        out.append(
            {"timestamp": round(b["timestamp"], 3), "speed": round(px_s * m_per_px * 3.6, 2)}
        )
    return out


def _possession(ball_traj: list[dict], w: float) -> dict:
    """Posesion por lado de cancha: fraccion de puntos de balon en cada mitad."""
    a = sum(1 for p in ball_traj if p["x"] < w * NET_X_FRAC)
    n = len(ball_traj)
    if n == 0:
        return {"team_a": 50.0, "team_b": 50.0}
    pa = round(100.0 * a / n, 1)
    return {"team_a": pa, "team_b": round(100.0 - pa, 1)}


def _statistics(events: list, ball_traj: list[dict], max_speed_kmh: float, sampled_fps: float) -> dict:
    saques = sum(1 for e in events if e.type == "saque")
    remates = sum(1 for e in events if e.type == "remate")
    rallies = saques if saques > 0 else _count_segments(ball_traj, sampled_fps)
    touches = len(events)
    # eficiencia de ataque heuristica: remates respecto a toques totales (clamp 0-100)
    eff = 100.0 * remates / touches if touches else 0.0
    return {
        "total_rallies": rallies,
        "total_ball_touches": touches,
        "max_ball_speed_kmh": round(max_speed_kmh, 2),
        "attack_efficiency_percentage": round(min(100.0, max(0.0, eff)), 1),
    }


def _count_segments(ball_traj: list[dict], sampled_fps: float) -> int:
    """Cuenta tramos continuos de trayectoria (corte = hueco temporal grande)."""
    if not ball_traj:
        return 0
    gap = 4.0 / sampled_fps if sampled_fps > 0 else 0.5
    segs = 1
    for a, b in zip(ball_traj, ball_traj[1:]):
        if b["timestamp"] - a["timestamp"] > gap:
            segs += 1
    return segs


def _heat_map(ball_traj: list[dict], w: float, h: float) -> list[dict]:
    """Grilla HEATMAP_BINS x HEATMAP_BINS; celda -> conteo de pasos del balon."""
    counts: dict[tuple[int, int], int] = {}
    for p in ball_traj:
        cx = min(HEATMAP_BINS - 1, int(_clamp01(p["x"] / w) * HEATMAP_BINS))
        cy = min(HEATMAP_BINS - 1, int(_clamp01(p["y"] / h) * HEATMAP_BINS))
        counts[(cx, cy)] = counts.get((cx, cy), 0) + 1
    return [
        {
            "x_norm": round((cx + 0.5) / HEATMAP_BINS, 4),
            "y_norm": round((cy + 0.5) / HEATMAP_BINS, 4),
            "intensity": n,
        }
        for (cx, cy), n in sorted(counts.items())
    ]


def _impact_zones(player_tracks: dict[int, list[dict]], w: float, h: float) -> list[dict]:
    """Posicion media normalizada por jugador. role/player_id reales = backlog."""
    zones: list[dict] = []
    for pid, boxes in player_tracks.items():
        if len(boxes) < MIN_PLAYER_DETS:
            continue
        xs, ys = [], []
        for b in boxes:
            nx, ny = _center_norm(b["xyxy"], w, h)
            xs.append(nx)
            ys.append(ny)
        zones.append(
            {
                "player_id": f"player_{pid}",
                "role": "unknown",
                "avg_x": round(sum(xs) / len(xs), 4),
                "avg_y": round(sum(ys) / len(ys), 4),
            }
        )
    return zones


def _timeline_events(events: list, w: float, h: float) -> list[dict]:
    out: list[dict] = []
    for e in events:
        out.append(
            {
                "event_id": e.event_id,
                "timestamp": round(e.timestamp, 3),
                "type": e.type,
                "team": e.team,
                "player_id": e.player_id,
                "confidence": round(e.confidence, 3),
                "ball_coordinates": {
                    "x_norm": _clamp01(e.x / w),
                    "y_norm": _clamp01(e.y / h),
                },
            }
        )
    return out


def _write_ball_track(
    out_path: Path, ball_traj: list[dict], w: float, h: float, job_id: str, sampled_fps: float
) -> Path:
    """Trayectoria densa normalizada, archivo aparte (no va en el payload principal)."""
    track_path = out_path.with_name(f"{out_path.stem}_ball_track.json")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "sampled_fps": sampled_fps,
        "points": [
            {
                "timestamp": round(p["timestamp"], 3),
                "x_norm": _clamp01(p["x"] / w),
                "y_norm": _clamp01(p["y"] / h),
                "interpolated": bool(p["interpolated"]),
            }
            for p in ball_traj
        ],
    }
    track_path.write_text(json.dumps(payload), encoding="utf-8")
    return track_path
