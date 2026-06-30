"""Motor standalone (Fase 1): video -> detect -> track -> events -> export.

CLI, sin FastAPI ni ARQ. Se valida por terminal (regla estructura).

  python -m engine.run --video X.mp4 --out Y.json --model yolo26n.pt --sample 10

Progreso a stderr (frames, %). Errores -> exit code != 0. CPU sin GPU es lento:
usar --start/--end para acotar una ventana al validar.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from engine.detect import Detector
from engine.events import detect_events
from engine.export import build_result
from engine.roi import CourtROI
from engine.scenes import SceneCutDetector
from engine.track import BallTracker, PlayerTracker
from engine.video import VideoOpenError, get_video_meta, read_frames

logger = logging.getLogger("engine.run")

PROGRESS_EVERY = 25  # cada cuantos frames muestreados se loguea progreso


def _expected_frames(meta: dict, sample_every_n: int, start_seg: float, end_seg: float | None) -> int:
    """Frames muestreados que se esperan en la ventana (para calcular %)."""
    fps = meta["fps"]
    end = end_seg if end_seg is not None else meta["duration_seconds"]
    span = max(0.0, min(end, meta["duration_seconds"]) - start_seg)
    return max(1, int(span * fps / sample_every_n))


# Dos pasadas adaptativas: pasada 1 gruesa (1/sample) sobre todo el video para
# jugadores + ubicar ventanas con balon; pasada 2 densa (1/dense) SOLO dentro de
# esas ventanas, para que detect_events tenga resolucion temporal en los rallies
# sin pagar la densidad en el tiempo muerto. WINDOW_* delimitan las ventanas.
WINDOW_MERGE_GAP_S = 1.5  # huecos de balon menores a esto = la misma ventana (mismo rally)
WINDOW_PAD_S = 0.6        # margen alrededor de cada ventana (no cortar el contacto del borde)


def _roi_filter(dets: list[dict], roi, w_px: float, h_px: float) -> list[dict]:
    """Filtra SOLO el balon por ROI (jugadores corren tambien fuera de lineas)."""
    if roi is None:
        return dets
    return [d for d in dets if d["cls"] != "balon" or roi.contains_norm(
        (d["xyxy"][0] + d["xyxy"][2]) / 2 / w_px,
        (d["xyxy"][1] + d["xyxy"][3]) / 2 / h_px)]


def _in_windows(ts: float, windows: list[tuple[float, float]]) -> bool:
    return any(a <= ts <= b for a, b in windows)


def _active_windows(coarse_traj: list[dict], duration: float) -> list[tuple[float, float]]:
    """Tramos de tiempo con balon (rallies), unidos por WINDOW_MERGE_GAP_S y con padding."""
    pts = sorted(p["timestamp"] for p in coarse_traj if p.get("x") is not None)
    if not pts:
        return []
    groups: list[list[float]] = [[pts[0], pts[0]]]
    for t in pts[1:]:
        if t - groups[-1][1] <= WINDOW_MERGE_GAP_S:
            groups[-1][1] = t
        else:
            groups.append([t, t])
    wins: list[tuple[float, float]] = []
    for a, b in groups:
        wa, wb = max(0.0, a - WINDOW_PAD_S), min(duration, b + WINDOW_PAD_S)
        if wins and wa <= wins[-1][1]:
            wins[-1] = (wins[-1][0], max(wins[-1][1], wb))
        else:
            wins.append((wa, wb))
    return wins


def _detection_pass(
    video, det, roi, w_px, h_px, sample, start_seg, end_seg,
    track_players, windows, total, on_progress, prog_lo, prog_hi, label,
    scene_detector=None, scene_cuts=None, residual_frac=1.0, detour_ratio=1.8,
    reacq_radius_frac=0.5,
):
    """Una pasada de deteccion. windows!=None procesa solo frames dentro de ellas.

    track_players: corre ByteTrack (solo se necesita en la pasada gruesa).
    scene_detector: SceneCutDetector alimentado con cada frame procesado (solo en la
      pasada gruesa, que recorre el video en orden). scene_cuts: cortes ya conocidos
      para segmentar el tracker (pasada densa). prog_lo/prog_hi: rango [0-1] global.
    Devuelve (players, ball_trajectory, frames_procesados, scene_cuts).
    """
    ptracker = PlayerTracker() if track_players else None
    btracker = BallTracker(frame_width=w_px, residual_frac=residual_frac, detour_ratio=detour_ratio,
                           reacq_radius_frac=reacq_radius_frac)
    players: dict[int, list[dict]] = {}
    processed = 0
    for frame_idx, ts, frame in read_frames(video, sample_every_n=sample):
        if ts < start_seg:
            continue
        if end_seg is not None and ts > end_seg:
            break
        if windows is not None and not _in_windows(ts, windows):
            continue
        if scene_detector is not None:
            scene_detector.update(frame, ts)
        dets = _roi_filter(det.detect(frame), roi, w_px, h_px)
        if ptracker is not None:
            for box in ptracker.update(dets, frame_idx, ts):
                players.setdefault(box["id"], []).append(box)
        btracker.update(dets, frame_idx, ts)
        processed += 1
        if on_progress is not None and total > 0:
            frac = prog_lo + (prog_hi - prog_lo) * min(1.0, processed / total)
            on_progress(int(frac * total), total)
        if processed % PROGRESS_EVERY == 0:
            pct = int(100 * (prog_lo + (prog_hi - prog_lo) * min(1.0, processed / total)))
            logger.info("%s: %d frames (~%d%%) t=%.1fs", label, processed, pct, ts)
    cuts = scene_detector.cuts() if scene_detector is not None else (scene_cuts or [])
    return players, btracker.trajectory(cuts), processed, cuts


def run_pipeline(
    video: str | Path,
    model: str = "yolo26n.pt",
    sample_every_n: int = 10,
    start_seg: float = 0.0,
    end_seg: float | None = None,
    ball_model: str = "ball_best.pt",
    device: str = "auto",
    half: bool | None = None,
    roi_path: str | Path | None = None,
    dense_sample: int | None = 3,
    residual_frac: float = 1.0,
    detour_ratio: float = 1.8,
    reacq_radius_frac: float = 0.5,
    ball_conf: float = 0.25,
    on_progress=None,
):
    """video -> (meta, ball_traj, player_tracks, events, sampled_fps). Todo en px.

    sample_every_n: muestreo grueso (pasada 1, todo el video).
    dense_sample: muestreo fino dentro de ventanas con balon (pasada 2). Si es None
      o >= sample_every_n, no hay 2a pasada (se usa la trayectoria gruesa).
    on_progress: callable(frames_procesados, total_estimado) opcional. Para que el
      worker ARQ reporte % a Redis (engine no conoce Redis).
    """
    meta = get_video_meta(video)
    w_px, h_px = float(meta["width"]), float(meta["height"])
    two_pass = dense_sample is not None and dense_sample < sample_every_n
    total = _expected_frames(meta, sample_every_n, start_seg, end_seg)
    logger.info(
        "video=%s %dx%d fps=%.1f dur=%.1fs -> ~%d frames a 1/%d%s",
        Path(video).name, meta["width"], meta["height"], meta["fps"],
        meta["duration_seconds"], total, sample_every_n,
        f" + densa 1/{dense_sample} en ventanas" if two_pass else "",
    )

    det = Detector(model=model, ball_model=ball_model, device=device, half=half, ball_conf=ball_conf)
    logger.info("detector listo: model=%s ball=%s device=%s half=%s",
                model, ball_model, det.device, det.half)

    roi = CourtROI.load(roi_path) if roi_path else None
    if roi is not None:
        logger.info("ROI de cancha activo: %s (%d vertices)", roi_path, len(roi.polygon_norm))

    # pasada 1 gruesa: jugadores (todo el video) + balon grueso para ubicar ventanas.
    # SceneCutDetector se alimenta aca (recorrido secuencial) -> cortes de camara.
    scene_detector = SceneCutDetector()
    p1_hi = 0.5 if two_pass else 1.0
    players, coarse_traj, n1, scene_cuts = _detection_pass(
        video, det, roi, w_px, h_px, sample_every_n, start_seg, end_seg,
        track_players=True, windows=None, total=total,
        on_progress=on_progress, prog_lo=0.0, prog_hi=p1_hi, label="pasada1",
        scene_detector=scene_detector, residual_frac=residual_frac, detour_ratio=detour_ratio,
        reacq_radius_frac=reacq_radius_frac,
    )
    logger.info("pasada1 fin: %d frames, balon=%d pts, cortes de escena=%d",
                n1, len(coarse_traj), len(scene_cuts))

    traj = coarse_traj
    sampled_fps = meta["fps"] / sample_every_n
    if two_pass:
        windows = _active_windows(coarse_traj, meta["duration_seconds"])
        span = sum(b - a for a, b in windows)
        logger.info("ventanas activas: %d (%.1fs de %.1fs)",
                    len(windows), span, meta["duration_seconds"])
        if windows:
            dense_total = max(1, int(span * meta["fps"] / dense_sample))
            _, dense_traj, n2, _ = _detection_pass(
                video, det, roi, w_px, h_px, dense_sample, start_seg, end_seg,
                track_players=False, windows=windows, total=dense_total,
                on_progress=on_progress, prog_lo=0.5, prog_hi=1.0, label="pasada2",
                scene_cuts=scene_cuts, residual_frac=residual_frac, detour_ratio=detour_ratio,
                reacq_radius_frac=reacq_radius_frac,
            )
            logger.info("pasada2 fin: %d frames, balon=%d pts", n2, len(dense_traj))
            if dense_traj:
                traj = dense_traj
                sampled_fps = meta["fps"] / dense_sample

    events = detect_events(traj, meta, scene_cuts)
    return meta, traj, players, events, sampled_fps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="engine.run", description="VLY motor standalone")
    ap.add_argument("--video", required=True, help="ruta al .mp4")
    ap.add_argument("--out", required=True, help="ruta del JSON de salida")
    ap.add_argument("--model", default="yolo26n.pt", help="modelo YOLO base (persona)")
    ap.add_argument("--sample", type=int, default=10, help="1 de cada N frames (pasada gruesa)")
    ap.add_argument("--dense", type=int, default=3,
                    help="1 de cada N en ventanas con balon (pasada densa para eventos). >=--sample lo desactiva")
    ap.add_argument("--start", type=float, default=0.0, help="segundo inicial")
    ap.add_argument("--end", type=float, default=None, help="segundo final")
    ap.add_argument("--ball-model", default="ball_best.pt", help="especialista de balon")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="auto detecta GPU (cuda/ROCm); cpu/cuda fuerzan (benchmark)")
    ap.add_argument("--half", default="auto", choices=["auto", "on", "off"],
                    help="FP16. auto: on en CUDA NVIDIA, off en ROCm (FP16 roto en MIOpen)")
    ap.add_argument("--roi", default=None,
                    help="json de poligono de cancha (scripts.pick_roi); filtra balon fuera de zona de juego")
    ap.add_argument("--job-id", default=None, help="id del job (default: vly_<epoch>)")
    ap.add_argument("--residual", type=float, default=1.0,
                    help="gate de residual del tracker vs prediccion (1.0=solo teleport; <1.0 aprieta, rechaza FP en oclusion)")
    ap.add_argument("--ball-conf", type=float, default=0.25,
                    help="umbral conf del especialista de balon. subir mata FP de oclusion (riesgo: recall en escena oscura)")
    ap.add_argument("--detour", type=float, default=1.8,
                    help="filtro de bursts: borra islote de FP si prev->FP->next es >N veces el directo (99 desactiva)")
    ap.add_argument("--reacq-radius", type=float, default=0.5,
                    help="re-adquisicion: solo candidatos a <N*ancho de la ultima pos conocida (corta balon de banda). bajar aprieta; 99 desactiva")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,  # progreso/logs a stderr; el JSON es el artefacto
    )

    if args.sample < 1:
        logger.error("--sample debe ser >= 1, vino %d", args.sample)
        return 2

    job_id = args.job_id or f"vly_{int(time.time())}"
    t0 = time.time()
    try:
        half = {"auto": None, "on": True, "off": False}[args.half]
        meta, traj, players, events, sampled_fps = run_pipeline(
            args.video, args.model, args.sample, args.start, args.end,
            args.ball_model, device=args.device, half=half, roi_path=args.roi,
            dense_sample=args.dense, residual_frac=args.residual,
            detour_ratio=args.detour, reacq_radius_frac=args.reacq_radius,
            ball_conf=args.ball_conf,
        )
        result = build_result(
            meta, traj, players, events,
            job_id=job_id, sampled_fps=sampled_fps, out_path=args.out,
        )
    except VideoOpenError as e:
        logger.error("video invalido: %s", e)
        return 3
    except FileNotFoundError as e:
        logger.error("archivo no encontrado: %s", e)
        return 4
    except Exception as e:  # noqa: BLE001 - frontera del CLI: cualquier fallo -> exit != 0
        logger.exception("fallo en el pipeline: %s", e)
        return 1

    s = result.match_metadata.statistics_summary
    logger.info("OK %.1fs -> %s", time.time() - t0, args.out)
    logger.info("balon=%d pts jugadores=%d eventos=%d %s",
                len(traj), len(players), len(events), [e.type for e in events])
    logger.info("rallies=%d toques=%d max_speed=%skm/h ef=%s%%",
                s.total_rallies, s.total_ball_touches,
                s.max_ball_speed_kmh, s.attack_efficiency_percentage)
    logger.info("trayectoria densa aparte: %s_ball_track.json", Path(args.out).stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
