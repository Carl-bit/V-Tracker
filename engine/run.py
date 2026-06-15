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
    on_progress=None,
):
    """video -> (meta, ball_traj, player_tracks, events, sampled_fps). Todo en px.

    on_progress: callable(frames_procesados, total_estimado) opcional. Se llama por
    frame; sirve para que el worker ARQ reporte % a Redis (engine no conoce Redis).
    """
    meta = get_video_meta(video)
    total = _expected_frames(meta, sample_every_n, start_seg, end_seg)
    logger.info(
        "video=%s %dx%d fps=%.1f dur=%.1fs -> ~%d frames a 1/%d",
        Path(video).name, meta["width"], meta["height"], meta["fps"],
        meta["duration_seconds"], total, sample_every_n,
    )

    det = Detector(model=model, ball_model=ball_model, device=device, half=half)
    logger.info("detector listo: model=%s ball=%s device=%s half=%s",
                model, ball_model, det.device, det.half)

    roi = CourtROI.load(roi_path) if roi_path else None
    if roi is not None:
        logger.info("ROI de cancha activo: %s (%d vertices)", roi_path, len(roi.polygon_norm))
    w_px, h_px = float(meta["width"]), float(meta["height"])

    players_tracker = PlayerTracker()
    ball_tracker = BallTracker(frame_width=meta["width"])
    players: dict[int, list[dict]] = {}

    processed = 0
    for frame_idx, ts, frame in read_frames(video, sample_every_n=sample_every_n):
        if ts < start_seg:
            continue
        if end_seg is not None and ts > end_seg:
            break
        dets = det.detect(frame)
        if roi is not None:
            # filtra SOLO el balon por ROI (jugadores quedan completos: corren tambien fuera de lineas)
            dets = [d for d in dets if d["cls"] != "balon" or roi.contains_norm(
                (d["xyxy"][0] + d["xyxy"][2]) / 2 / w_px,
                (d["xyxy"][1] + d["xyxy"][3]) / 2 / h_px)]
        for box in players_tracker.update(dets, frame_idx, ts):
            players.setdefault(box["id"], []).append(box)
        ball_tracker.update(dets, frame_idx, ts)
        processed += 1
        if on_progress is not None:
            on_progress(processed, total)
        if processed % PROGRESS_EVERY == 0:
            pct = min(100, int(100 * processed / total))
            logger.info("progreso: %d frames (~%d%%) t=%.1fs", processed, pct, ts)

    logger.info("deteccion fin: %d frames procesados", processed)
    traj = ball_tracker.trajectory()
    events = detect_events(traj, meta)
    sampled_fps = meta["fps"] / sample_every_n
    return meta, traj, players, events, sampled_fps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="engine.run", description="VLY motor standalone")
    ap.add_argument("--video", required=True, help="ruta al .mp4")
    ap.add_argument("--out", required=True, help="ruta del JSON de salida")
    ap.add_argument("--model", default="yolo26n.pt", help="modelo YOLO base (persona)")
    ap.add_argument("--sample", type=int, default=10, help="1 de cada N frames")
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
