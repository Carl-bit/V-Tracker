"""Tarea ARQ analyze_video + WorkerSettings (Fase 2: cola async).

El worker corre SEPARADO de la API (regla: jobs largos NUNCA en el endpoint).
Llama al motor como funcion (engine.run.run_pipeline), no por subprocess.

Estado y resultado viven en Redis:
  vly:job:{job_id}:status      -> JSON {job_id, status, progress, [error]}
  vly:job:{job_id}:result      -> JSON del contrato (AnalysisResult)
  vly:job:{job_id}:ball_track  -> JSON trayectoria densa (aparte, INVESTIGACION sec 6)

status: en_cola (lo pone quien encola) -> procesando -> <progress%> -> completado | error.

Hardware (regla 3): 1 worker GPU, 1 job a la vez. En ARQ eso = max_jobs=1.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from arq.connections import RedisSettings

from engine.export import build_result
from engine.run import run_pipeline

logger = logging.getLogger("worker.tasks")

# --- Config (env, con defaults de dev) --------------------------------------
KEY_PREFIX = "vly:job"
RESULT_TTL = int(os.getenv("VLY_RESULT_TTL", str(7 * 24 * 3600)))  # seg que sobrevive el resultado
DEFAULT_SAMPLE = int(os.getenv("VLY_SAMPLE", "10"))
DEFAULT_MODEL = os.getenv("VLY_MODEL", "yolo26n.pt")
DEFAULT_BALL_MODEL = os.getenv("VLY_BALL_MODEL", "ball_best.pt")
DEFAULT_DEVICE = os.getenv("VLY_DEVICE", "auto")  # auto|cpu|cuda
DEFAULT_HALF = {"auto": None, "on": True, "off": False}[os.getenv("VLY_HALF", "auto")]
DELETE_SOURCE = os.getenv("VLY_DELETE_SOURCE", "0") == "1"  # regla 8: borrar .mp4 tras JSON


def redis_settings() -> RedisSettings:
    return RedisSettings(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        database=int(os.getenv("REDIS_DB", "0")),
        password=os.getenv("REDIS_PASSWORD") or None,
    )


def _status_key(job_id: str) -> str:
    return f"{KEY_PREFIX}:{job_id}:status"


def _result_key(job_id: str) -> str:
    return f"{KEY_PREFIX}:{job_id}:result"


def _ball_track_key(job_id: str) -> str:
    return f"{KEY_PREFIX}:{job_id}:ball_track"


async def _set_status(redis, job_id: str, status: str, progress: int, error: str | None = None) -> None:
    payload = {"job_id": job_id, "status": status, "progress": int(progress)}
    if error is not None:
        payload["error"] = error
    await redis.set(_status_key(job_id), json.dumps(payload))


def _process(
    video_path: str, sample: int, model: str, start: float, end: float | None, on_progress
) -> tuple[str, str]:
    """Trabajo pesado y sincrono (corre en thread). Devuelve (result_json, ball_track_json)."""
    meta, traj, players, events, sampled_fps = run_pipeline(
        video_path,
        model=model,
        sample_every_n=sample,
        start_seg=start,
        end_seg=end,
        ball_model=DEFAULT_BALL_MODEL,
        device=DEFAULT_DEVICE,
        half=DEFAULT_HALF,
        on_progress=on_progress,
    )
    result = build_result(meta, traj, players, events, sampled_fps=sampled_fps)
    ball_track = {
        "schema_version": result.match_metadata.schema_version,
        "sampled_fps": sampled_fps,
        "points": [
            {
                "timestamp": round(p["timestamp"], 3),
                "x_norm": max(0.0, min(1.0, p["x"] / meta["width"])),
                "y_norm": max(0.0, min(1.0, p["y"] / meta["height"])),
                "interpolated": bool(p["interpolated"]),
            }
            for p in traj
        ],
    }
    return result.model_dump_json(), json.dumps(ball_track)


async def analyze_video(
    ctx: dict,
    job_id: str,
    video_path: str,
    sample: int = DEFAULT_SAMPLE,
    model: str = DEFAULT_MODEL,
    start: float = 0.0,
    end: float | None = None,
) -> dict:
    """Tarea ARQ: procesa el video y deja status + result en Redis.

    El motor es sincrono y bloqueante (CPU/GPU): corre en un thread para no
    congelar el loop. Un reporter async vuelca el progreso a Redis en paralelo.
    """
    redis = ctx["redis"]
    logger.info("job %s START video=%s sample=%d model=%s", job_id, video_path, sample, model)
    await _set_status(redis, job_id, "procesando", 0)

    progress = {"pct": 0}
    stop = asyncio.Event()

    def on_progress(done: int, total: int) -> None:
        progress["pct"] = min(99, int(100 * done / total)) if total else 0

    async def reporter() -> None:
        last = -1
        while not stop.is_set():
            if progress["pct"] != last:
                last = progress["pct"]
                await _set_status(redis, job_id, "procesando", last)
            await asyncio.sleep(0.5)

    rep = asyncio.create_task(reporter())
    try:
        result_json, ball_track_json = await asyncio.to_thread(
            _process, video_path, sample, model, start, end, on_progress
        )
    except Exception as e:  # noqa: BLE001 - cualquier fallo del motor -> estado error
        stop.set()
        await rep
        logger.exception("job %s ERROR: %s", job_id, e)
        await _set_status(redis, job_id, "error", progress["pct"], error=str(e))
        raise
    stop.set()
    await rep

    await redis.set(_result_key(job_id), result_json, ex=RESULT_TTL)
    await redis.set(_ball_track_key(job_id), ball_track_json, ex=RESULT_TTL)
    await _set_status(redis, job_id, "completado", 100)
    logger.info("job %s DONE -> %s", job_id, _result_key(job_id))

    if DELETE_SOURCE:  # regla 8: no saturar disco con el .mp4 original
        try:
            Path(video_path).unlink(missing_ok=True)
            logger.info("job %s: borrado %s", job_id, video_path)
        except OSError as e:
            logger.warning("job %s: no se pudo borrar %s: %s", job_id, video_path, e)

    return {"job_id": job_id, "status": "completado", "result_key": _result_key(job_id)}


class WorkerSettings:
    """arq worker worker.tasks.WorkerSettings"""

    functions = [analyze_video]
    redis_settings = redis_settings()
    max_jobs = 1            # regla 3: 1 job en VRAM a la vez (concurrency=1)
    job_timeout = 6 * 3600  # video de 1.5h en 3050: dar margen amplio
    max_tries = 1           # job pesado: no reintentar en loop
    keep_result = RESULT_TTL
