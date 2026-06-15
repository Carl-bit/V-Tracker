"""Encola un job de analisis en ARQ (contra Redis real).

Uso:
  python -m scripts.enqueue_job <video.mp4> [job_id] [sample]
Requiere Redis corriendo y un worker arq (ver docs/COMANDOS.md).
"""

import asyncio
import json
import sys
import time

from arq import create_pool

from worker.tasks import _status_key, analyze_video, redis_settings  # noqa: F401


async def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("uso: python -m scripts.enqueue_job <video.mp4> [job_id] [sample]")
    video = sys.argv[1]
    job_id = sys.argv[2] if len(sys.argv) > 2 else f"vly_{int(time.time())}"
    sample = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    pool = await create_pool(redis_settings())
    # status inicial en_cola lo dejamos nosotros; el worker lo pasa a procesando
    await pool.set(_status_key(job_id), json.dumps({"job_id": job_id, "status": "en_cola", "progress": 0}))
    await pool.enqueue_job("analyze_video", job_id, video, sample, _job_id=job_id)
    print(f"encolado job_id={job_id} video={video} sample={sample}")
    print(f"seguir: python -m scripts.job_status {job_id}")
    await pool.aclose()


if __name__ == "__main__":
    asyncio.run(main())
