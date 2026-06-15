"""Lee status (y opcionalmente result) de un job desde Redis.

Uso:
  python -m scripts.job_status <job_id>            -> imprime status
  python -m scripts.job_status <job_id> --result   -> vuelca el JSON result
"""

import asyncio
import sys

from arq import create_pool

from worker.tasks import _result_key, _status_key, redis_settings


async def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("uso: python -m scripts.job_status <job_id> [--result]")
    job_id = sys.argv[1]
    want_result = "--result" in sys.argv[2:]

    pool = await create_pool(redis_settings())
    status = await pool.get(_status_key(job_id))
    print(status.decode() if status else "(sin status: job desconocido)")
    if want_result:
        result = await pool.get(_result_key(job_id))
        print(result.decode() if result else "(sin result aun)")
    await pool.aclose()


if __name__ == "__main__":
    asyncio.run(main())
