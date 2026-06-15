"""DoD worker: encolar/ejecutar un job procesa el video y deja result en Redis.

No requiere Redis ni docker: usa un doble async en memoria que imita get/set, asi
el DoD es deterministico. Ejercita la tarea real analyze_video end-to-end (motor
incluido) y valida que el result guardado cumple el contrato.

Para la prueba contra Redis REAL ver docs/COMANDOS.md (enqueue_job + arq worker).

Uso: python -m tests.check_worker [path_video] [sample] [start_seg] [end_seg]
Default: primer .mp4 en data/, sample 10, ventana 25s-40s (rapido en CPU).
"""

import asyncio
import json
import sys
from pathlib import Path

from api.schemas import AnalysisResult
from worker.tasks import _result_key, _status_key, analyze_video


class FakeRedis:
    """Doble async minimo: get/set en memoria + historial de status escritos."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.status_history: list[dict] = []

    async def set(self, key: str, val: str, ex: int | None = None) -> None:
        self.store[key] = val
        if key.endswith(":status"):
            self.status_history.append(json.loads(val))

    async def get(self, key: str):
        v = self.store.get(key)
        return v.encode() if isinstance(v, str) else v


def main() -> None:
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
    else:
        candidates = sorted(Path("data").rglob("*.mp4"))
        if not candidates:
            sys.exit("No hay .mp4 en data/")
        video = candidates[0]
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    start_seg = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
    end_seg = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0

    job_id = "vly_check_worker"
    fake = FakeRedis()
    ctx = {"redis": fake, "job_id": job_id}

    ret = asyncio.run(
        analyze_video(ctx, job_id, str(video), sample=sample, start=start_seg, end=end_seg)
    )

    # status final
    status = json.loads(fake.store[_status_key(job_id)])
    seq = [s["status"] for s in fake.status_history]
    print(f"retorno tarea: {ret}")
    print(f"secuencia status: {seq}")
    print(f"status final: {status}")

    # result en Redis -> valida contrato
    raw = fake.store.get(_result_key(job_id))
    if raw is None:
        sys.exit("FALLA: no hay result en Redis")
    result = AnalysisResult.model_validate(json.loads(raw))
    print(f"result en Redis OK: eventos={len(result.timeline_events)} "
          f"{[e.type for e in result.timeline_events]}")

    ok = (
        status["status"] == "completado"
        and status["progress"] == 100
        and "procesando" in seq
        and seq.index("procesando") < len(seq) - 1  # hubo updates antes de completar
    )
    print(f"\nDoD (job procesa video + result valido en Redis): {'OK' if ok else 'FALLA'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
