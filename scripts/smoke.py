"""Prueba unica de Fase 1: una sola pasada de deteccion valida todo el motor.

Corre el pipeline 1 vez sobre una ventana y chequea de golpe:
  - trayectoria de balon continua (reales + interpolados)
  - tracks de jugadores con id
  - eventos con al menos saque + remate
  - AnalysisResult valida el JSON exportado

Mas barato que correr check_track + check_events + check_export por separado
(esos re-detectan cada uno). Para el DoD del worker usar tests.check_worker.

Uso: python -m scripts.smoke [path_video] [sample] [start_seg] [end_seg]
Default: primer .mp4 en data/, sample 3, ventana 25s-40s.
"""

import sys
from pathlib import Path

from api.schemas import AnalysisResult
from engine.export import build_result
from engine.run import run_pipeline


def main() -> None:
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
    else:
        candidates = sorted(Path("data").rglob("*.mp4"))
        if not candidates:
            sys.exit("No hay .mp4 en data/")
        video = candidates[0]
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    start_seg = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
    end_seg = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0

    meta, traj, players, events, sampled_fps = run_pipeline(
        video, sample_every_n=sample, start_seg=start_seg, end_seg=end_seg
    )
    result = build_result(meta, traj, players, events, sampled_fps=sampled_fps)

    real = sum(1 for p in traj if not p["interpolated"])
    interp = sum(1 for p in traj if p["interpolated"])
    types = {e.type for e in events}

    checks = {
        "trayectoria balon (>=2 reales)": real >= 2,
        "huecos interpolados presentes": interp >= 1,
        "jugadores con id": len(players) >= 1,
        "eventos saque + remate": {"saque", "remate"} <= types,
        "AnalysisResult valida": isinstance(result, AnalysisResult),
    }

    print(f"video={video.name} ventana={start_seg}-{end_seg}s sample={sample}")
    print(f"balon: {len(traj)} pts ({real} reales + {interp} interp) | "
          f"jugadores: {len(players)} | eventos: {len(events)} {sorted(types)}")
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'FALLA'}] {name}")

    all_ok = all(checks.values())
    print(f"\nSMOKE: {'OK' if all_ok else 'FALLA'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
