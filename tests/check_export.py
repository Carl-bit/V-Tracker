"""DoD export.py: AnalysisResult valida el output sin error.

Corre el pipeline sobre una ventana, ensambla el JSON, lo escribe, lo recarga del
disco y lo re-valida con AnalysisResult (prueba que el archivo en disco es valido,
no solo el objeto en memoria).

Uso: python -m tests.check_export [path_video] [sample_every_n] [start_seg] [end_seg]
Default: primer .mp4 en data/, 1 de cada 3 frames, ventana 25s-40s.
"""

import json
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
    sample_every_n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    start_seg = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
    end_seg = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0

    out = Path("data/out_check_export.json")
    meta, traj, players, events, sampled_fps = run_pipeline(
        video, sample_every_n=sample_every_n, start_seg=start_seg, end_seg=end_seg
    )
    build_result(meta, traj, players, events, job_id="vly_check", sampled_fps=sampled_fps, out_path=out)

    # re-validar desde disco
    doc = json.loads(out.read_text(encoding="utf-8"))
    result = AnalysisResult.model_validate(doc)

    track_path = out.with_name(f"{out.stem}_ball_track.json")
    print(f"escrito {out} ({out.stat().st_size} bytes)")
    print(f"trayectoria densa aparte: {track_path.name} (existe={track_path.is_file()})")
    print(f"eventos en payload: {len(result.timeline_events)} -> {[e.type for e in result.timeline_events]}")
    print(f"heatmap celdas: {len(result.spatial_data.ball_heat_map)} | "
          f"impact zones: {len(result.spatial_data.player_impact_zones)} | "
          f"speed points: {len(result.charts_data.ball_speed_timeline)}")
    print(f"sampled_fps={result.match_metadata.sampled_fps} "
          f"schema_version={result.match_metadata.schema_version}")
    print("\nDoD (AnalysisResult valida sin error): OK")


if __name__ == "__main__":
    main()
