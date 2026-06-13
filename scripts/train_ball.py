"""Fine-tune YOLO26 especialista de balon (P1.3b).

Standalone. NO es parte de engine/ (util de un solo uso).
Corre en el venv modo_train (ROCm, RX 9070 XT) o en Colab T4 (contingencia).
Mismo comando en ambos: lo unico que cambia es el venv/maquina.

Uso:
    python scripts/train_ball.py                  # defaults del plan
    python scripts/train_ball.py --batch 4        # 3050 / OOM
    python scripts/train_ball.py --no-amp         # si ROCm da NaN con AMP
    python scripts/train_ball.py --data /content/volleyball.v1i.yolov8/data.yaml

Salida: models/runs/ball_ft/weights/best.pt
Luego:  copiar best.pt -> models/  y correr
        python -m tests.validate_ball --start 30 --end 90
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO / "data/dataset/volleyball.v1i.yolov8/data.yaml"
DEFAULT_MODEL = REPO / "models/yolo26n.pt"


def resolve_data_yaml(path: Path) -> str:
    """Reescribe un data.yaml robusto ignorando rutas rotas del export Roboflow.

    El export de Roboflow trae 'train: ../train/images' que no resuelve.
    Aca se reconstruye desde los dirs reales y se fija 'path' absoluto, asi
    funciona igual en Windows local que en Colab sin tocar nada a mano.
    """
    import yaml

    path = path.resolve()
    src = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent
    out = {
        "path": str(root),
        "nc": src.get("nc", 1),
        "names": src.get("names", ["balon"]),
    }
    for split, key in (("train", "train"), ("valid", "val"), ("test", "test")):
        if (root / split / "images").is_dir():
            out[key] = f"{split}/images"
    if "train" not in out:
        sys.exit(f"ERROR: no encuentro {root}/train/images")
    fixed = root / "_resolved.yaml"
    fixed.write_text(yaml.safe_dump(out, allow_unicode=True), encoding="utf-8")
    return str(fixed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)  # 9070 XT / T4 16GB; 3050 -> 4
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="ball_ft")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        sys.exit(
            "ERROR: sin GPU (torch.cuda.is_available()=False). "
            "Train a imgsz=1280 en CPU = inviable (dias).\n"
            "  - Local: usar venv modo_train con torch ROCm 7.2.1 (ver plan).\n"
            "  - Contingencia: docs/colab-train-ball.ipynb (Colab T4)."
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}  | torch {torch.__version__}")

    from ultralytics import YOLO

    data = resolve_data_yaml(Path(args.data))
    print(f"data: {data}")

    model = YOLO(args.model)
    model.train(
        data=data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=False,
        amp=args.amp,
        project=str(REPO / "models/runs"),
        name=args.name,
    )
    # save_dir real (ultralytics auto-incrementa el nombre si ya existe: ball_ft, ball_ft2...)
    best = Path(model.trainer.save_dir) / "weights/best.pt"
    print(f"\nOK. best.pt -> {best}")
    print("Siguiente: copiar a models\\ y validar.")
    print("  CUDA (3050/prod): python -m tests.validate_ball --start 30 --end 90 --ball-model best.pt")
    print("  ROCm (esta PC):   python scripts\\ball_recall.py  (Detector con 2 modelos se corrompe en ROCm)")


if __name__ == "__main__":
    main()
