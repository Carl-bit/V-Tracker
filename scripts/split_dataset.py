"""Split train/val del dataset YOLO (copia images+labels a train/ y valid/).

Deja el master images/ + labels/ intacto y crea train/{images,labels} y
valid/{images,labels} + data.yaml, listo para scripts/train_ball.py.
Los negativos (imagenes sin label) se reparten igual (son fondo).

Uso: python -m scripts.split_dataset [--ds ...] [--val-frac 0.15] [--seed 42]
"""

import argparse
import random
import shutil
from pathlib import Path


def _clear(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="data/dataset/vly_frontal")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = Path(args.ds)
    images = sorted((ds / "images").glob("*.jpg"))
    if not images:
        raise SystemExit(f"sin imagenes en {ds/'images'}")

    random.seed(args.seed)
    random.shuffle(images)
    n_val = max(1, round(len(images) * args.val_frac))
    splits = {"valid": images[:n_val], "train": images[n_val:]}

    _clear(ds / "train")  # re-split limpio (evita mezclar con split anterior)
    _clear(ds / "valid")
    for split, imgs in splits.items():
        (ds / split / "images").mkdir(parents=True, exist_ok=True)
        (ds / split / "labels").mkdir(parents=True, exist_ok=True)
        pos = 0
        for img in imgs:
            shutil.copy2(img, ds / split / "images" / img.name)
            lbl = ds / "labels" / f"{img.stem}.txt"
            if lbl.is_file() and lbl.read_text(encoding="utf-8").strip():
                shutil.copy2(lbl, ds / split / "labels" / lbl.name)
                pos += 1
        print(f"{split}: {len(imgs)} imgs ({pos} con balon, {len(imgs)-pos} negativos)")

    (ds / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: valid/images\nnc: 1\nnames: ['balon']\n",
        encoding="utf-8",
    )
    print(f"\ndata.yaml -> {ds/'data.yaml'}")
    print("entrenar (GPU):")
    print(f"  .\\gpu.cmd scripts/train_ball.py --data {ds/'data.yaml'} --model models/ball_best.pt --no-amp --device 0 --name ball_frontal2")


if __name__ == "__main__":
    main()
