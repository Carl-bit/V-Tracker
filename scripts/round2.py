"""Staging para anotar SOLO los frames nuevos (iteracion 2), sin recorrer los viejos.

stage : mueve los frames nuevos (los que no estan en el split round-1 train/valid)
        + sus pre-labels a _round2/, dejando images/ y labels/ con solo lo de round-1.
        Anotas _round2/images en labelImg (rapido, solo lo nuevo).
merge : devuelve _round2/ (imagenes + labels corregidos) a images/ y labels/.

Uso:
  python -m scripts.round2 stage
  # ... anotar en labelImg ...
  python -m scripts.round2 merge
"""

import argparse
import shutil
from pathlib import Path


def _names(d: Path) -> set[str]:
    return {p.name for p in d.glob("*.jpg")} if d.exists() else set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stage", "merge"])
    ap.add_argument("--ds", default="data/dataset/vly_frontal")
    args = ap.parse_args()
    ds = Path(args.ds)
    images, labels, prev = ds / "images", ds / "labels", ds / "_preview"
    r2 = ds / "_round2"
    r2i, r2l = r2 / "images", r2 / "labels"

    if args.mode == "stage":
        round1 = _names(ds / "train" / "images") | _names(ds / "valid" / "images")
        if not round1:
            raise SystemExit("no encuentro el split round-1 (train/valid). Corre split_dataset antes.")
        new = [p for p in sorted(images.glob("*.jpg")) if p.name not in round1]
        r2i.mkdir(parents=True, exist_ok=True)
        r2l.mkdir(parents=True, exist_ok=True)
        moved, with_lbl = 0, 0
        for img in new:
            shutil.move(str(img), r2i / img.name)
            lbl = labels / f"{img.stem}.txt"
            if lbl.is_file():
                shutil.move(str(lbl), r2l / lbl.name)
                with_lbl += 1
            pv = prev / img.name
            if pv.is_file():
                pv.unlink()
            moved += 1
        (r2 / "classes.txt").write_text("balon\n", encoding="utf-8")
        print(f"staged {moved} frames nuevos a {r2i} ({with_lbl} con pre-label, {moved-with_lbl} sin)")
        print(f"images/ queda con {len(_names(images))} (round-1)")
        print("\nAnotar SOLO lo nuevo:")
        print(f"  modo_ia\\Scripts\\labelImg.exe {r2i} {r2/'classes.txt'}")
        print("  (Save Dir -> _round2\\labels, formato YOLO). Luego: python -m scripts.round2 merge")

    else:  # merge
        if not r2i.exists():
            raise SystemExit(f"no existe {r2i} (nada que fusionar)")
        back_i = back_l = 0
        for img in sorted(r2i.glob("*.jpg")):
            shutil.move(str(img), images / img.name)
            back_i += 1
        for lbl in sorted(r2l.glob("*.txt")):
            shutil.move(str(lbl), labels / lbl.name)
            back_l += 1
        shutil.rmtree(r2)
        print(f"fusionados {back_i} imagenes y {back_l} labels a {ds}")
        print(f"images/ total: {len(_names(images))}")
        print("\nSiguiente: python -m scripts.check_dataset ; python -m scripts.split_dataset")


if __name__ == "__main__":
    main()
