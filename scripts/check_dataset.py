"""Valida el dataset YOLO de balon: formato de labels, negativos, archivos sueltos.

Uso: python -m scripts.check_dataset [--ds data/dataset/vly_frontal]
Reporta: positivos/negativos, cajas, labels invalidos, y .xml/.txt mal ubicados
en images/ (de cuando labelImg guardaba en el dir/formato equivocado).
"""

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="data/dataset/vly_frontal")
    args = ap.parse_args()
    ds = Path(args.ds)
    img_dir, lbl_dir = ds / "images", ds / "labels"

    images = sorted(img_dir.glob("*.jpg"))
    stray_xml = sorted(img_dir.glob("*.xml"))
    stray_txt = sorted(img_dir.glob("*.txt"))

    positives, negatives, boxes, bad = 0, 0, 0, []
    for img in images:
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.is_file() or not lbl.read_text(encoding="utf-8").strip():
            negatives += 1
            continue
        positives += 1
        for ln, line in enumerate(lbl.read_text(encoding="utf-8").splitlines(), 1):
            p = line.split()
            if len(p) != 5:
                bad.append(f"{lbl.name}:{ln} -> {len(p)} campos (esperaba 5)")
                continue
            cls = p[0]
            vals = [float(x) for x in p[1:]]
            if cls != "0":
                bad.append(f"{lbl.name}:{ln} -> clase {cls} (esperaba 0)")
            if any(not (0.0 <= v <= 1.0) for v in vals):
                bad.append(f"{lbl.name}:{ln} -> coords fuera de 0-1: {vals}")
            if vals[2] <= 0 or vals[3] <= 0:
                bad.append(f"{lbl.name}:{ln} -> w/h <= 0")
            boxes += 1

    orphan_txt = [t for t in stray_txt if not (lbl_dir / t.name).is_file()]

    print(f"dataset: {ds}")
    print(f"imagenes:           {len(images)}")
    print(f"  con balon (pos):  {positives} ({boxes} cajas)")
    print(f"  sin balon (neg):  {negatives}")
    print(f"labels invalidos:   {len(bad)}")
    for b in bad[:20]:
        print(f"    {b}")
    print(f"sueltos en images/: {len(stray_xml)} .xml, {len(stray_txt)} .txt")
    if orphan_txt:
        print(f"  .txt en images/ SIN copia en labels/ (anotacion en riesgo): {len(orphan_txt)}")
        for t in orphan_txt:
            print(f"    {t.name}")
    ok = not bad
    print(f"\nFORMATO: {'OK' if ok else 'REVISAR'}")


if __name__ == "__main__":
    main()
