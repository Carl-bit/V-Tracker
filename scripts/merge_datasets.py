"""Une varios datasets YOLO (master images/+labels/) en uno solo.

Copia cada src/images/*.jpg y su src/labels/*.txt al --out/images y --out/labels.
Conserva labels vacios (= negativos/fondo). Limpia el --out antes (merge limpio,
no acumula basura de corridas viejas). Despues correr split_dataset sobre --out.

Uso:
  python -m scripts.merge_datasets --out data/dataset/vly_combo \
      --src data/dataset/vly_frontal --src data/dataset/vly_calle --src data/triage_calle
"""
import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(prog="scripts.merge_datasets")
    ap.add_argument("--out", required=True, help="dataset destino (se limpia images/ y labels/)")
    ap.add_argument("--src", action="append", required=True, help="dataset origen (repetible)")
    args = ap.parse_args()

    out = Path(args.out)
    out_img = out / "images"
    out_lbl = out / "labels"
    for d in (out_img, out_lbl):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    # tambien limpiar splits viejos para que split_dataset re-arme de cero
    for old in (out / "train", out / "valid"):
        if old.exists():
            shutil.rmtree(old)

    tot_img = tot_pos = tot_neg = 0
    for src in args.src:
        src = Path(src)
        imgs = sorted((src / "images").glob("*.jpg"))
        if not imgs:
            print(f"AVISO: sin .jpg en {src/'images'} (saltado)")
            continue
        s_pos = s_neg = 0
        for img in imgs:
            if (out_img / img.name).exists():
                print(f"AVISO: nombre repetido {img.name} (de {src}) -> NO se sobreescribe")
                continue
            shutil.copy2(img, out_img / img.name)
            lbl = src / "labels" / f"{img.stem}.txt"
            if lbl.is_file() and lbl.read_text(encoding="utf-8").strip():
                shutil.copy2(lbl, out_lbl / lbl.name)
                s_pos += 1
            else:
                s_neg += 1  # sin label o vacio = negativo/fondo
            tot_img += 1
        tot_pos += s_pos
        tot_neg += s_neg
        print(f"{src.name}: {len(imgs)} imgs ({s_pos} con balon, {s_neg} negativos)")

    print(f"\nTOTAL -> {out}: {tot_img} imgs ({tot_pos} con balon, {tot_neg} negativos)")
    neg_frac = 100.0 * tot_neg / tot_img if tot_img else 0
    print(f"negativos = {neg_frac:.0f}% del set"
          + ("  OJO: >30% puede colapsar la deteccion" if neg_frac > 30 else "  (ok)"))
    print(f"\nsiguiente: .\\cpu.cmd -m scripts.split_dataset --ds {out}")


if __name__ == "__main__":
    main()
