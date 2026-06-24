"""Triage manual del dump de ball_recall: vos decidis TP vs FP frame a frame.

El conf NO separa balon de basura (un FP de piso llega a 0.78), asi que la unica
verdad la pone el ojo humano. Este tool te muestra cada deteccion (caja amarilla)
y con una tecla la clasificas. Al salir arma un dataset YOLO listo para mergear:

  - balon OK (b)  -> positivo: etiqueta con la caja del modelo (no encajonas a mano)
  - falso pos (f) -> NEGATIVO: imagen + label vacio (esto mata los FP al reentrenar)
  - mixto/duda(m) -> a carpeta mixed/ para corregir en LabelImg

Resume: relee triage.jsonl, no repetis lo ya marcado.

Uso:
  python -m scripts.triage_dump --dump out/fp_calle --out data/triage_calle
  # solo revisar (sin escribir dataset):
  python -m scripts.triage_dump --dump out/fp_calle --no-build

Teclas: [b]alon OK  [f]also positivo  [m]ixto/duda  [space]=siguiente
        [a]=atras  [u]=borra verdicto  [q]/esc=guardar y salir
"""
import argparse
import json
import shutil
from pathlib import Path


def _yolo_lines(boxes, w, h, cls):
    """xyxy abs -> lineas YOLO 'cls xc yc bw bh' normalizadas 0-1."""
    out = []
    for x1, y1, x2, y2, _c in boxes:
        xc = (x1 + x2) / 2 / w
        yc = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        out.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="scripts.triage_dump")
    ap.add_argument("--dump", required=True, help="dir generado por ball_recall --dump")
    ap.add_argument("--out", default=None, help="dir dataset de salida (images/ labels/ mixed/)")
    ap.add_argument("--cls", type=int, default=0, help="id de clase del balon para labels YOLO")
    ap.add_argument("--no-build", action="store_true", help="solo triage, no escribir dataset")
    args = ap.parse_args()

    import cv2

    dump = Path(args.dump)
    manifest_path = dump / "manifest.jsonl"
    if not manifest_path.is_file():
        raise SystemExit(f"falta {manifest_path}: re-corre ball_recall --dump con el script nuevo")
    items = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    items.sort(key=lambda d: d["name"])  # conf asc: FPs candidatos arriba

    verdict_path = dump / "triage.jsonl"
    verdicts: dict[str, str] = {}
    if verdict_path.is_file():
        for l in verdict_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                d = json.loads(l)
                verdicts[d["name"]] = d["verdict"]
        print(f"resume: {len(verdicts)} ya marcados")

    KEY = {ord("b"): "tp", ord("f"): "fp", ord("m"): "mixed"}
    win = "triage  [b]alon  [f]also  [m]ixto  space=sig  a=atras  u=undo  q=salir"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    i = 0
    n = len(items)
    while 0 <= i < n:
        it = items[i]
        name = it["name"]
        img = cv2.imread(str(dump / f"{name}.jpg"))  # anotado (con caja)
        if img is None:
            i += 1
            continue
        v = verdicts.get(name, "-")
        counts = {x: sum(1 for q in verdicts.values() if q == x) for x in ("tp", "fp", "mixed")}
        bar = f"{i+1}/{n}  {name}  verdicto={v}  | TP={counts['tp']} FP={counts['fp']} MIX={counts['mixed']}"
        cv2.putText(img, bar, (10, img.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(img, bar, (10, img.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(win, img)
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k in KEY:
            verdicts[name] = KEY[k]
            i += 1
        elif k in (ord(" "), ord("n")):
            i += 1
        elif k in (ord("a"),):
            i = max(0, i - 1)
        elif k == ord("u"):
            verdicts.pop(name, None)
    cv2.destroyAllWindows()

    # persistir verdictos
    with verdict_path.open("w", encoding="utf-8") as fh:
        for it in items:
            if it["name"] in verdicts:
                fh.write(json.dumps({"name": it["name"], "verdict": verdicts[it["name"]]}) + "\n")
    c = {x: sum(1 for q in verdicts.values() if q == x) for x in ("tp", "fp", "mixed")}
    print(f"triage: TP={c['tp']} FP={c['fp']} MIX={c['mixed']} (de {n}) -> {verdict_path}")

    if args.no_build or args.out is None:
        return

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "mixed").mkdir(parents=True, exist_ok=True)
    by_name = {it["name"]: it for it in items}
    n_pos = n_neg = n_mix = 0
    for name, v in verdicts.items():
        it = by_name.get(name)
        if it is None:
            continue
        clean = dump / "clean" / f"{name}.jpg"
        if not clean.is_file():
            continue
        if v == "mixed":
            shutil.copy(clean, out / "mixed" / f"{name}.jpg")
            n_mix += 1
            continue
        shutil.copy(clean, out / "images" / f"{name}.jpg")
        label = out / "labels" / f"{name}.txt"
        if v == "tp":
            label.write_text("\n".join(_yolo_lines(it["boxes"], it["w"], it["h"], args.cls)) + "\n",
                             encoding="utf-8")
            n_pos += 1
        else:  # fp -> negativo: label vacio
            label.write_text("", encoding="utf-8")
            n_neg += 1
    print(f"dataset -> {out}  | positivos={n_pos}  negativos={n_neg}  mixed(LabelImg)={n_mix}")


if __name__ == "__main__":
    main()
