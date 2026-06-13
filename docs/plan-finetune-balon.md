# Plan fine-tune YOLO26 para balon (P1.3b)

Estado: PLAN aprobado, sin ejecutar. Nada descargado.
Baseline P1.3 (yolo26n COCO, conf>=0.15, sample 1/10):
recall ~16.0% en juego activo (30-90s) / 15.2% video completo. Umbral: 60%.

## Arquitectura: modelo especialista de balon

Fine-tunear yolo26n SOLO con clase balon. Mantener yolo26n COCO para personas.
Dos pasadas por frame (offline, ambos nano: aceptable).

Por que no un solo modelo: datasets publicos son ball-only o con labels de
persona flojos; entrenar con ellos destruye la deteccion de persona de COCO.
Especialista ademas deja camino limpio a TrackNet en fase B (drop-in).

## Dataset (Roboflow Universe)

Candidatos (verificar a mano en navegador; Universe bloquea scraping):

1. `primaryws/volleyball_ball_object_detection_dataset` - anotado para balon en
   partidos reales. Candidato principal.
2. `salo-levy-nlqrn/volley-ball-detection` - ~324 imgs. Complemento.
3. `volleyballyolo/volleyballyolo` - ball + person.

Criterio de seleccion, en orden:
- Imagenes de TRANSMISION / plano abierto. Un dataset de primeros planos no
  arregla nada: COCO ya detecta primeros planos (verificado en P1.3).
- >= 1000 imagenes con balon anotado; si no llega, combinar 1+2.
- Licencia permisiva (CC BY 4.0 tipico).

Plan B si ningun dataset tiene plano abierto decente: auto-etiquetar 300-500
frames del propio video con COCO a conf alta + correccion manual en Roboflow
(free tier alcanza). Mas trabajo, datos identicos al dominio.

Formato export: "YOLOv8/YOLO11 TXT" (labels txt + data.yaml, nc=1,
names=[balon]). YOLO26 lo consume directo. Descarga via pip `roboflow` o curl
del zip con API key gratis.

Destino: `data/datasets/volleyball_ball/` (gitignored).

## Comando de train (RTX 3050 6GB)

```
yolo detect train \
  data=data/datasets/volleyball_ball/data.yaml \
  model=models/yolo26n.pt \
  epochs=80 patience=20 \
  imgsz=1280 batch=4 \
  device=0 workers=2 cache=False \
  project=models/runs name=ball_ft
```

Justificacion (reglas duras CLAUDE.md):
- imgsz=1280: regla 6 lo permite "solo si balon lo exige". Lo exige: balon
  <10px a 640 desaparece en el downscale. Es la variable que mas mueve recall.
- batch=4: nano + 1280 + AMP cabe en 6GB con margen. OOM -> batch=2. No subir.
- epochs=80 patience=20: dataset chico converge antes; early stop corta.
  Estimado en 3050: 4-8 h. Correr de noche, worker ARQ parado durante el train.
- cache=False workers=2: i7 4th gen, RAM modesta.
- Augmentations: defaults ultralytics (mosaic/scale ya ayudan a objeto chico).
  No tocar sin baseline del fine-tune.

Donde: VPS con la 3050. La maquina dev no tiene GPU NVIDIA (train CPU inviable).
Alternativa costo-0: Colab free T4, mismo comando, bajar best.pt.

## Integracion post-train (cambio chico, no ahora)

`engine/detect.py`: parametro opcional `ball_model` en Detector. Si esta:
segunda pasada con especialista (su clase 0 -> "balon") y se ignoran los
"balon" de COCO.

## Corroborar mejora

Mismo harness P1.3, cero codigo nuevo:

```
python -m tests.validate_ball --start 30 --end 90   # con especialista integrado
```

- recall >= 60% -> P1.4 (tracking) con especialista.
- recall < 60%  -> tiling/SAHI sobre el especialista; si tampoco -> TrackNet /
  vball-net (INVESTIGACION.md sec 2, opcion 2).

## Pasos

1. [x] Este plan commiteado.
2. [ ] (manual) Verificar candidatos en Roboflow Universe, elegir dataset.
3. [ ] (VPS o Colab) Descargar dataset + train.
4. [ ] Integrar best.pt en Detector + re-correr validate_ball vs baseline 16%.
