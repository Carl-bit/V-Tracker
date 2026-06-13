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

Candidato elegido por el usuario:

0. `actions-players/volleyball-actions` - PENDIENTE verificar en navegador que
   tenga clase balon anotada (por el nombre parece dataset de acciones:
   remate/bloqueo/etc). Si no tiene balon como clase, NO sirve para esto.

Alternativos (verificar a mano; Universe bloquea scraping):

1. `primaryws/volleyball_ball_object_detection_dataset` - anotado para balon en
   partidos reales.
2. `salo-levy-nlqrn/volley-ball-detection` - ~324 imgs. Complemento.
3. `volleyballyolo/volleyballyolo` - ball + person.

Para descargar se necesita UNO de estos (cuenta Roboflow gratis):
- API key del usuario (Settings -> API) -> descarga via pip `roboflow`, o
- descarga manual del zip en el navegador (export "YOLOv8" / TXT) y dejarlo en
  `data/datasets/`.

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

## Donde entrenar: PC dev con RX 9070 XT (local, costo 0)

La PC dev tiene Radeon RX 9070 XT (16GB). AMD publica PyTorch oficial para
Windows (ROCm 7.2.1) con la 9070 XT soportada. Eso habilita train local.
Con 16GB VRAM: subir a batch=8 en imgsz=1280 (el comando de arriba es para la
3050; en la 9070 XT mismo comando con batch=8).

Prerequisitos (una vez):
1. Driver Adrenalin >= 26.2.2 (verificar/actualizar a mano).
2. Python 3.12 (las wheels son cp312; en la PC solo hay 3.13):
   `winget install Python.Python.3.12`
3. Venv SEPARADO de train (no tocar modo_ia, que queda CPU para el pipeline):
   `py -3.12 -m venv modo_train`
4. Instalar ROCm SDK + torch AMD en modo_train (CMD):
   ```
   pip install --no-cache-dir ^
     https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl ^
     https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl ^
     https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl ^
     https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz
   pip install --no-cache-dir ^
     https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl ^
     https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl
   pip install ultralytics --no-deps && pip install <deps de ultralytics sin torch>
   ```
5. Smoke test: `python -c "import torch; print(torch.cuda.is_available())"`
   (ROCm se expone como cuda en torch).

Riesgo: stack PyTorch-Windows-AMD es reciente (preview 2025, estable 2026).
Si el train falla (op no soportada, NaN con AMP), fallback inmediato costo-0:
Colab free T4, mismo comando, bajar best.pt. No pelearse dias con ROCm.

OJO: el train local NO cambia el target de inferencia. El .pt resultante corre
igual en la 3050 del VPS (regla 6: half=True ahi). En esta PC la inferencia de
validacion corre por CPU u GPU AMD, da lo mismo: lo que se mide es recall.

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
2. [ ] (manual, usuario) Verificar que actions-players/volleyball-actions tenga
       clase balon; pasar API key de Roboflow o el zip exportado (YOLOv8/TXT).
3. [ ] (manual, usuario) Driver Adrenalin >= 26.2.2.
4. [ ] Setup modo_train (py 3.12 + torch ROCm 7.2.1) + smoke test GPU.
5. [ ] Train local en la 9070 XT (batch=8, imgsz=1280). Fallback: Colab T4.
6. [ ] Integrar best.pt en Detector + re-correr validate_ball vs baseline 16%.
