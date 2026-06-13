# Plan fine-tune YOLO26 para balon (P1.3b)

Estado: HECHO. Entrenado en RX 9070 XT (ROCm), 80 epochs / 2.1h, batch=8,
imgsz=1280. Metricas valid del dataset: mAP50 0.946, P 0.955, R 0.912.
RECALL EN EL VIDEO REAL (30-90s, conf>=0.25, ball solo): 99.4% (180/181).
Baseline COCO P1.3 era ~16%. Umbral 60% SUPERADO con margen enorme.
-> P1.3b cerrado. Sigue P1.4 (tracking) con el especialista.

Pesos: models/ball_best.pt (copiado de models/runs/ball_ft-2/weights/best.pt).

CAVEAT ROCm (solo dev AMD, NO prod): cargar 2 modelos YOLO en un proceso
corrompe la inferencia en gfx1201 (COCO da 0 personas, ball da ruido conf ~0.1).
Es bug del stack ROCm preview (autotuner MIOpen colisiona entre modelos), NO del
modelo. En CUDA (RTX 3050, target real) NO pasa. Por eso en esta PC se valida con
`scripts/ball_recall.py` (ball solo); en el 3050 vale
`python -m tests.validate_ball --ball-model ball_best.pt` (Detector con 2 pasadas).

## Arquitectura: modelo especialista de balon

Fine-tunear yolo26n SOLO con clase balon. Mantener yolo26n COCO para personas.
Dos pasadas por frame (offline, ambos nano: aceptable).

Por que no un solo modelo: datasets publicos son ball-only o con labels de
persona flojos; entrenar con ellos destruye la deteccion de persona de COCO.
Especialista ademas deja camino limpio a TrackNet en fase B (drop-in).

## Dataset (Roboflow Universe) - VERIFICADO

El usuario bajo a mano dos datasets a `data/dataset/`:

- `volleyball.v1i.yolov8` -> ESTE es el bueno. nc=1, names=['Volleyball'].
  1138 train / 149 valid / 286 test (1573 total). 1 label vacio (ignorable).
  Supera el umbral >=1000 del plan. Tamano de box (norm): mediana ~0.077
  (balon chico, plano de TRANSMISION = el dominio que importa); ~37% de boxes
  >10% ancho (primeros planos, no estorban pero COCO ya los detecta). Mezcla
  sana. APTO para la prueba, sin necesidad de mas datos de entrada.
- `Volleyball Actions.v5` -> DESCARTADO para balon. nc=5
  (block/defense/serve/set/spike). NO tiene clase balon. Resuelve la duda del
  candidato `volleyball-actions`: era dataset de acciones. Util a futuro para
  clasificar eventos (backlog), no aca.

API key Roboflow del usuario: NO se necesita para esta prueba (los datasets ya
estan bajados). Solo serviria para tirar MAS datasets via pip `roboflow`. Si se
usa va en `.env` (gitignored), nunca commiteada.

Plan-data si v1i no alcanza (verificar a mano; Universe bloquea scraping):

1. `primaryws/volleyball_ball_object_detection_dataset` - balon en partidos
   reales. Combinar con v1i para pasar de 175 imgs.
2. `salo-levy-nlqrn/volley-ball-detection` - ~324 imgs. Complemento.
3. Auto-etiquetar 300-500 frames del propio video (COCO conf alta + correccion
   manual en Roboflow free). Datos identicos al dominio.

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

## Comando de train

Envuelto en `scripts/train_ball.py` (portable: mismo script en ROCm local y
Colab; reconstruye el data.yaml solo, ignora el '../' roto de Roboflow):

```
python scripts/train_ball.py                  # defaults del plan (batch=8)
python scripts/train_ball.py --batch 4        # 3050 / si hay OOM
python scripts/train_ball.py --no-amp         # si ROCm da NaN con AMP
```

Equivale a: yolo detect train data=<v1i>/data.yaml model=models/yolo26n.pt
epochs=80 patience=20 imgsz=1280 batch=8 device=0 workers=2 cache=False
project=models/runs name=ball_ft

Justificacion (reglas duras CLAUDE.md):
- imgsz=1280: regla 6 lo permite "solo si balon lo exige". Lo exige: balon
  <10px a 640 desaparece en el downscale. Es la variable que mas mueve recall.
- batch=8: train corre en 9070 XT (16GB) o T4 (16GB), no en la 3050. Cabe.
  Si alguna vez se entrena en 3050 -> --batch 4 (OOM -> 2). No subir de 8.
- epochs=80 patience=20: 1138 imgs; early stop corta si se estanca.
  En T4/9070 XT, ~minutos por epoch a imgsz=1280.
- cache=False workers=2: conservador, RAM modesta.
- Augmentations: defaults ultralytics (mosaic/scale ya ayudan a objeto chico).
  No tocar sin baseline del fine-tune.
- El .pt resultante corre inferencia igual en la 3050 del VPS (half=True ahi).

## VIA PRINCIPAL: train local, PC dev con RX 9070 XT (costo 0)

La PC dev tiene Radeon RX 9070 XT (16GB). AMD publica PyTorch oficial para
Windows (ROCm 7.2.1) con la 9070 XT soportada. Eso habilita train local.
Con 16GB VRAM: batch=8 en imgsz=1280 (default del script).

Prerequisitos:
1. (manual) Driver Adrenalin >= 26.2.2.
2. (manual) Python 3.12 (las wheels son cp312; 3.13 NO sirve):
   `winget install Python.Python.3.12`
3. (manual) CRITICO: Visual Studio Build Tools 2022 con workload C++. MIOpen
   compila kernels HIP en runtime con clang y en Windows toma la STL de MSVC;
   sin esto -> "'type_traits' file not found" -> miopenStatusUnknownError al
   primer batch. Instalar:
   ```
   winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   ```
4. Todo lo demas en un script (venv 3.12 + ROCm SDK + torch/torchvision + ultra
   + smoke test). Desde la RAIZ del repo, en PowerShell:
   ```
   .\scripts\setup_modo_train.ps1            # venv en C:\vly_train
   ```
5. Disparar (python del venv + ruta del script entre comillas por el espacio):
   ```
   & "C:\vly_train\Scripts\python.exe" "$PWD\scripts\train_ball.py"
   ```

Nota ruta sin espacios (menor): el venv en C:\vly_train evita un warning
cosmetico de `offload-arch.exe` por el espacio en "stats Vol". NO era la causa
del fallo de MIOpen (eso es MSVC, paso 3). La ruta de datos puede tener espacio.
No mover un venv ya creado (rompe los launchers .exe); recrear con el script.

OJO sintaxis: el setup va en `setup_modo_train.ps1` porque los comandos pip
crudos del repo de AMD usan continuacion CMD (`^`) y `&&`, que PowerShell NO
entiende. No copiarlos a mano en PS; correr el script.

Riesgo: stack PyTorch-Windows-AMD es reciente (preview 2025, estable 2026).
Si el train falla (op no soportada, NaN con AMP) probar `--no-amp`; si sigue
fallando -> via de contingencia (abajo). No pelearse dias con ROCm.

OJO: el train local NO cambia el target de inferencia. El .pt resultante corre
igual en la 3050 del VPS (regla 6: half=True ahi). En esta PC la inferencia de
validacion corre por CPU u GPU AMD, da lo mismo: lo que se mide es recall.

## VIA DE CONTINGENCIA: Colab T4 (solo si la principal falla)

Notebook listo: `docs/colab-train-ball.ipynb`. Turnkey, no hay que re-investigar:
1. Subir el notebook a Colab. Runtime -> GPU (T4).
2. Correr celdas en orden: verifica GPU, instala ultralytics, sube el zip del
   dataset (`volleyball.v1i.yolov8`), entrena (mismos params: imgsz=1280
   batch=8), baja `best.pt`.
3. Copiar `best.pt` a `models/` del repo.

Mismos hiperparametros que el script local -> resultado comparable.

## Integracion post-train - HECHA

`engine/detect.py`: Detector ya acepta `ball_model` opcional. Si se pasa:
2da pasada con el especialista (toda deteccion -> "balon", imgsz=1280) y COCO
queda solo para persona (se ignora su balon clase 32). Sin especialista =
comportamiento viejo intacto. Listo para drop-in de TrackNet en fase B.

## Corroborar mejora - HECHO

Resultado: recall 99.4% (30-90s, conf>=0.25). Supera el umbral 60% holgado.

En CUDA (3050/prod) el harness completo:
```
python -m tests.validate_ball --start 30 --end 90 --ball-model ball_best.pt
```
En ROCm (esta PC) NO sirve el Detector con 2 modelos (ver CAVEAT arriba); validar
con el ball solo:
```
python scripts/ball_recall.py --start 30 --end 90
```

## Pasos

1. [x] Plan commiteado.
2. [x] Datasets verificados: v1i = balon (plano abierto, 1573 imgs, supera
       umbral); Actions v5 sin clase balon (descartado). API key no hace falta.
3. [x] data.yaml de v1i corregido (rutas), script de train portable, notebook
       Colab, integracion ball_model en Detector + flag en validate_ball.
4. [x] (manual, usuario) Driver Adrenalin + Python 3.12 + VS Build Tools (C++).
5. [x] Setup modo_train (venv C:\vly_train, torch ROCm 7.2.1) + smoke test GPU OK.
6. [x] Entrenado: scripts/train_ball.py, 80 epochs / 2.1h en RX 9070 XT.
7. [x] best.pt -> models/ball_best.pt. Recall 99.4% (vs 16% baseline). P1.3b OK.
