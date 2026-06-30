# COMANDOS - VolleyVision (VLY)

Dos caminos:

- **A. DESARROLLO (este PC)**: probar el motor directo, sin Redis ni docker. CPU o
  GPU AMD (ROCm). Iterar rapido.
- **B. VPS (produccion)**: cola async real (Redis + worker ARQ + docker). Esto puede
  esperar; se levanta en la VPS.

## DOS venvs (CPU y GPU AMD) - usar los wrappers cpu.cmd / gpu.cmd

Ambos venvs son **Python 3.12** (igual que tu `python` del sistema y que vly_train).

NO uses `py` ni `python` a secas: en PowerShell resuelven al Python del SISTEMA (sin
deps) -> `ModuleNotFoundError: torch/pydantic`. Para no pelear con eso hay dos
wrappers en la raiz del repo que llaman al python correcto:

```
.\cpu.cmd -m scripts.smoke      # corre en el venv CPU  (modo_ia)
.\gpu.cmd -m scripts.smoke      # corre en el venv GPU AMD (C:\vly_train, ROCm)
```
`cpu.cmd` -> `modo_ia\Scripts\python.exe` (torch 2.12.0+cpu).
`gpu.cmd` -> `C:\vly_train\Scripts\python.exe` (torch 2.9.1+rocm7.2.1, cuda=True).

En este doc, donde diga `python`, reemplaza por `.\cpu.cmd` o `.\gpu.cmd`.

El venv ROCm vive en `C:\vly_train` (ruta SIN espacios: MIOpen no soporta el espacio
de "stats Vol").

> NO vuelvas a correr `python -m venv modo_ia`: re-crea el venv con tu `python` del
> sistema y deja el lanzador inconsistente con los paquetes -> WinError 126 / numpy
> roto. El venv ya esta creado; el paso 0 es UNA sola vez.

VIDEO de prueba del repo: `data/sample/HighlightsMens VNL 2026.mp4` (tiene espacios
-> SIEMPRE entre comillas).

---

## 0. Entorno (UNA sola vez por venv - no repetir)

### CPU (modo_ia)
```
py -3.12 -m venv modo_ia
.\cpu.cmd -m pip install -r requirements.txt
```

### GPU AMD (vly_train, ROCm)
```
.\scripts\setup_modo_train.ps1      # crea C:\vly_train con torch ROCm + ultralytics
# completar deps de inferencia (pydantic/fastapi/arq/lap/...):
.\gpu.cmd -m pip install lap==0.5.13 fastapi==0.136.3 pydantic==2.13.4 "uvicorn[standard]==0.49.0" python-multipart==0.0.32 arq==0.28.0 redis==5.3.1
```

### Chequear que cada venv ve la GPU
```
.\cpu.cmd -c "import torch; print(torch.__version__, 'gpu:', torch.cuda.is_available())"   # -> +cpu gpu: False
.\gpu.cmd -c "import torch; print(torch.__version__, 'gpu:', torch.cuda.is_available())"   # -> +rocm7.2.1 gpu: True
```
En los logs del motor: `device=cuda:0 half=True` (GPU) o `device=cpu half=False` (CPU).

GPU AMD (ROCm 7.2.1 Windows preview) - RESUELTO: el FP16 (half=True) de MIOpen esta
roto (1a inferencia ok, el resto devuelve 0 cajas). En FP32 (half=False) es estable y
correcto. El motor ahora auto-desactiva half en ROCm (detecta `torch.version.hip`) y
lo mantiene en CUDA NVIDIA (VPS, regla 6). En GPU sale TODO igual que en CPU.
Override manual: `--half on|off|auto` (CLI) o `VLY_HALF` (worker).

Benchmark (ventana 0-30s, sample 5, mismas detecciones): CPU ~24s vs GPU AMD ~13s.

---

# A. DESARROLLO (este PC) - sin Redis, sin docker

Todo `engine/` corre standalone. Esto es lo que usas mientras pruebas rendimiento
CPU vs GPU AMD.

## A.1 Probar TODO de una pasada (recomendado)

Una sola corrida de deteccion valida balon + jugadores + eventos + contrato:
```
.\cpu.cmd -m scripts.smoke
```
Default ventana 25-40s, sample 3. Termina en `SMOKE: OK`. Otra ventana/video:
```
.\cpu.cmd -m scripts.smoke "data/sample/HighlightsMens VNL 2026.mp4" 3 25 40
```

## A.2 Generar el JSON real (CLI del motor)

```
# ventana corta: validar rapido (ideal en CPU)
.\cpu.cmd -m engine.run --video "data/sample/HighlightsMens VNL 2026.mp4" --out out.json --model yolo26n.pt --sample 10 --start 25 --end 40

# video largo COMPLETO: saca todo el material (lento en CPU, ok en GPU)
.\cpu.cmd -m engine.run --video "data/mi_partido.mp4" --out partido.json --model yolo26n.pt --sample 10
```
Genera `out.json` (contrato, coords 0-1) + `out_ball_track.json` (trayectoria densa
aparte). Progreso (frames, %) por stderr. Exit code != 0 si falla.

Flags: `--sample N` (1 de cada N; 10 rapido, 3 eventos mas finos), `--start/--end`
(segundos), `--model`, `--ball-model`, `--job-id`.

## A.3 Medir rendimiento CPU vs GPU

Flag `--device` fuerza el backend (fiable en AMD/ROCm; no depende de variables de
entorno). Misma ventana, comparar el tiempo final que loguea `engine.run`
(`OK Ns -> out.json`):
```
# CPU forzado
.\cpu.cmd -m engine.run --video "data/sample/HighlightsMens VNL 2026.mp4" --out cpu.json --sample 10 --start 0 --end 30 --device cpu

# GPU forzado (AMD ROCm o NVIDIA)
.\gpu.cmd -m engine.run --video "data/sample/HighlightsMens VNL 2026.mp4" --out gpu.json --sample 10 --start 0 --end 30 --device cuda
```
El log dice `device=cuda|cpu half=...` para confirmar. `--device auto` (default)
detecta GPU solo.

## A.4 Checks por etapa (debug; cada uno re-detecta = mas lento)

```
.\cpu.cmd -m tests.check_track    # trayectoria balon continua + ids jugadores
.\cpu.cmd -m tests.check_events   # eventos: saque + remate
.\cpu.cmd -m tests.check_export   # AnalysisResult valida el JSON
```

## A.5 DoD del worker SIN infra (logica de la cola, sin Redis)

Usa un Redis en memoria; ejercita la tarea real `analyze_video` end-to-end:
```
.\cpu.cmd -m tests.check_worker
```
Termina en `DoD ... OK`: status `procesando -> completado` + result valido en Redis.
Sirve para validar la Fase 2 en este PC antes de tener Redis.

## A.6 Probar el motor con CUALQUIER video (receta generica)

Para meter un .mp4 nuevo y ver si el trackeo sale bien. Dos artefactos: el JSON
del contrato y un `overlay.mp4` para juzgar con el ojo (balon + estela + eventos).

Pasos (todos los videos viven en `data/sample/`, rutas con espacios -> comillas):

1) Correr el motor sobre una VENTANA (un rally, ~15-40s). En CPU acotar SIEMPRE
   con `--start/--end`; el video entero solo en GPU. Elegir el `--ball-model`
   segun la vista (ver tabla abajo).
2) Pintar el overlay de esa misma ventana (no re-infiere, sin GPU, rapido).
3) Abrir el `overlay.mp4` y mirar: circulo lleno = balon detectado, hueco =
   interpolado. Estela continua = trackeo ok. Banners = eventos.
4) Leer el log de `engine.run`: `balon=N pts`, `eventos=[...]`, `cortes de escena`.

Que modelo de balon usar:

```
vista          video de prueba           --ball-model       --roi
-----------    -----------------------   ----------------   --------------------
frontal        VODME-vs-Mamba.mp4        ball_best.pt       no (camara unica)
panoramica     FINAL-LIVOME-2024.mp4     ball_best.pt       si (roi.json) recom.
```
`ball_best.pt` es el generalista (frontal + Calle Larga). `ball_frontal2.pt` es el
especialista solo-frontal: alternativa si en frontal el generalista mete falsos
positivos. Panoramica NO esta en el set de entrenamiento del balon -> esperar
recall bajo; es una PRUEBA de hasta donde llega, no un caso resuelto (regla 1).

### Frontal: VODME-vs-Mamba

```
# 1) motor, ventana corta de prueba (ajustar --start/--end al rally que quieras)
.\cpu.cmd -m engine.run --video "data/sample/VODME-vs-Mamba.mp4" --out vodme.json --ball-model ball_best.pt --sample 10 --start 60 --end 90

# 2) overlay de la MISMA ventana
.\cpu.cmd -m scripts.overlay --video "data/sample/VODME-vs-Mamba.mp4" --json vodme.json --start 60 --end 90 --out vodme_overlay.mp4
```
Si ves falsos positivos del balon (luces, lineas), repetir el paso 1 con
`--ball-model ball_frontal2.pt` y comparar.

### Panoramica: FINAL-LIVOME-2024

La vista abierta muestra publico/marcador fuera de la cancha -> conviene ROI para
descartar detecciones de balon ahi. Marcar el poligono UNA vez (clic vertices,
guarda json), luego pasarlo a las dos etapas con `--roi`:
```
# 0) (una vez) marcar la cancha -> genera roi_livome.json
.\cpu.cmd -m scripts.pick_roi --video "data/sample/FINAL-LIVOME-2024.mp4" --out roi_livome.json

# 1) motor con ROI
.\cpu.cmd -m engine.run --video "data/sample/FINAL-LIVOME-2024.mp4" --out livome.json --ball-model ball_best.pt --roi roi_livome.json --sample 10 --start 60 --end 90

# 2) overlay con el mismo ROI dibujado (linea azul = zona valida)
.\cpu.cmd -m scripts.overlay --video "data/sample/FINAL-LIVOME-2024.mp4" --json livome.json --roi roi_livome.json --start 60 --end 90 --out livome_overlay.mp4
```
Sin ROI tambien corre (omitir `--roi` en ambos pasos); solo filtra menos ruido.

### Video COMPLETO (cuando la ventana ya se ve bien)

Quitar `--start/--end`. Lento en CPU; usar `.\gpu.cmd` para el partido entero:
```
.\gpu.cmd -m engine.run --video "data/sample/VODME-vs-Mamba.mp4" --out vodme.json --ball-model ball_best.pt --sample 10
```
El overlay del partido entero pesa: acotarlo por tramos (`--start/--end`) o subir
`--stride` para aligerar.

---

# B. VPS (produccion) - cola async real

La API NO procesa el video: encola un job y un worker SEPARADO lo corre (regla 3:
1 job en GPU a la vez). Estado y resultado viven en Redis.

## B.1 Levantar Redis

VPS (Linux) con docker:
```
docker run -d --name vly-redis -p 6379:6379 redis:7
```
(En este PC sin docker, si alguna vez queres probar el path real: Memurai
https://www.memurai.com deja Redis en localhost:6379, o usar WSL `redis-server`.)

Config por env (defaults localhost:6379 db0):
```
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

## B.2 Levantar el worker (dejar corriendo)

```
arq worker.tasks.WorkerSettings
```
1 worker, 1 job a la vez. `job_timeout` 6h (aguanta video de 1.5h).

## B.3 Encolar y seguir un job

```
.\cpu.cmd -m scripts.enqueue_job "data/mi_partido.mp4" mi_job 10      # video, job_id, sample
.\cpu.cmd -m scripts.job_status mi_job                                # status: en_cola|procesando|N%|completado|error
.\cpu.cmd -m scripts.job_status mi_job --result                       # vuelca el JSON del contrato
```

Claves en Redis:
```
vly:job:{job_id}:status      JSON {status, progress}
vly:job:{job_id}:result      JSON AnalysisResult (contrato)
vly:job:{job_id}:ball_track  JSON trayectoria densa (aparte)
```

Env del worker: `VLY_SAMPLE` (default 10), `VLY_MODEL`, `VLY_BALL_MODEL`,
`VLY_RESULT_TTL`, `VLY_DELETE_SOURCE=1` (borra el .mp4 tras generar JSON, regla 8).

## B.4 API (mediador) - hoy mock

```
uvicorn api.main:app --reload      # docs en http://localhost:8000/docs
```
Hoy devuelve el mock del contrato. Wirear `/upload` -> encolar y `/status` `/results`
-> leer de Redis es Fase 3.

---

## Notas

- El motor decide GPU/CPU solo (`torch.cuda.is_available()`); ROCm AMD aparece como
  `cuda`. Forzar con `--device cpu|cuda` (CLI) o `VLY_DEVICE` (worker).
- Velocidades en km/h son aprox (sin homografia; constante `COURT_VIEW_M` en
  engine/export.py).
- Material de un video largo: `engine.run` completo con `--sample 10`. Bajar a
  `--sample 5/3` da mas densidad a costa de tiempo.
- `engine/` no importa FastAPI ni ARQ. El worker llama al motor como funcion, no por
  subprocess. Mismo motor en dev (A) y en la cola (B).
