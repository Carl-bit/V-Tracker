# CLAUDE.md - VolleyVision (VLY)

Guia agente. Leer antes de tocar codigo. Estilo caveman: boca chica, cerebro grande. ASCII only. Sin emoji.

## Estilo respuesta

- Output terso. Cero preambulo. Cero relleno. Cero halago.
- Idea mala = decir mala + dar alternativa.
- Codigo primero. Explicar minimo, solo si aporta.
- Codigo que funciona = no reescribir "para mejorar" sin pedido.
- Preguntar solo si ambiguedad bloquea. Si no, asumir razonable, seguir.
- ASCII. Flecha = ->. Sin unicode decorativo.

## Que es

Analisis video voley. In: .mp4. Out: JSON con balon trackeado + metricas + eventos (saque/recepcion/armado/remate). Front React/Next aparte. Proceso OFFLINE async. No real-time.

## Hardware (VPS)

i7 4th gen + RTX 3050 LP, 6GB VRAM, Ubuntu/Debian.
=> YOLO nano/small. FP16 (half=True). batch 1-4. 1 worker GPU concurrency=1.
No asumir VRAM sobra. No proponer medium/large ni multi-GPU.

## Stack

- Inferencia: python, ultralytics (YOLO26), opencv-python, numpy, ByteTrack. Fase B: TrackNet/torch para balon.
- API: FastAPI + Pydantic + uvicorn.
- Cola/estado: ARQ + Redis, worker separado. Jobs NUNCA en BackgroundTasks.
- Contenedor: Docker + compose + NVIDIA Container Toolkit. Base = imagen ultralytics.
- Front (otro repo): Next.js, Recharts/Chart.js, SVG/Canvas. No mezclar aqui.

## Reglas duras (no romper)

1. Balon = riesgo del proyecto. YOLO COCO out-of-the-box NO sirve (objeto pequeno + blur + oclusion). Plan: YOLO26 fine-tuned + tiling. No alcanza -> TrackNet. No asumir "balon ya funciona".
2. Job largo = cola async (ARQ/Redis). Nunca bloquear endpoint.
3. 1 worker GPU, concurrency=1. 3050 no aguanta 2 jobs en VRAM.
4. Coordenadas SIEMPRE normalizar 0.0-1.0 antes de salir en JSON.
5. JSON salida valida con Pydantic. Contrato manda. Ver INVESTIGACION.md sec 6.
6. half=True en inferencia. imgsz=640 default. 1280 solo si balon lo exige.
7. Muestrear frames 1/N default. No procesar 30fps completos sin razon.
8. Borrar .mp4 original tras generar JSON. No saturar disco.
9. Cero deps innecesarias. Cada lib nueva = justificar.
10. Costo recurrente ~0. Sin APIs de pago en core.

## Endpoints (contrato)

- POST /api/v1/analyze/upload       -> recibe .mp4, encola, responde {job_id}
- GET  /api/v1/analyze/status/{id}  -> en_cola | procesando | <pct> | completado | error
- GET  /api/v1/analyze/results/{id} -> JSON final

JSON = 3 niveles: match_metadata, charts_data, spatial_data, timeline_events.
Incluir schema_version + sampled_fps. Trayectoria densa balon NO va en payload principal (endpoint aparte o comprimido). Detalle: INVESTIGACION.md.

## Estructura

```
vly/
  api/            FastAPI: rutas, modelos pydantic, deps
    main.py
    schemas.py    contrato JSON (pydantic)
    routes.py
  worker/         ARQ: tasks, settings
    tasks.py
  engine/         motor de inferencia (independiente de la web)
    video.py      lectura/muestreo de frames
    detect.py     YOLO26 + (tracknet fase B)
    track.py      ByteTrack + trayectoria balon
    events.py     trayectoria -> eventos (heuristicas)
    export.py     dict -> JSON validado
  models/         pesos (.pt) - gitignore
  data/           samples, datasets - gitignore lo pesado
  docker/         Dockerfile, compose
  tests/
  CLAUDE.md  ROADMAP.md  INVESTIGACION.md
```

Regla: `engine/` no importa FastAPI ni ARQ. Corre standalone por terminal (asi se valida Fase 1).

## Comandos (rellenar al crearlos)

```
# entorno
python -m venv modo_ia && source modo_ia/bin/activate
pip install -r requirements.txt

# motor standalone (Fase 1)
python -m engine.run --video data/sample.mp4 --out out.json

# api local
uvicorn api.main:app --reload

# worker
arq worker.tasks.WorkerSettings

# docker
docker compose -f docker/compose.yml up --build

# verificar GPU en contenedor
docker compose exec worker nvidia-smi
```

## Estado actual

Fase 0 (setup + contrato). Sin codigo aun. Empezar: schemas pydantic + mocks 3 endpoints. Ver ROADMAP.md.

## Que NO hacer

- No real-time / streaming. Es batch.
- No player_id "real" ni homografia en MVP (backlog).
- No modelos grandes. No multi-GPU.
- No BackgroundTasks para procesar video.
- No volcar datos crudos por frame al front.
- No agregar front aqui.
