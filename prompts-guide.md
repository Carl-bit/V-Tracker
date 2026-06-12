# prompts-guide.md - VolleyVision (VLY)

Prompts para ejecucion ordenada con Claude Code. Pegar uno a la vez, en orden.
Estilo caveman. Cada prompt = tarea chica + verificacion. Sigue ROADMAP.md.

## Como usar

- CLAUDE.md se auto-carga: prompts asumen sus reglas. No repetir reglas.
- 1 prompt = 1 commit. Pasar al siguiente solo si DoD verde.
- Usar plan mode para prompts con `[PLAN]`. Revisar plan antes de aplicar.
- Tras cada DoD verde: commit + marcar checkbox en ROADMAP.md.
- Si agente se desvia de scope (toca 3+ archivos sin pedirlo): parar, recordar scope.
- Validar el balon (P1.3) lo antes posible. Es el riesgo.

Prefijo opcional si el agente pierde el hilo:
`Lee CLAUDE.md, ROADMAP.md, INVESTIGACION.md antes de actuar. Respeta reglas duras.`

---

# FASE 0 - Setup + contrato

## P0.1 - Scaffold

```
Scaffold del repo segun estructura en CLAUDE.md. Crea:
- arbol de carpetas vacio (api/ worker/ engine/ models/ data/ docker/ tests/)
- requirements.txt con: ultralytics, opencv-python, numpy, fastapi, pydantic,
  uvicorn[standard], arq, redis. Pin versiones actuales estables.
- .gitignore: modo_ia/, models/*.pt, data/ pesado, __pycache__, *.json de outputs, .env
- README.md minimo (1 parrafo + como correr)
No instales nada. No escribas logica. Solo estructura + archivos config.
DoD: arbol coincide con CLAUDE.md, requirements.txt completo.
```

## P0.2 - Contrato Pydantic

```
Crea api/schemas.py: modelos Pydantic v2 del JSON de salida segun INVESTIGACION.md
sec 6 y el ejemplo de first_step.txt.
Modelos: MatchMetadata (con schema_version y sampled_fps), VideoInfo,
StatisticsSummary, ChartsData, SpatialData, TimelineEvent, AnalysisResult (raiz).
Coordenadas siempre float 0.0-1.0. Usa Literal para event type
(saque|recepcion|armado|remate). confidence float 0-1.
Incluye un AnalysisResult.example() classmethod que devuelva el mock del documento.
DoD: `python -c "from api.schemas import AnalysisResult; AnalysisResult.example()"` sin error.
```

## P0.3 - API mock

```
Crea api/main.py + api/routes.py: FastAPI con los 3 endpoints de CLAUDE.md.
- POST /api/v1/analyze/upload: acepta UploadFile .mp4, NO procesa aun,
  genera job_id (uuid con prefijo vly_), responde {job_id, status:"en_cola"}.
- GET /api/v1/analyze/status/{job_id}: devuelve mock {status:"completado", progress:100}.
- GET /api/v1/analyze/results/{job_id}: devuelve AnalysisResult.example().
Sin cola aun, sin estado real. Solo contrato vivo.
DoD: `uvicorn api.main:app` levanta; /docs muestra los 3 endpoints; results devuelve JSON valido.
```

## P0.4 - Mocks para front

```
Crea tests/mocks/result_example.json (dump de AnalysisResult.example()) y
docs/curl-examples.md con curl de los 3 endpoints contra localhost.
DoD: el JSON valida contra el schema; los curl funcionan contra uvicorn local.
```

---

# FASE 1 - Motor standalone (engine/)

Recordatorio: engine/ NO importa FastAPI ni ARQ. Corre por terminal.

## P1.1 - Lectura de video

```
Crea engine/video.py: funcion read_frames(path, sample_every_n=10) que abre un .mp4
con opencv y yielda (frame_idx, timestamp_seg, frame_bgr) muestreando 1 de cada N.
Expone tambien get_video_meta(path) -> {duration_seconds, fps, width, height}.
Maneja apertura fallida con excepcion clara. Sin YOLO aqui.
DoD: script de prueba imprime meta + cuenta de frames muestreados de un .mp4 en data/.
```

## P1.2 - Deteccion YOLO

```
Crea engine/detect.py: clase Detector que carga YOLO26 (yolo26n.pt o yolo26s.pt,
configurable) con half=True, imgsz=640. Metodo detect(frame_bgr) -> lista de
detecciones {cls, conf, xyxy}. Filtra por clases relevantes (persona, balon).
Carga el modelo UNA vez (no por frame). Si no hay GPU, cae a CPU con warning.
NO normalices aqui (eso va en export). Sin tracking aun.
DoD: corre sobre 1 frame de data/ y lista detecciones con conf.
```

## P1.3 - [CHECKPOINT] Validar balon

```
[PLAN] Crea tests/validate_ball.py: corre engine/video + engine/detect sobre un
.mp4 real de voleibol en data/ y reporta: frames totales muestreados, frames con
balon detectado, recall aproximado del balon (% frames con balon visible que lo
detecto), conf promedio del balon.
Imprime veredicto: si recall < 60%, recomienda fine-tune o TrackNet (ver
INVESTIGACION.md sec 2). NO implementes TrackNet aun, solo mide y reporta.
DoD: numero concreto de recall del balon impreso. DECISION manual antes de seguir.
```

Si recall malo -> abrir mini-fase fine-tune (P1.3b) antes de P1.4:

```
[PLAN] Propon plan para fine-tune YOLO26 sobre dataset de balon de voleibol de
Roboflow (sin descargar nada aun): que dataset, formato, comando de train con
imgsz y epochs realistas para RTX 3050 6GB (batch chico). Solo el plan.
```

## P1.4 - Tracking

```
Crea engine/track.py:
- Jugadores: integra ByteTrack sobre detecciones persona -> ids estables por frame.
- Balon: trayectoria por asociacion frame-a-frame (cercania) + interpolacion lineal
  de huecos cortos (balon perdido pocos frames). Devuelve lista temporal de
  posiciones de balon {timestamp, x, y} en pixeles (normalizar va despues).
DoD: sobre el .mp4 de prueba, genera trayectoria de balon continua (con huecos
interpolados) + tracks de jugadores con id.
```

## P1.5 - Eventos

```
Crea engine/events.py: detect_events(ball_trajectory, video_meta) -> lista de
TimelineEvent. Heuristicas (no ML): cambio brusco de direccion = toque;
pico de velocidad hacia campo contrario = saque/remate; usa altura + zona de
cancha para desambiguar tipo (saque|recepcion|armado|remate). Umbrales como
constantes nombradas y ajustables al inicio del archivo. Asigna confidence
heuristico 0-1. team por lado de cancha (no player_id real aun).
DoD: sobre el .mp4 de prueba, devuelve al menos saque + remate con timestamp plausible.
```

## P1.6 - Export

```
Crea engine/export.py: build_result(meta, ball_traj, player_tracks, events) que:
- normaliza TODAS las coordenadas a 0.0-1.0 (dividir por width/height),
- arma statistics_summary, charts_data (arrays planos), spatial_data (heatmap +
  zonas), timeline_events,
- valida contra AnalysisResult de api/schemas.py (importa el schema, no redefine),
- incluye schema_version + sampled_fps,
- NO mete la trayectoria densa del balon en el payload principal (endpoint aparte
  o array comprimido, segun INVESTIGACION.md sec 6),
- escribe el JSON a un path dado.
DoD: AnalysisResult valida el output sin error.
```

## P1.7 - CLI del motor

```
Crea engine/run.py con CLI: `python -m engine.run --video X.mp4 --out Y.json
--model yolo26n.pt --sample 10`. Encadena video -> detect -> track -> events ->
export. Logging de progreso a stderr (frames procesados, %). Maneja errores con
exit code != 0.
DoD: `python -m engine.run --video data/sample.mp4 --out out.json` genera JSON
valido end-to-end por terminal.
```

---

# FASE 2 - Backend + cola async

## P2.1 - Worker ARQ

```
Crea worker/tasks.py: tarea ARQ analyze_video(ctx, job_id, video_path) que llama
al motor (engine.run como funcion, no subprocess) y guarda progreso + resultado en
Redis bajo claves vly:job:{job_id}:status y vly:job:{job_id}:result.
WorkerSettings con redis_settings, concurrency=1 (regla 3 de CLAUDE.md),
max_jobs=1. Actualiza status: en_cola -> procesando -> progress% -> completado/error.
DoD: encolar un job manual procesa el video y deja result en Redis.
```

## P2.2 - Conectar API a cola

```
Modifica api/routes.py:
- POST /upload: guarda .mp4 en data/uploads/{job_id}.mp4, encola analyze_video en
  ARQ, responde {job_id, status:"en_cola"}.
- GET /status/{job_id}: lee status real de Redis.
- GET /results/{job_id}: lee result de Redis; si no completado, 409 con status actual.
Sin mocks ya. Maneja job_id inexistente con 404.
DoD: upload real -> polling de status -> results devuelve JSON del motor. API
responde fluido mientras el worker procesa.
```

## P2.3 - Errores + limpieza

```
Endurece worker/tasks.py: try/except que marca status error con motivo si el motor
falla. Tras completar OK, borra data/uploads/{job_id}.mp4 (regla 8). Set TTL en
las claves Redis (p.ej. 24h) para no acumular.
DoD: job que falla deja status error legible; .mp4 original se borra al completar.
```

---

# FASE 3 - Docker + GPU

## P3.1 - Dockerfile

```
[PLAN] Crea docker/Dockerfile partiendo de imagen base ultralytics (trae torch+
CUDA+opencv). Copia codigo, instala deps faltantes (arq, fastapi extras), expone
puerto, CMD para api. Si en vez de ultralytics usas python base, agrega libs de
sistema de OpenCV: libgl1, libglib2.0-0, libsm6, libxext6.
DoD: `docker build` exitoso; imagen razonable.
```

## P3.2 - Compose + GPU

```
Crea docker/compose.yml con servicios: api, worker, redis.
- worker expone GPU con deploy.resources.reservations.devices:
  [{driver: nvidia, count: all, capabilities: [gpu]}] (sintaxis 2026).
- redis:7-alpine con maxmemory.
- volumenes para data/ y models/.
- api y worker comparten la imagen, distinto command (uvicorn vs arq worker).
DoD: `docker compose up --build` levanta los 3 servicios sin crash.
```

## P3.3 - Verificar GPU

```
Documenta en docs/gpu-setup.md los pasos host: instalar NVIDIA Container Toolkit,
`sudo nvidia-ctk runtime configure --runtime=docker`, restart docker, verificar con
`docker run --rm --gpus all nvidia/cuda:<tag> nvidia-smi`. Luego
`docker compose exec worker nvidia-smi`.
DoD: nvidia-smi funciona dentro del worker; un job dockerizado usa GPU (verificar
speedup vs CPU en un video de prueba).
```

---

# FASE 4 - Pulido para front

## P4.1 - Contrato listo para render

```
[PLAN] Revisa el JSON real que produce el motor contra lo que el front necesita
(INVESTIGACION.md sec 6, first_step.txt seccion "por que graficar"):
- statistics_summary: valores listos para tarjetas KPI, sin calculo en navegador.
- charts_data: arrays planos {timestamp, speed} para Recharts/Chart.js.
- spatial_data: x_norm/y_norm 0-1 para heatmap sobre SVG de cancha.
- timeline_events: ordenados por timestamp, listos para seek de video HTML5.
- sampled_fps presente.
Propon ajustes minimos al export. Decide donde vive la trayectoria densa del balon.
DoD: el front puede pintar KPIs + grafico + heatmap + timeline solo con el JSON,
sin recalcular nada.
```

---

# Backlog (no MVP)

Prompts para despues, no ahora: TrackNet dedicado balon, homografia de cancha,
player_id estable + equipos por reid, WebSocket/SSE en vez de polling,
clasificador de eventos por ML, Flower para monitoreo. Ver ROADMAP.md backlog.
