# ROADMAP - VolleyVision (VLY)

Plan por fases. Cada fase tiene entregable y criterio de aceptacion (DoD).
Regla: no pasar de fase sin cumplir el DoD. El balon es el riesgo; validalo temprano.

Leyenda: [ ] pendiente  [~] en curso  [x] hecho

---

## Fase 0 - Setup y contrato (1-2 dias)

Objetivo: tener el esqueleto y el contrato de datos para trabajar en paralelo.

- [x] Repo + estructura de carpetas (ver CLAUDE.md)
- [x] Entorno virtual `modo_ia` + deps base (ultralytics, opencv-python, numpy, fastapi, pydantic, uvicorn, arq, redis)
- [x] Modelos Pydantic del JSON de salida (match_metadata, charts_data, spatial_data, timeline_events) con `schema_version`
- [x] FastAPI con los 3 endpoints devolviendo MOCK fijo:
      POST /api/v1/analyze/upload, GET /status/{job_id}, GET /results/{job_id}
- [x] Mocks para front: tests/mocks/result_example.json + docs/curl-examples.md

DoD: el front puede consumir mocks sin que la IA exista. /docs de FastAPI muestra el contrato.

---

## Fase 1 - Motor de inferencia por terminal (3-6 dias)

Objetivo: el nucleo funciona standalone, sin web. Aqui se valida el balon.

- [x] Script: leer un .mp4 local frame a frame (opencv), con muestreo 1/N frames
- [x] YOLO26 (n/s) inferencia con `half=True` sobre cada frame -> detecciones
- [ ] Validar deteccion del balon en VIDEOS REALES propios (no demos)
- [ ] Si recall del balon es bajo: fine-tune YOLO26 con dataset de voleibol (Roboflow) o tiling/SAHI
- [ ] Tracking: ByteTrack para jugadores; trayectoria de balon por asociacion + interpolacion
- [ ] Algoritmo de eventos (heuristico): saque, recepcion, armado, remate sobre la trayectoria
- [ ] Normalizar coordenadas a 0.0-1.0
- [ ] Exportar a .json local con el contrato de Fase 0

DoD: dado un .mp4, el script genera un .json valido con balon trackeado y al menos
saque/remate detectados con confianza razonable. Mide y anota recall del balon.

Checkpoint de decision: si YOLO no alcanza para el balon -> abrir mini-fase TrackNet
(vball-net como base) solo para el balon. No bloquear el resto.

---

## Fase 2 - Integracion backend + cola async (2-4 dias)

Objetivo: unir API (Fase 0) con motor (Fase 1) sin bloquear el servidor.

- [ ] Worker ARQ + Redis: POST /upload encola job y responde job_id al instante
- [ ] Worker GPU con concurrency=1 (la 3050 no aguanta 2 jobs)
- [ ] Estado del job en Redis: en_cola -> procesando -> %% -> completado / error
- [ ] /status refleja progreso real; /results entrega el JSON al completar
- [ ] Manejo de errores: job fallido marca estado error con motivo
- [ ] Limpieza: borrar .mp4 original tras generar el JSON

DoD: subes un video por Postman, recibes job_id, haces polling de /status hasta
completado, y /results devuelve el JSON correcto. El API responde fluido mientras procesa.

---

## Fase 3 - Contenerizacion + GPU en VPS (2-3 dias)

Objetivo: que corra igual en cualquier entorno y use la RTX 3050.

- [ ] Dockerfile (base ultralytics o python + libgl1/libglib2.0-0/libsm6/libxext6)
- [ ] docker-compose: api + worker + redis (+ postgres si aplica)
- [ ] NVIDIA Container Toolkit en el VPS + runtime configurado
- [ ] compose expone GPU (deploy.resources.reservations.devices, capabilities: [gpu])
- [ ] Verificar `nvidia-smi` dentro del contenedor del worker
- [ ] Confirmar speedup GPU vs CPU en un video de prueba

DoD: `docker compose up` levanta todo; un job procesado dentro del contenedor usa
la GPU y termina en tiempo aceptable.

---

## Fase 4 - Pulido del contrato para el front (1-2 dias)

Objetivo: que el front renderice sin recalcular nada en el navegador.

- [ ] statistics_summary listo para tarjetas KPI (dato ya masticado)
- [ ] charts_data como arrays planos para Recharts/Chart.js
- [ ] spatial_data con x_norm/y_norm para heatmap sobre SVG de cancha
- [ ] timeline_events ordenados, con timestamp para seek del video HTML5
- [ ] Decidir donde vive la trayectoria densa del balon (endpoint aparte vs comprimido)
- [ ] Anadir sampled_fps al metadata

DoD: el front pinta KPIs, grafico de velocidad, heatmap y timeline clickable
(seek de video) usando solo el JSON, sin calculos extra.

---

## Backlog / mejoras post-MVP

- TrackNet dedicado para el balon (si no se hizo en Fase 1)
- Homografia de cancha: coordenadas tacticas reales (no de pantalla)
- player_id estable y asignacion de equipos por reidentificacion
- WebSocket/SSE en vez de polling para /status
- Clasificador de eventos por ML (reemplazar heuristicas)
- Soporte multi-camara / 3D del balon
- Cache de resultados, reintentos de jobs, panel de monitoreo (Flower)

---

## Hitos

- M1: Fase 0+1 -> el motor genera JSON correcto desde un .mp4 (riesgo del balon resuelto)
- M2: Fase 2 -> flujo web completo async funcionando local
- M3: Fase 3 -> corriendo dockerizado con GPU en el VPS
- M4: Fase 4 -> front consume el JSON end-to-end (MVP demostrable)
