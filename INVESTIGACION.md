# INVESTIGACION - VolleyVision (VLY)

Analisis tecnico previo al desarrollo. Objetivo: validar el plan de `first_step.txt`
contra el estado del arte 2026 y fijar decisiones de arquitectura antes de codear.

Proyecto: analisis automatico de video de voleibol. Subes un .mp4, una IA detecta
balon y eventos (saque, recepcion, armado, remate), devuelve JSON con metricas,
trayectorias y eventos para que un front en React/Next dibuje stats y timeline.

Hardware objetivo (VPS): i7 4th gen + RTX 3050 LP (6GB VRAM), Debian/Ubuntu.

---

## 1. Resumen de decisiones

| Tema | Decision | Por que |
|---|---|---|
| Deteccion de jugadores/cancha | YOLO26 (nano/small) + ByteTrack | SOTA, ligero, corre en 6GB |
| Deteccion del balon | Fase A: YOLO26 fine-tuned. Fase B: TrackNet | COCO out-of-the-box NO basta para el balon |
| Tracking temporal | ByteTrack (jugadores), heatmap (balon) | Motion-centric, robusto a uniformes iguales |
| Backend API | FastAPI + Pydantic | Async nativo, integra librerias IA |
| Cola de tareas | ARQ o Celery + Redis | BackgroundTasks pierde trabajo si cae el proceso |
| Estado de jobs | Redis (MVP) o PostgreSQL | Ya tienes Redis en el stack |
| GPU en Docker | NVIDIA Container Toolkit + imagen ultralytics | Evita compilar OpenCV+CUDA a mano |
| Procesamiento | Offline / batch, NO real-time | 6GB VRAM => batch 1-4, modelos nano/small |
| Coordenadas | Normalizadas 0.0-1.0 (como en el doc) | Responsive en el front, correcto |

---

## 2. El problema central: detectar el balon

Es la parte dificil del proyecto. NO subestimar.

YOLO trae la clase `sports ball` en COCO, pero en voleibol real el balon:
- ocupa muy pocos pixeles (<10x10 en planos abiertos),
- sale con motion blur tras saques/remates,
- desaparece por oclusion (manos, red, sale de cuadro).

YOLO procesa cada frame por separado => pierde el balon cuando se vuelve borroso.

### Opciones (de menos a mas esfuerzo)

1. **YOLO26 fine-tuned sobre dataset de voleibol** (recomendado para empezar).
   - Hay datasets publicos de balon de voleibol en Roboflow.
   - YOLO26 trae STAL (Small-Target-Aware Label Assignment), pensado justo para
     objetos pequenos. Mejor base que YOLOv8 para esto.
   - Tiling / SAHI (cortar el frame en parches) mejora recall del balon a costa de
     latencia. Aceptable porque es offline.

2. **TrackNet (familia V2-V5) para el balon** (mejor precision, mas trabajo).
   - Toma 3 frames consecutivos -> predice un heatmap de posicion del balon.
   - Usa informacion temporal: "recuerda" el movimiento, aguanta blur y oclusion.
   - Referencia voley-especifica y abierta: vball-net (Asigatchov), basado en
     TrackNet, corre 200+ FPS en CPU. Buen punto de partida / benchmark.

3. **Hibrido (objetivo final)**: YOLO para jugadores y cancha + TrackNet para balon.
   Es el patron estandar en analitica deportiva 2026.

### Plan recomendado

Empieza con (1) YOLO26 fine-tuned. Mide recall del balon en tus videos reales.
Si el recall es insuficiente (balon perdido en remates/saques), agrega (2) TrackNet
solo para el balon. No montes TrackNet en el MVP si YOLO te alcanza para validar el
pipeline completo end-to-end.

---

## 3. Tracking y eventos

Detectar != trackear. Detectar da cajas por frame; trackear da identidad y trayectoria.

- **Jugadores**: ByteTrack u OC-SORT (motion-centric). Mejores que metodos por
  apariencia, que fallan cuando los jugadores visten igual.
- **Balon**: la trayectoria sale del heatmap (TrackNet) o de asociar detecciones
  YOLO frame a frame por cercania + interpolacion de huecos.

### Traduccion trayectoria -> eventos

La logica de eventos (saque/recepcion/armado/remate) es un algoritmo sobre la
trayectoria, no IA pesada:
- cambios bruscos de direccion = toque,
- pico de velocidad hacia campo contrario = saque/remate,
- altura + zona de la cancha desambigua el tipo.

Empieza con reglas heuristicas (umbrales de velocidad/angulo/zona). Si no alcanza,
mas adelante un clasificador ligero sobre ventanas de trayectoria. NO empieces por
un modelo de accion complejo.

Nota importante: para que `avg_x/avg_y` y zonas signifiquen algo tactico real,
necesitas mapear coordenadas de pantalla a coordenadas de cancha via homografia
(detectar 4 esquinas de la cancha -> matriz de perspectiva). En el MVP puedes
quedarte en coordenadas de pantalla normalizadas; marca la homografia como mejora.

---

## 4. Restriccion de hardware (RTX 3050 LP, 6GB)

Define toda la arquitectura. Con 6GB de VRAM:
- modelos **nano o small** unicamente (yolo26n / yolo26s), no medium/large,
- **FP16** en inferencia (`half=True`) para ahorrar VRAM,
- **batch 1-4**,
- NO esperes real-time. En GPUs de esta gama, modelos `s` a 720p rinden pocos FPS.

### Implicacion practica (planificacion de jobs)

Video 2 min @ 30fps = ~3600 frames. A pocos FPS de proceso, un job tarda minutos.
Esto confirma el modelo asincrono del documento. Para acelerar:
- **muestrear frames**: procesar 1 de cada N (p.ej. 10-15 fps efectivos) suele bastar
  para trayectoria de balon y eventos,
- bajar `imgsz` (640 estandar; 1280 mejora balon pero cuesta VRAM y tiempo),
- liberar el .mp4 original al terminar (ya esta en el plan).

CPU es fallback valido (vball-net corre en CPU), pero mas lento. Usa la GPU.

---

## 5. Backend asincrono

`BackgroundTasks` de FastAPI corre en el mismo proceso. Para jobs de minutos,
GPU-bound, eso bloquea/satura y pierde trabajo si el proceso reinicia.

### Recomendacion

Cola real con worker separado. Dos opciones, ambas sobre Redis (ya lo tienes):
- **ARQ**: async-native, ligero, encaja natural con FastAPI. Bueno para single-user.
- **Celery + Redis**: mas maduro, mas features (retries, Flower para monitoreo),
  mas overhead operativo.

Para un homelab de pocos usuarios, **ARQ** es suficiente y mas simple. Si quieres
colas separadas (una CPU, una GPU con concurrency=1) y monitoreo, ve a Celery.

Patron: POST /upload encola job -> responde `job_id` al instante. El worker procesa
y actualiza estado en Redis (en_cola -> procesando -> %% -> completado). El front
hace polling de GET /status. (Mejora futura: WebSocket o SSE en vez de polling.)

Regla clave: solo 1 worker GPU con `concurrency=1`. La 3050 no aguanta dos jobs
en paralelo en VRAM.

---

## 6. Contrato JSON

El JSON del documento esta bien disenado (3 niveles: metadata global, series para
graficos, eventos discretos; coordenadas normalizadas). Se mantiene casi tal cual.
Mejoras sugeridas:

- Anadir `schema_version` en `match_metadata` para versionar el contrato.
- Validar todo con **modelos Pydantic** en el backend (sirven de contrato y de
  documentacion automatica via /docs de FastAPI).
- Decidir donde vive la trayectoria densa del balon (cientos de puntos): NO meterla
  en el payload principal. Opciones: endpoint aparte
  GET /results/{job_id}/ball_track, o array comprimido. El doc ya evita el volcado
  crudo por frame, mantener esa decision.
- `confidence` por evento ya esta, bien. Considera un umbral minimo configurable.
- `original_resolution` ya esta; agrega `sampled_fps` (fps efectivo procesado) para
  que el front sepa la granularidad real.

Mock-first sigue siendo correcto: define el JSON, crea mocks en Postman, y el front
arranca en paralelo sin esperar a la IA.

---

## 7. Contenedores y GPU

- Host: instalar **NVIDIA Container Toolkit** y registrar el runtime:
  `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
  Verificar: `docker run --rm --gpus all nvidia/cuda:<tag> nvidia-smi`.
- Imagen base: lo mas simple es partir de la imagen oficial **ultralytics/ultralytics**
  (trae torch + CUDA + opencv ya resueltos) en vez de compilar OpenCV+CUDA a mano.
- Si construyes imagen propia desde python: instalar libs de sistema de OpenCV
  (`libgl1`, `libglib2.0-0`, `libsm6`, `libxext6`).
- docker-compose (sintaxis 2026) expone GPU con:
  `deploy.resources.reservations.devices: [{driver: nvidia, count: all, capabilities: [gpu]}]`.
- Ojo a la compatibilidad driver host <-> version CUDA de la imagen. Mantener driver
  NVIDIA del VPS al dia.

---

## 8. Riesgos y decisiones abiertas

- **Recall del balon**: riesgo principal. Mitigacion: fine-tune + tiling, TrackNet
  como plan B. Validar con TUS videos reales pronto, no con clips de demo.
- **Tiempo de proceso por video**: definir un techo (p.ej. video <= 3 min) para el
  MVP y muestrear frames.
- **Homografia de cancha**: sin ella las "zonas" son de pantalla, no tacticas.
  Decision: aceptable en MVP, marcar como Fase posterior.
- **Etiquetado de equipos/jugadores** (team_a/team_b, player_id): asignar identidad
  estable es no trivial. MVP: trackear sin nombrar jugadores; equipos por lado de
  cancha. player_id real = mejora futura.
- **Dataset**: necesitas video etiquetado de voleibol. Roboflow Universe tiene
  datasets; evaluar licencia y calidad antes de entrenar.

---

## 9. Stack final propuesto

- Inferencia: Python, ultralytics (YOLO26), opencv-python, numpy, (TrackNet/torch
  en fase B), ByteTrack.
- API: FastAPI + Pydantic + uvicorn.
- Cola/estado: ARQ + Redis (o Celery + Redis).
- Persistencia opcional: PostgreSQL (si crece mas alla de estado de jobs).
- Contenedores: Docker + docker-compose, NVIDIA Container Toolkit, imagen ultralytics.
- Front (separado): Next.js/React + Recharts/Chart.js + SVG/Canvas para cancha.

---

## Referencias consultadas

- Ultralytics YOLO26 (release ene-2026): STAL para objetos pequenos, NMS-free.
- vball-net (Asigatchov, 2026): tracker de balon de voleibol basado en TrackNet, CPU.
- TrackNet V1-V5 (papers): heatmap temporal para balones rapidos y pequenos.
- GetStream (2025): patron hibrido YOLO+TrackNet+ByteTrack en analitica deportiva.
- Comparativas FastAPI BackgroundTasks vs Celery (2026): cuando cada uno se rompe.
- NVIDIA Container Toolkit docs + guias compose GPU 2026.
