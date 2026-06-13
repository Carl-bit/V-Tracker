# curl - API VLY (localhost)

Levantar primero: `uvicorn api.main:app --reload` (puerto 8000 default).

## 1. POST /api/v1/analyze/upload

Sube un .mp4, responde 202 con job_id.

```bash
curl -X POST http://localhost:8000/api/v1/analyze/upload \
  -F "file=@data/sample.mp4;type=video/mp4"
```

Respuesta:

```json
{"job_id": "vly_<hex>", "status": "en_cola"}
```

Archivo no .mp4 -> 415.

## 2. GET /api/v1/analyze/status/{job_id}

```bash
curl http://localhost:8000/api/v1/analyze/status/vly_demo123
```

Respuesta (mock Fase 0, siempre completado):

```json
{"job_id": "vly_demo123", "status": "completado", "progress": 100}
```

## 3. GET /api/v1/analyze/results/{job_id}

```bash
curl http://localhost:8000/api/v1/analyze/results/vly_demo123
```

Respuesta: AnalysisResult completo (match_metadata, charts_data,
spatial_data, timeline_events). Mock identico a
`tests/mocks/result_example.json`.

## Flujo completo

```bash
JOB=$(curl -s -X POST http://localhost:8000/api/v1/analyze/upload \
  -F "file=@data/sample.mp4;type=video/mp4" | python -c "import sys,json;print(json.load(sys.stdin)['job_id'])")
curl -s http://localhost:8000/api/v1/analyze/status/$JOB
curl -s http://localhost:8000/api/v1/analyze/results/$JOB
```
