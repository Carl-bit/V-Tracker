# VolleyVision (VLY)

Analisis offline de video de voley. Entra un .mp4, sale un JSON con el balon
trackeado, metricas y eventos (saque/recepcion/armado/remate). Pipeline:
FastAPI recibe el video, lo encola en ARQ/Redis y un worker GPU (YOLO26 +
ByteTrack) lo procesa async. El front (React/Next) vive en otro repo.

## Como correr

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
```
