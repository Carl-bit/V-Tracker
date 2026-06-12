"""App FastAPI. Correr: uvicorn api.main:app --reload"""

from fastapi import FastAPI

from api.routes import router
from api.schemas import SCHEMA_VERSION

app = FastAPI(
    title="VolleyVision (VLY)",
    description="Analisis offline de video de voley. Contrato JSON v" + SCHEMA_VERSION,
    version=SCHEMA_VERSION,
)

app.include_router(router)
