"""Rutas /api/v1/analyze. Fase 0: contrato vivo, sin cola ni estado real."""

import uuid

from fastapi import APIRouter, HTTPException, UploadFile

from api.schemas import AnalysisResult, StatusResponse, UploadResponse

router = APIRouter(prefix="/api/v1/analyze", tags=["analyze"])


@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_video(file: UploadFile) -> UploadResponse:
    if not (file.filename or "").lower().endswith(".mp4"):
        raise HTTPException(status_code=415, detail="Solo .mp4")
    # TODO Fase 3: guardar temporal + encolar en ARQ. Por ahora se descarta.
    job_id = f"vly_{uuid.uuid4().hex}"
    return UploadResponse(job_id=job_id, status="en_cola")


@router.get("/status/{job_id}", response_model=StatusResponse)
async def job_status(job_id: str) -> StatusResponse:
    # TODO Fase 3: leer estado real de Redis.
    return StatusResponse(job_id=job_id, status="completado", progress=100)


@router.get("/results/{job_id}", response_model=AnalysisResult)
async def job_results(job_id: str) -> AnalysisResult:
    # TODO Fase 3: leer resultado real. Mock fijo del contrato.
    return AnalysisResult.example()
