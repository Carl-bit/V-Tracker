"""Contrato JSON de salida (GET /api/v1/analyze/results/{job_id}).

Fuente: INVESTIGACION.md sec 6 + mock de first step.txt.
Coordenadas SIEMPRE normalizadas 0.0-1.0.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"

NormCoord = Annotated[float, Field(ge=0.0, le=1.0)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]

EventType = Literal["saque", "recepcion", "armado", "remate"]
Team = Literal["team_a", "team_b"]
JobStatus = Literal["en_cola", "procesando", "completado", "error"]


class UploadResponse(BaseModel):
    job_id: str
    status: JobStatus = "en_cola"


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)


class VideoInfo(BaseModel):
    duration_seconds: float = Field(gt=0)
    fps_processed: float = Field(gt=0)
    original_resolution: str


class StatisticsSummary(BaseModel):
    total_rallies: int = Field(ge=0)
    total_ball_touches: int = Field(ge=0)
    max_ball_speed_kmh: float = Field(ge=0)
    attack_efficiency_percentage: float = Field(ge=0, le=100)


class MatchMetadata(BaseModel):
    schema_version: str = SCHEMA_VERSION
    job_id: str
    status: JobStatus
    processed_at: str
    sampled_fps: float = Field(gt=0, description="fps efectivo procesado (1/N)")
    video: VideoInfo
    statistics_summary: StatisticsSummary


class SpeedPoint(BaseModel):
    timestamp: float = Field(ge=0)
    speed: float = Field(ge=0)


class TeamPossession(BaseModel):
    team_a: float = Field(ge=0, le=100)
    team_b: float = Field(ge=0, le=100)


class ChartsData(BaseModel):
    ball_speed_timeline: list[SpeedPoint]
    team_possession_percentage: TeamPossession


class HeatMapPoint(BaseModel):
    x_norm: NormCoord
    y_norm: NormCoord
    intensity: int = Field(ge=0)


class PlayerImpactZone(BaseModel):
    player_id: str
    role: str
    avg_x: NormCoord
    avg_y: NormCoord


class SpatialData(BaseModel):
    ball_heat_map: list[HeatMapPoint]
    player_impact_zones: list[PlayerImpactZone]


class BallCoordinates(BaseModel):
    x_norm: NormCoord
    y_norm: NormCoord


class TimelineEvent(BaseModel):
    event_id: str
    timestamp: float = Field(ge=0)
    type: EventType
    team: Team
    player_id: str
    confidence: Confidence
    ball_coordinates: BallCoordinates


class AnalysisResult(BaseModel):
    match_metadata: MatchMetadata
    charts_data: ChartsData
    spatial_data: SpatialData
    timeline_events: list[TimelineEvent]

    @classmethod
    def example(cls) -> "AnalysisResult":
        """Mock del documento. Sirve para los 3 endpoints mock y Postman."""
        return cls.model_validate(
            {
                "match_metadata": {
                    "schema_version": SCHEMA_VERSION,
                    "job_id": "vly_987654321_chk",
                    "status": "completado",
                    "processed_at": "2026-06-11T22:45:00Z",
                    "sampled_fps": 10.0,
                    "video": {
                        "duration_seconds": 124.5,
                        "fps_processed": 30,
                        "original_resolution": "1920x1080",
                    },
                    "statistics_summary": {
                        "total_rallies": 12,
                        "total_ball_touches": 142,
                        "max_ball_speed_kmh": 82.4,
                        "attack_efficiency_percentage": 64.5,
                    },
                },
                "charts_data": {
                    "ball_speed_timeline": [
                        {"timestamp": 0.0, "speed": 0.0},
                        {"timestamp": 0.5, "speed": 45.2},
                        {"timestamp": 1.2, "speed": 78.1},
                        {"timestamp": 1.8, "speed": 12.3},
                    ],
                    "team_possession_percentage": {
                        "team_a": 52.3,
                        "team_b": 47.7,
                    },
                },
                "spatial_data": {
                    "ball_heat_map": [
                        {"x_norm": 0.25, "y_norm": 0.45, "intensity": 8},
                        {"x_norm": 0.72, "y_norm": 0.12, "intensity": 14},
                        {"x_norm": 0.50, "y_norm": 0.50, "intensity": 25},
                    ],
                    "player_impact_zones": [
                        {
                            "player_id": "player_1",
                            "role": "setter",
                            "avg_x": 0.51,
                            "avg_y": 0.48,
                        }
                    ],
                },
                "timeline_events": [
                    {
                        "event_id": "evt_001",
                        "timestamp": 14.2,
                        "type": "saque",
                        "team": "team_a",
                        "player_id": "player_5",
                        "confidence": 0.94,
                        "ball_coordinates": {"x_norm": 0.15, "y_norm": 0.85},
                    },
                    {
                        "event_id": "evt_002",
                        "timestamp": 15.8,
                        "type": "recepcion",
                        "team": "team_b",
                        "player_id": "player_3",
                        "confidence": 0.89,
                        "ball_coordinates": {"x_norm": 0.75, "y_norm": 0.32},
                    },
                    {
                        "event_id": "evt_003",
                        "timestamp": 17.1,
                        "type": "armado",
                        "team": "team_b",
                        "player_id": "player_1",
                        "confidence": 0.97,
                        "ball_coordinates": {"x_norm": 0.52, "y_norm": 0.45},
                    },
                    {
                        "event_id": "evt_004",
                        "timestamp": 18.3,
                        "type": "remate",
                        "team": "team_b",
                        "player_id": "player_4",
                        "confidence": 0.91,
                        "ball_coordinates": {"x_norm": 0.48, "y_norm": 0.61},
                    },
                ],
            }
        )
