from typing import Literal
from pydantic import BaseModel, Field


class MissionRequest(BaseModel):
    mission: str = Field(..., description="Supported mission screening profile")


class TelemetryPoint(BaseModel):
    hour: int
    value: float
    forecast: bool = False


class ComponentResult(BaseModel):
    id: str
    lot: str
    verdict: Literal["QUALIFIED", "MODULE A ANOMALY", "EARLY DRIFT REJECT", "FINAL REJECT"]
    moduleA: Literal["PASS", "FLAGGED", "REJECT"]
    moduleB: Literal["PASS", "FLAGGED", "REJECT"]
    iddq0: float
    iddq24: float
    iddq96: float | None = None
    iddq168: float | None = None
    forecast168: float
    slope: float
    mahalanobis: float
    patViolated: bool
    covarianceViolated: bool
    moduleAReject: bool
    moduleBEarlyReject: bool
    finalSystemReject: bool
    reason: str
    moduleBReason: str
    telemetry: list[TelemetryPoint]


class LotResult(BaseModel):
    id: str
    population: int
    median: float
    robustSigma: float
    upperPat: float
    lowerPat: float
    anomalies: int
    skewness: float
    kurtosis: float
    isContaminated: bool
    adaptiveK: float
    distribution: list[float]
    components: list[str]


class AssessmentResponse(BaseModel):
    id: str
    assessment_id: str
    mission: str
    scenario: str
    screening_scenario: str
    dataset: str
    source_type: Literal["MISSION_ASSESSMENT", "ATE_CSV"]
    source_filename: str | None = None
    assessedAt: str
    status: Literal["COMPLETE"]
    assessment_status: Literal["COMPLETE"]
    total_components: int
    qualified_components: int
    module_a_anomaly_count: int
    early_drift_rejection_count: int
    final_rejection_count: int
    components: list[ComponentResult]
    lots: list[LotResult]


class ErrorResponse(BaseModel):
    error: str
    message: str
    missing_columns: list[str] = []
