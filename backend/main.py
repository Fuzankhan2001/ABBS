from __future__ import annotations

from io import BytesIO

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas.assessment import AssessmentResponse, ErrorResponse, MissionRequest
from backend.services.screening_service import ScreeningService, ScreeningValidationError

app = FastAPI(title="ABSS Screening API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
screening_service = ScreeningService()


@app.exception_handler(ScreeningValidationError)
async def validation_exception_handler(_, exc: ScreeningValidationError):
    return JSONResponse(status_code=422, content={"error": exc.error, "message": str(exc), "missing_columns": exc.missing_columns})


@app.get("/api/health")
def health() -> dict:
    return {"status": "ready", "service": "ABSS Screening API"}


@app.post("/api/assessment/mission", response_model=AssessmentResponse, responses={422: {"model": ErrorResponse}})
def mission_assessment(request: MissionRequest) -> dict:
    return screening_service.mission_assessment(request.mission)


@app.post("/api/assessment/validate")
async def validate_upload(file: UploadFile = File(...)) -> dict:
    size = file.size or 0
    raw_df = await read_csv(file)
    screening_service.validate_dataframe(raw_df)
    return {"name": file.filename or "telemetry.csv", "size": size, "rows": len(raw_df), "valid": True}


@app.post("/api/assessment/upload", response_model=AssessmentResponse, responses={422: {"model": ErrorResponse}})
async def upload_assessment(file: UploadFile = File(...)) -> dict:
    raw_df = await read_csv(file)
    return screening_service.run_upload(raw_df, file.filename or "telemetry.csv")


@app.get("/api/components/{component_id}")
def component_investigation(component_id: str) -> dict:
    component = screening_service.component(component_id)
    if component is None:
        raise HTTPException(status_code=404, detail={"error": "NOT_FOUND", "message": "Component was not found in the active assessment."})
    return component


@app.get("/api/lots/{lot_id}")
def lot_analysis(lot_id: str) -> dict:
    lot = screening_service.lot(lot_id)
    if lot is None:
        raise HTTPException(status_code=404, detail={"error": "NOT_FOUND", "message": "Lot was not found in the active assessment."})
    return lot


async def read_csv(file: UploadFile) -> pd.DataFrame:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise ScreeningValidationError("Please provide a CSV telemetry file.", "INVALID_FILE_TYPE")
    contents = await file.read()
    if not contents.strip():
        raise ScreeningValidationError("The uploaded ATE dataset is empty.", "EMPTY_DATASET")
    try:
        return pd.read_csv(BytesIO(contents))
    except Exception:
        raise ScreeningValidationError("The uploaded file could not be parsed as CSV telemetry.", "MALFORMED_CSV")
