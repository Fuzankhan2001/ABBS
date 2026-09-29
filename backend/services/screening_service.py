from __future__ import annotations

from datetime import datetime
import secrets
from threading import Lock

import pandas as pd
import numpy as np

from module_A import AdvancedScreeningEngine
import module_B as module_b
from synthetic_data_set import generate_burnin_dataset

REQUIRED_COLUMNS = {"Component_ID", "Lot_ID", "Iddq_0h_uA", "Iddq_24h_uA"}
MISSION_PROFILES = {
    "EOS-08": "Payload Subsystem Burn-In",
    "GAGANYAAN": "Avionics Tier-1 Screening",
}
class ScreeningValidationError(ValueError):
    """A client-facing telemetry validation failure."""

    def __init__(self, message: str, error: str = "VALIDATION_ERROR", missing_columns: list[str] | None = None, row: int | None = None, column: str | None = None) -> None:
        super().__init__(message)
        self.error = error
        self.missing_columns = missing_columns or []
        self.row = row
        self.column = column


class ScreeningService:
    def __init__(self) -> None:
        self._assessments: dict[str, dict] = {}
        self._latest_assessment_id: str | None = None
        self._run_counters: dict[str, int] = {}
        self._lock = Lock()

    def mission_assessment(self, mission: str) -> dict:
        normalized_mission = mission.strip().upper()
        if normalized_mission not in MISSION_PROFILES:
            raise ScreeningValidationError("Mission profile must be EOS-08 or GAGANYAAN.")
        raw_df = generate_burnin_dataset(n_samples=2500, random_state=secrets.randbits(32))
        return self.run(
            raw_df,
            mission="Gaganyaan" if normalized_mission == "GAGANYAAN" else "EOS-08",
            scenario=MISSION_PROFILES[normalized_mission],
            dataset="Mission Assessment Dataset",
            source_type="MISSION_ASSESSMENT",
        )

    def validate_dataframe(self, raw_df: pd.DataFrame) -> None:
        if raw_df.empty:
            raise ScreeningValidationError("The uploaded ATE dataset is empty.", "EMPTY_DATASET")
        missing = sorted(REQUIRED_COLUMNS - set(raw_df.columns))
        if missing:
            raise ScreeningValidationError("The uploaded ATE dataset does not contain the required telemetry fields.", "INVALID_DATASET_SCHEMA", missing)
        for column in ("Component_ID", "Lot_ID"):
            missing_row = raw_df[column].isna() | raw_df[column].astype(str).str.strip().eq("")
            if missing_row.any():
                row = int(raw_df.index[missing_row][0]) + 2
                raise ScreeningValidationError(f"Required identifier '{column}' is missing at CSV row {row}.", "MISSING_IDENTIFIER", row=row, column=column)
        for column in ("Iddq_0h_uA", "Iddq_24h_uA"):
            values = pd.to_numeric(raw_df[column], errors="coerce")
            invalid = values.isna() | ~np.isfinite(values)
            if invalid.any():
                row = int(raw_df.index[invalid][0]) + 2
                raise ScreeningValidationError(f"Telemetry field '{column}' has a missing, non-numeric, or non-finite value at CSV row {row}.", "INVALID_DATASET_VALUES", row=row, column=column)

    def run_upload(self, raw_df: pd.DataFrame, filename: str) -> dict:
        self.validate_dataframe(raw_df)
        return self.run(
            raw_df,
            mission="Imported ATE Data",
            scenario="Imported Burn-In Telemetry Screening",
            dataset="Imported ATE Dataset",
            source_type="ATE_CSV",
            source_filename=filename,
        )

    def run(self, raw_df: pd.DataFrame, mission: str, scenario: str, dataset: str, source_type: str, source_filename: str | None = None) -> dict:
        self.validate_dataframe(raw_df)
        # Reuse the same in-memory flow as the Streamlit reference implementation.
        engine = AdvancedScreeningEngine(base_k_sigma=3.0)
        engine.fit(raw_df)
        screened_module_a = engine.predict(raw_df)
        final_df = module_b.run_drift_forecast(screened_module_a)

        assessment_id = self._next_assessment_id()
        components = [self._component_record(row) for _, row in final_df.iterrows()]
        lots = self._lot_records(final_df, engine.lot_diagnostics)
        payload = {
            "id": assessment_id,
            "assessment_id": assessment_id,
            "mission": mission,
            "scenario": scenario,
            "screening_scenario": scenario,
            "dataset": dataset,
            "source_type": source_type,
            "source_filename": source_filename,
            "assessedAt": datetime.now().astimezone().strftime("%d %b %Y · %H:%M %Z").upper(),
            "status": "COMPLETE",
            "assessment_status": "COMPLETE",
            "total_components": len(components),
            "qualified_components": sum(component["verdict"] == "QUALIFIED" for component in components),
            "module_a_anomaly_count": sum(component["moduleA"] != "PASS" for component in components),
            "early_drift_rejection_count": sum(component["moduleBEarlyReject"] for component in components),
            "final_rejection_count": sum(component["finalSystemReject"] for component in components),
            "components": components,
            "lots": lots,
        }
        self._assessments[assessment_id] = payload
        self._latest_assessment_id = assessment_id
        return payload

    def _next_assessment_id(self) -> str:
        date_key = datetime.now().strftime("%Y%m%d")
        with self._lock:
            self._run_counters[date_key] = self._run_counters.get(date_key, 0) + 1
            return f"ABSS-{date_key}-{self._run_counters[date_key]:03d}"

    def component(self, component_id: str) -> dict | None:
        assessment = self._latest_assessment()
        if assessment is None:
            return None
        return next((item for item in assessment["components"] if item["id"] == component_id), None)

    def lot(self, lot_id: str) -> dict | None:
        assessment = self._latest_assessment()
        if assessment is None:
            return None
        return next((item for item in assessment["lots"] if item["id"] == lot_id), None)

    def _latest_assessment(self) -> dict | None:
        return self._assessments.get(self._latest_assessment_id) if self._latest_assessment_id else None

    @staticmethod
    def _component_record(row: pd.Series) -> dict:
        module_a_reject = bool(row["Module_A_Reject"])
        module_b_reject = bool(row["Module_B_Early_Reject"])
        final_reject = bool(row["Final_System_Reject"])
        if final_reject and module_a_reject and module_b_reject:
            verdict = "FINAL REJECT"
        elif module_a_reject:
            verdict = "MODULE A ANOMALY"
        elif module_b_reject:
            verdict = "EARLY DRIFT REJECT"
        else:
            verdict = "QUALIFIED"
        module_b_evaluated = bool(row.get("Module_B_Evaluated", 1))
        module_b_reason = (
            f"Projected degradation slope of {float(row['Projected_Slope']):.4f} µA/hr exceeds the {module_b.DEFAULT_SLOPE_THRESHOLD:.4f} µA/hr early-rejection threshold."
            if module_b_reject
            else f"Projected leakage growth remains below the {module_b.DEFAULT_SLOPE_THRESHOLD:.4f} µA/hr early-rejection threshold."
        )
        if not module_b_evaluated:
            module_b_reason = "Module B was not evaluated: this component was rejected by Module A and stopped by the strict cascade."
        telemetry = [
            {"hour": 0, "value": float(row["Iddq_0h_uA"]), "forecast": False},
            {"hour": 24, "value": float(row["Iddq_24h_uA"]), "forecast": False},
        ]
        if "Iddq_96h_uA" in row.index and pd.notna(row["Iddq_96h_uA"]):
            telemetry.append({"hour": 96, "value": float(row["Iddq_96h_uA"]), "forecast": False})
        if "Iddq_168h_uA" in row.index and pd.notna(row["Iddq_168h_uA"]):
            telemetry.append({"hour": 168, "value": float(row["Iddq_168h_uA"]), "forecast": False})
        if module_b_evaluated:
            telemetry.append({"hour": 168, "value": float(row["Pred_Iddq_168h_uA"]), "forecast": True})
        return {
            "id": str(row["Component_ID"]), "lot": str(row["Lot_ID"]), "verdict": verdict,
            "moduleA": "FLAGGED" if module_a_reject else "PASS",
            "moduleB": "NOT EVALUATED" if not module_b_evaluated else "REJECT" if module_b_reject else "PASS",
            "iddq0": float(row["Iddq_0h_uA"]), "iddq24": float(row["Iddq_24h_uA"]),
            "iddq96": float(row["Iddq_96h_uA"]) if "Iddq_96h_uA" in row.index and pd.notna(row["Iddq_96h_uA"]) else None,
            "iddq168": float(row["Iddq_168h_uA"]) if "Iddq_168h_uA" in row.index and pd.notna(row["Iddq_168h_uA"]) else None,
            "forecast168": float(row["Pred_Iddq_168h_uA"]) if module_b_evaluated else None,
            "slope": float(row["Projected_Slope"]) if module_b_evaluated else None,
            "mahalanobis": float(row["Mahalanobis_Distance"]), "patViolated": bool(row["PAT_Violated"]),
            "covarianceViolated": bool(row["Covariance_Violated"]), "moduleAReject": module_a_reject,
            "moduleBEarlyReject": module_b_reject, "moduleBEvaluated": module_b_evaluated, "finalSystemReject": final_reject,
            "reason": str(row["QA_Engineering_Reason"]), "moduleBReason": module_b_reason, "telemetry": telemetry,
        }

    @staticmethod
    def _lot_records(final_df: pd.DataFrame, diagnostics: dict) -> list[dict]:
        records = []
        for lot_id, group in final_df.groupby("Lot_ID", sort=True):
            diag = diagnostics[str(lot_id)]
            records.append({
                "id": str(lot_id), "population": len(group), "median": float(diag["median"]),
                "robustSigma": float(diag["robust_sigma"]), "upperPat": float(diag["upper_pat"]),
                "lowerPat": float(diag["lower_pat"]), "anomalies": int(group["Module_A_Reject"].sum()),
                "skewness": float(diag["skewness"]), "kurtosis": float(diag["kurtosis"]),
                "isContaminated": bool(diag["is_contaminated"]), "adaptiveK": float(diag["adaptive_k"]),
                "distribution": [float(value) for value in group["Iddq_0h_uA"].tolist()],
                "components": [str(value) for value in group["Component_ID"].tolist()],
            })
        return records
