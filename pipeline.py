"""Offline, held-out evaluation for the ABBS strict screening cascade."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from module_A import AdvancedScreeningEngine
from module_B import TARGET_COLUMN, predict_drift, train_drift_forecaster
from synthetic_data_set import generate_burnin_dataset


def _classification_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """Binary metrics with explicit zero-denominator handling."""
    actual, predicted = actual.astype(bool), predicted.astype(bool)
    tp, tn = int((actual & predicted).sum()), int((~actual & ~predicted).sum())
    fp, fn = int((~actual & predicted).sum()), int((actual & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"total_test_samples": int(len(actual)), "actual_anomalies": int(actual.sum()), "predicted_anomalies": int(predicted.sum()), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "false_negative_rate": fn / (fn + tp) if fn + tp else None, "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def _regression_metrics(actual: pd.Series, predicted: pd.Series, baseline: pd.Series) -> dict:
    """Held-out prediction and directional-error metrics.

    The carry-forward baseline is retained solely as a prediction-error reference:
    its implied 24--168h slope is always zero, so it cannot use this cascade's
    positive-slope early-rejection rule.
    """
    if len(actual) == 0:
        return {key: None for key in ("components_entering_module_b", "mae", "rmse", "r2", "baseline_mae", "mae_improvement_vs_baseline", "mean_signed_error", "median_signed_error", "underprediction_count", "underprediction_rate", "overprediction_count", "overprediction_rate")}
    error = predicted - actual
    mae = float(np.abs(error).mean())
    total_variance = float(np.sum(np.square(actual - actual.mean())))
    underprediction, overprediction = predicted < actual, predicted > actual
    baseline_mae = float(np.abs(baseline - actual).mean())
    return {"components_entering_module_b": int(len(actual)), "mae": mae, "rmse": float(np.sqrt(np.mean(np.square(error)))), "r2": float(1 - np.sum(np.square(error)) / total_variance) if total_variance else None, "baseline_mae": baseline_mae, "mae_improvement_vs_baseline": baseline_mae - mae, "mean_signed_error": float(error.mean()), "median_signed_error": float(error.median()), "underprediction_count": int(underprediction.sum()), "underprediction_rate": float(underprediction.mean()), "overprediction_count": int(overprediction.sum()), "overprediction_rate": float(overprediction.mean())}


def _prediction_error_profile(actual: pd.Series, predicted: pd.Series) -> dict:
    """Error profile used to compare the model with a non-screening reference."""
    if len(actual) == 0:
        return {"sample_count": 0, "mae": None, "rmse": None, "mean_signed_error": None, "median_signed_error": None, "underprediction_count": None, "underprediction_rate": None, "overprediction_count": None, "overprediction_rate": None}
    error = predicted - actual
    underprediction, overprediction = predicted < actual, predicted > actual
    return {"sample_count": int(len(actual)), "mae": float(error.abs().mean()), "rmse": float(np.sqrt(np.mean(np.square(error)))), "mean_signed_error": float(error.mean()), "median_signed_error": float(error.median()), "underprediction_count": int(underprediction.sum()), "underprediction_rate": float(underprediction.mean()), "overprediction_count": int(overprediction.sum()), "overprediction_rate": float(overprediction.mean())}


def _classwise_metrics(candidates: pd.DataFrame, predictions: pd.Series) -> dict:
    """Regression/error metrics by true class, including explicitly small groups."""
    result = {}
    for class_id in (0, 1, 2):
        rows = candidates["Ground_Truth_Class"] == class_id
        metrics = _regression_metrics(candidates.loc[rows, TARGET_COLUMN], predictions.loc[rows], candidates.loc[rows, "Iddq_24h_uA"])
        metrics["sample_count"] = int(rows.sum())
        metrics["interpretation_note"] = "Too few samples for a stable class-level estimate." if rows.sum() < 10 else None
        result[f"Ground_Truth_Class_{class_id}"] = metrics
    return result


def _temporal_diagnostics(raw_df: pd.DataFrame) -> dict:
    """Describe generator-induced temporal dependence without changing it."""
    transitions = (("0h_to_24h", "Iddq_0h_uA", "Iddq_24h_uA"), ("24h_to_96h", "Iddq_24h_uA", "Iddq_96h_uA"), ("96h_to_168h", "Iddq_96h_uA", "Iddq_168h_uA"))
    result = {}
    for name, start, end in transitions:
        delta = raw_df[end] - raw_df[start]
        result[name] = {"pearson_correlation": float(raw_df[start].corr(raw_df[end])), "delta_mean_uA": float(delta.mean()), "delta_median_uA": float(delta.median()), "delta_p95_uA": float(delta.quantile(.95))}
    persistence_error = (raw_df["Iddq_24h_uA"] - raw_df[TARGET_COLUMN]).abs()
    result["24h_persistence_error_by_class"] = {f"Ground_Truth_Class_{class_id}": {"sample_count": int((raw_df["Ground_Truth_Class"] == class_id).sum()), "mae": float(persistence_error[raw_df["Ground_Truth_Class"] == class_id].mean())} for class_id in (0, 1, 2)}
    return result


def run_full_pipeline(test_size: float = 0.20, random_state: int = 42) -> tuple[pd.DataFrame, dict]:
    """Evaluate the cascade on a reproducible, class-stratified holdout.

    Components—not lots—are split so Module A can score every test component
    against reference statistics learned only from training peers in its lot.
    """
    dataset_path = Path(__file__).with_name("synthetic_burnin_dataset.csv")
    raw_df = pd.read_csv(dataset_path) if dataset_path.exists() else generate_burnin_dataset(random_state=random_state)
    train_df, test_df = train_test_split(raw_df, test_size=test_size, random_state=random_state, stratify=raw_df["Ground_Truth_Class"])
    train_df, test_df = train_df.copy(), test_df.copy()

    # Reference stage: test values never influence Module-A statistics.
    module_a = AdvancedScreeningEngine(base_k_sigma=3.0).fit(train_df)
    train_a, test_a = module_a.predict(train_df), module_a.predict(test_df)

    # Train Module B only on Module-A-pass rows from the training partition.
    module_b_train = train_a.loc[train_a["Module_A_Reject"] == 0]
    forecaster = train_drift_forecaster(module_b_train)

    # Inference stage: only unseen Module-A-pass test rows enter Module B.
    module_b_test = test_a.loc[test_a["Module_A_Reject"] == 0]
    b_predictions = predict_drift(forecaster, module_b_test)
    final_test = test_a.copy()
    final_test["Module_B_Evaluated"] = 0
    # Explicit placeholders identify components stopped by Module A.
    final_test["Pred_Iddq_168h_uA"], final_test["Projected_Slope"], final_test["Module_B_Early_Reject"] = final_test["Iddq_24h_uA"], 0.0, 0
    final_test.loc[module_b_test.index, b_predictions.columns] = b_predictions
    final_test.loc[module_b_test.index, "Module_B_Evaluated"] = 1
    final_test["Final_System_Reject"] = final_test["Module_A_Reject"].astype(int) | final_test["Module_B_Early_Reject"].astype(int)

    actual_any_anomaly = final_test["Ground_Truth_Class"] != 0
    module_a_metrics = _classification_metrics(actual_any_anomaly, final_test["Module_A_Reject"] == 1)
    module_b_metrics = _regression_metrics(module_b_test[TARGET_COLUMN], b_predictions["Pred_Iddq_168h_uA"], module_b_test["Iddq_24h_uA"])
    classwise = _classwise_metrics(module_b_test, b_predictions["Pred_Iddq_168h_uA"])
    persistence_comparison = {
        "scope": "Prediction-error reference only; persistence has no slope-based early-rejection capability.",
        "LightGBM_overall": _prediction_error_profile(module_b_test[TARGET_COLUMN], b_predictions["Pred_Iddq_168h_uA"]),
        "persistence_overall": _prediction_error_profile(module_b_test[TARGET_COLUMN], module_b_test["Iddq_24h_uA"]),
        "LightGBM_class_2": _prediction_error_profile(module_b_test.loc[module_b_test["Ground_Truth_Class"] == 2, TARGET_COLUMN], b_predictions.loc[module_b_test["Ground_Truth_Class"] == 2, "Pred_Iddq_168h_uA"]),
        "persistence_class_2": _prediction_error_profile(module_b_test.loc[module_b_test["Ground_Truth_Class"] == 2, TARGET_COLUMN], module_b_test.loc[module_b_test["Ground_Truth_Class"] == 2, "Iddq_24h_uA"]),
    }
    severe_candidates = module_b_test["Ground_Truth_Class"] == 2
    severe_rejected = b_predictions.loc[severe_candidates, "Module_B_Early_Reject"] == 1
    all_b_rejected = b_predictions["Module_B_Early_Reject"] == 1
    severe_count, severe_caught = int(severe_candidates.sum()), int(severe_rejected.sum())
    drift_detection = {
        "true_severe_drift_components_class_2": severe_count,
        "correctly_early_rejected": severe_caught,
        "missed": severe_count - severe_caught,
        "recall": severe_caught / severe_count if severe_count else None,
        "false_negatives": severe_count - severe_caught,
        "false_negative_rate": (severe_count - severe_caught) / severe_count if severe_count else None,
        "class_2_early_rejection_precision": int((module_b_test.loc[all_b_rejected, "Ground_Truth_Class"] == 2).sum()) / int(all_b_rejected.sum()) if all_b_rejected.any() else None,
        "module_b_early_reject_count": int(all_b_rejected.sum()),
        "persistence_baseline": {"projected_slope": 0.0, "early_reject_count": 0, "class_2_recall": 0.0 if severe_count else None, "note": "Persistence predicts Iddq_168h = Iddq_24h; it has no positive-slope early-rejection capability."},
    }
    module_a_false_negatives = final_test.loc[actual_any_anomaly & (final_test["Module_A_Reject"] == 0)].copy()
    false_negative_audit = []
    for _, row in module_a_false_negatives.iterrows():
        false_negative_audit.append({
            "Component_ID": row["Component_ID"], "Lot_ID": row["Lot_ID"], "Ground_Truth_Class": int(row["Ground_Truth_Class"]),
            "Iddq_0h_uA": float(row["Iddq_0h_uA"]), "Iddq_24h_uA": float(row["Iddq_24h_uA"]),
            "Module_A_decision": "pass", "Module_A_reason": row["QA_Engineering_Reason"],
            "Module_B_candidate_status": "entered Module B" if row["Module_B_Evaluated"] else "did not enter Module B",
            "Module_B_predicted_168h_uA": float(row["Pred_Iddq_168h_uA"]) if row["Module_B_Evaluated"] else None,
            "actual_168h_uA": float(row[TARGET_COLUMN]), "projected_slope_uA_per_hr": float(row["Projected_Slope"]) if row["Module_B_Evaluated"] else None,
            "Module_B_early_reject_decision": int(row["Module_B_Early_Reject"]), "final_decision": "reject" if row["Final_System_Reject"] else "pass",
            "decision_feature_audit": "Module B features are 0h/24h-derived only; actual 168h is evaluation-only and is not used by predict_drift.",
        })
    system_recall = _classification_metrics(actual_any_anomaly, final_test["Final_System_Reject"] == 1)
    system_metrics = {"total_test_components": int(len(final_test)), "module_a_rejects": int(final_test["Module_A_Reject"].sum()), "components_entering_module_b": int(final_test["Module_B_Evaluated"].sum()), "module_b_early_rejects": int(final_test["Module_B_Early_Reject"].sum()), "final_rejects": int(final_test["Final_System_Reject"].sum()), "final_passes": int((final_test["Final_System_Reject"] == 0).sum()), "stopped_before_168h_percentage": float(100 * final_test["Module_A_Reject"].mean()), "estimated_chamber_hours_avoided": int(final_test["Module_A_Reject"].sum() * 144), "synthetic_overall_anomaly_detection": _classification_metrics(actual_any_anomaly, final_test["Final_System_Reject"] == 1)}
    report = {"evaluation_note": "Synthetic-data evaluation only; it does not represent real ISRO hardware performance.", "split": {"strategy": "stratified component split by Ground_Truth_Class", "random_state": random_state, "train_samples": int(len(train_df)), "test_samples": int(len(test_df)), "leakage_controls": "Module A is fit only on training data; Module B is trained only on Module-A-pass training rows and evaluated only on Module-A-pass held-out rows."}, "module_a": module_a_metrics, "module_b": {"overall_prediction_metrics": module_b_metrics, "classwise_prediction_metrics": classwise, "directional_bias": {key: module_b_metrics[key] for key in ("mean_signed_error", "median_signed_error", "underprediction_count", "underprediction_rate", "overprediction_count", "overprediction_rate")}, "class_2_directional_bias": {key: classwise["Ground_Truth_Class_2"][key] for key in ("mean_signed_error", "median_signed_error", "underprediction_count", "underprediction_rate", "overprediction_count", "overprediction_rate")}, "persistence_baseline_comparison": persistence_comparison, "drift_detection": drift_detection}, "module_a_false_negative_audit": false_negative_audit, "system": system_metrics, "system_level_recall": system_recall, "slope_threshold_audit": {"threshold_uA_per_hr": forecaster.slope_threshold, "provenance": "Code-level default argument in train_drift_forecaster / DriftForecaster, not selected by pipeline evaluation.", "test_label_tuning": "No: this pipeline does not inspect test labels when setting the threshold.", "historical_selection": "Not documented in the repository; treat as an unvalidated engineering assumption until independently justified or selected on a training-only validation split."}, "synthetic_temporal_data_analysis": _temporal_diagnostics(raw_df)}
    return final_test, report


if __name__ == "__main__":
    _, evaluation_report = run_full_pipeline()
    print(json.dumps(evaluation_report, indent=2, allow_nan=False))
