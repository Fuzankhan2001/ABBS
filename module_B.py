"""168-hour leakage-drift forecasting for the ABBS screening cascade."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ("Iddq_0h_uA", "Iddq_24h_uA", "delta_24_0", "drift_velocity", "drift_ratio", "kinetic_rate_proxy")
TARGET_COLUMN = "Iddq_168h_uA"
# Selected on a fixed validation split within the training partition (seed
# 31415), maximizing severe-drift recall before minimizing false rejections.
# It was not selected against the held-out test partition.
DEFAULT_SLOPE_THRESHOLD = 0.12


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare only telemetry available at 24h for a 168h forecast."""
    feats = pd.DataFrame(index=df.index)
    val_0h, val_24h = df["Iddq_0h_uA"], df["Iddq_24h_uA"]
    feats["Iddq_0h_uA"], feats["Iddq_24h_uA"] = val_0h, val_24h
    feats["delta_24_0"] = np.maximum(0, val_24h - val_0h)
    feats["drift_velocity"] = feats["delta_24_0"] / 24.0
    feats["drift_ratio"] = val_24h / (val_0h + 1e-6)
    feats["kinetic_rate_proxy"] = feats["delta_24_0"] / (24.0 ** 0.25)
    return feats.loc[:, FEATURE_COLUMNS]


def asymmetric_space_loss(y_true, y_pred):
    """Penalize underestimating degradation more heavily than overestimation."""
    residual = y_true - y_pred
    penalty_weight = 10.0
    return (np.where(residual > 0, -2.0 * penalty_weight * residual, -2.0 * residual), np.where(residual > 0, 2.0 * penalty_weight, 2.0))


@dataclass
class DriftForecaster:
    model: lgb.LGBMRegressor
    slope_threshold: float = DEFAULT_SLOPE_THRESHOLD


def train_drift_forecaster(training_df: pd.DataFrame, slope_threshold: float = DEFAULT_SLOPE_THRESHOLD) -> DriftForecaster:
    """Fit LightGBM only on labelled Module-A-pass training candidates."""
    if TARGET_COLUMN not in training_df.columns:
        raise ValueError(f"'{TARGET_COLUMN}' is required to train the drift forecaster.")
    if training_df.empty:
        raise ValueError("Cannot train without Module-A-pass training components.")
    model = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, objective=asymmetric_space_loss, random_state=42, verbose=-1)
    model.fit(compute_features(training_df), training_df[TARGET_COLUMN])
    return DriftForecaster(model=model, slope_threshold=slope_threshold)


def predict_drift(forecaster: DriftForecaster, candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Run inference for already Module-A-approved components only."""
    if candidates_df.empty:
        return pd.DataFrame(index=candidates_df.index, columns=["Pred_Iddq_168h_uA", "Projected_Slope", "Module_B_Early_Reject"])
    features = compute_features(candidates_df)
    predictions = forecaster.model.predict(features)
    projected_slope = (predictions - features["Iddq_24h_uA"].to_numpy()) / 144.0
    return pd.DataFrame({"Pred_Iddq_168h_uA": np.round(predictions, 3), "Projected_Slope": np.round(projected_slope, 4), "Module_B_Early_Reject": (projected_slope > forecaster.slope_threshold).astype(int)}, index=candidates_df.index)


@lru_cache(maxsize=1)
def get_runtime_forecaster() -> DriftForecaster:
    """Create one development model for backend inference, never per request.

    Production should load a versioned model artifact validated offline.
    """
    from module_A import AdvancedScreeningEngine
    from synthetic_data_set import generate_burnin_dataset
    dataset_path = Path(__file__).with_name("synthetic_burnin_dataset.csv")
    training_df = pd.read_csv(dataset_path) if dataset_path.exists() else generate_burnin_dataset(random_state=42)
    screened_training = AdvancedScreeningEngine(base_k_sigma=3.0).fit(training_df).predict(training_df)
    return train_drift_forecaster(screened_training.loc[screened_training["Module_A_Reject"] == 0])


def run_drift_forecast(df: pd.DataFrame, forecaster: DriftForecaster | None = None) -> pd.DataFrame:
    """Apply strict cascade inference; Module-A rejects never enter Module B."""
    if "Module_A_Reject" not in df.columns:
        raise ValueError("Module A must run before Module B.")
    out_df = df.copy()
    eligible = out_df["Module_A_Reject"].astype(int) == 0
    out_df["Module_B_Evaluated"] = eligible.astype(int)
    # Display-safe placeholders identify stopped parts; they are not forecasts.
    out_df["Pred_Iddq_168h_uA"] = out_df["Iddq_24h_uA"].astype(float)
    out_df["Projected_Slope"] = 0.0
    out_df["Module_B_Early_Reject"] = 0
    if eligible.any():
        predictions = predict_drift(forecaster or get_runtime_forecaster(), out_df.loc[eligible])
        out_df.loc[eligible, predictions.columns] = predictions
    out_df["Final_System_Reject"] = out_df["Module_A_Reject"].astype(int) | out_df["Module_B_Early_Reject"].astype(int)
    return out_df
