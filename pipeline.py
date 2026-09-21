"""
pipeline.py
End-to-end Automated Screening Pipeline.
Runs Data Simulation -> Module A (PAT & Covariance) -> Module B (Drift Regressor)
in a single memory-backed sequence.
"""

from synthetic_data_set import generate_burnin_dataset
from module_A import AdvancedScreeningEngine
import module_B as mod_b
import numpy as np
import pandas as pd

def run_full_pipeline():
    print("[1/3] Synthesizing Telemetry...")
    raw_df = generate_burnin_dataset(n_samples=2500)

    print("[2/3] Executing Module A (AEC-Q001 PAT + Multivariate Screening)...")
    engine = AdvancedScreeningEngine(base_k_sigma=3.0)
    engine.fit(raw_df)
    screened_a = engine.predict(raw_df)

    print("[3/3] Executing Module B (Asymmetric Drift Predictor & Early Rejection)...")
    X = mod_b.compute_features(screened_a)
    y = screened_a['Iddq_168h_uA']
    
    # Train on 80% baseline
    split_idx = int(0.8 * len(screened_a))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_df = screened_a.iloc[split_idx:].copy()

    import lightgbm as lgb
    regressor = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4,
        objective=mod_b.asymmetric_space_loss, random_state=42, verbose=-1
    )
    regressor.fit(X_train, y_train)

    preds_168h = regressor.predict(X_test)
    projected_slope = (preds_168h - X_test['Iddq_24h_uA'].values) / 144.0
    early_reject = (projected_slope > 0.04).astype(int)

    test_df['Pred_Iddq_168h_uA'] = np.round(preds_168h, 3)
    test_df['Projected_Slope'] = np.round(projected_slope, 4)
    test_df['Module_B_Early_Reject'] = early_reject
    test_df['Final_System_Reject'] = np.bitwise_or(test_df['Module_A_Reject'], early_reject)

    test_df.to_csv("screened_final_submission.csv", index=False)
    print(" Pipeline complete. Output ready for dashboard display at 'screened_final_submission.csv'.")

if __name__ == "__main__":
    run_full_pipeline()