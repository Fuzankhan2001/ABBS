"""
module_a.py
AEC-Q001 Adaptive Dynamic Part Average Testing (PAT) & 
Multivariate Covariance (Mahalanobis Distance) Engine.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import mahalanobis


class AdvancedScreeningEngine:
    def __init__(self, base_k_sigma=3.0):
        self.base_k_sigma = base_k_sigma
        self.lot_diagnostics = {}
        self.lot_covariance_models = {}

    def fit(self, df):
        self.lot_diagnostics = {}
        self.lot_covariance_models = {}

        for lot, group in df.groupby('Lot_ID'):
            vals_0h = group['Iddq_0h_uA'].values
            
            # 1. Distribution Diagnostics (Lot Health)
            lot_skew = skew(vals_0h)
            lot_kurt = kurtosis(vals_0h)
            
            is_contaminated_lot = bool((lot_skew > 0.8) or (lot_kurt > 1.5))
            adaptive_k = self.base_k_sigma * (0.85 if is_contaminated_lot else 1.0)

            # Robust Non-Parametric Limits (AEC-Q001 via Median & IQR)
            median = float(np.median(vals_0h))
            iqr = float(np.percentile(vals_0h, 75) - np.percentile(vals_0h, 25))
            robust_sigma = 0.7413 * iqr

            upper_pat = median + (adaptive_k * robust_sigma)
            lower_pat = max(0.1, median - (adaptive_k * robust_sigma))

            self.lot_diagnostics[lot] = {
                'median': round(median, 3),
                'robust_sigma': round(robust_sigma, 3),
                'skewness': round(float(lot_skew), 2),
                'kurtosis': round(float(lot_kurt), 2),
                'is_contaminated': is_contaminated_lot,
                'adaptive_k': round(float(adaptive_k), 2),
                'upper_pat': round(float(upper_pat), 3),
                'lower_pat': round(float(lower_pat), 3)
            }

            # 2. Multivariate Covariance [Iddq_0h, Iddq_24h]
            bivariate_data = group[['Iddq_0h_uA', 'Iddq_24h_uA']].values
            mean_vec = np.mean(bivariate_data, axis=0)
            cov_mat = np.cov(bivariate_data, rowvar=False)
            inv_cov_mat = np.linalg.pinv(cov_mat)

            self.lot_covariance_models[lot] = {
                'mean': mean_vec,
                'inv_cov': inv_cov_mat
            }
        return self

    def evaluate_component(self, row):
        lot = row['Lot_ID']
        val_0h = row['Iddq_0h_uA']
        val_24h = row['Iddq_24h_uA']

        diag = self.lot_diagnostics.get(lot)
        cov_model = self.lot_covariance_models.get(lot)

        if not diag or not cov_model:
            return {
                'Mahalanobis_Distance': 0.0,
                'PAT_Violated': 0,
                'Covariance_Violated': 0,
                'Module_A_Reject': 0,
                'QA_Engineering_Reason': 'UNKNOWN LOT'
            }

        # Stage 1: Adaptive Dynamic PAT
        pat_violation = (val_0h > diag['upper_pat']) or (val_0h < diag['lower_pat'])

        # Stage 2: Multivariate Mahalanobis Distance
        point = np.array([val_0h, val_24h])
        m_dist = mahalanobis(point, cov_model['mean'], cov_model['inv_cov'])
        mahalanobis_violation = m_dist > 3.03

        is_reject = bool(pat_violation or mahalanobis_violation)

        if pat_violation and mahalanobis_violation:
            reason = f"CRITICAL: Baseline ({val_0h:.2f}µA > {diag['upper_pat']:.2f}µA) & Covariance Anomaly (D_M={m_dist:.2f})"
        elif pat_violation:
            reason = f"AEC-Q001 Outlier: Baseline {val_0h:.2f}µA exceeded adaptive limit ({diag['upper_pat']:.2f}µA)"
        elif mahalanobis_violation:
            reason = f"Covariance Breakdown: Abnormal 0h-24h trajectory vector (D_M={m_dist:.2f})"
        else:
            reason = "NOMINAL: Conforms to dynamic lot distribution envelopes"

        return {
            'Mahalanobis_Distance': round(float(m_dist), 2),
            'PAT_Violated': int(pat_violation),
            'Covariance_Violated': int(mahalanobis_violation),
            'Module_A_Reject': int(is_reject),
            'QA_Engineering_Reason': reason
        }

    def predict(self, df):
        audit_records = [self.evaluate_component(row) for _, row in df.iterrows()]
        audit_df = pd.DataFrame(audit_records)
        return pd.concat([df.reset_index(drop=True), audit_df], axis=1)