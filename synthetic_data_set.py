"""
synthetic_data_set.py
Synthesizes burn-in test data compliant with MIL-STD-883 Method 1015
and AEC-Q001 Part Average Testing standards.
"""

import numpy as np
import pandas as pd

def generate_burnin_dataset(n_samples=2500, anomaly_ratio=0.08, random_state=42):
    np.random.seed(random_state)
    
    lots = ['LOT_ALPHA', 'LOT_BETA', 'LOT_GAMMA', 'LOT_DELTA']
    lot_assignments = np.random.choice(lots, size=n_samples, p=[0.3, 0.25, 0.25, 0.2])
    
    lot_profiles = {
        'LOT_ALPHA': {'mu': 10.2, 'sigma': 1.1},
        'LOT_BETA':  {'mu': 13.5, 'sigma': 1.4},
        'LOT_GAMMA': {'mu': 8.8,  'sigma': 0.9},
        'LOT_DELTA': {'mu': 15.0, 'sigma': 1.6},
    }
    
    class_probs = [1.0 - anomaly_ratio, anomaly_ratio * 0.45, anomaly_ratio * 0.55]
    labels = np.random.choice([0, 1, 2], size=n_samples, p=class_probs)
    
    records = []
    
    for idx in range(n_samples):
        lot = lot_assignments[idx]
        label = labels[idx]
        mu = lot_profiles[lot]['mu']
        sig = lot_profiles[lot]['sigma']
        
        if label == 1:
            val_0h = np.random.uniform(mu + (3.8 * sig), 42.0)
        else:
            val_0h = np.random.normal(mu, sig)
            
        A = np.random.uniform(0.18, 0.32)
        n = np.random.uniform(0.20, 0.28)
        noise = lambda: np.random.normal(0, 0.12)
        
        if label == 2:
            val_24h = val_0h + (A * (24 ** n)) + np.random.uniform(1.2, 2.5)
            val_96h = val_24h + np.random.uniform(5.0, 10.0)
            val_168h = val_96h + np.random.uniform(22.0, 48.0)
        else:
            val_24h = val_0h + (A * (24 ** n)) + noise()
            val_96h = val_0h + (A * (96 ** n)) + noise()
            val_168h = val_0h + (A * (168 ** n)) + noise()
            
        records.append({
            'Component_ID': f'ISRO_IC_{idx:05d}',
            'Lot_ID': lot,
            'Iddq_0h_uA': round(float(val_0h), 3),
            'Iddq_24h_uA': round(float(val_24h), 3),
            'Iddq_96h_uA': round(float(val_96h), 3),
            'Iddq_168h_uA': round(float(val_168h), 3),
            'Ground_Truth_Class': int(label)
        })
        
    return pd.DataFrame(records)

if __name__ == "__main__":
    df = generate_burnin_dataset()
    output_path = "synthetic_burnin_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Data saved to: {output_path}")
    print(f"Total components generated: {len(df)}")
    print("\nClass distribution:")
    print(df['Ground_Truth_Class'].value_counts())