import pandas as pd
import numpy as np
from src.features import CryptoKinematics

def test_causality():
    print("Testing Feature Causality...")
    
    # 1. Create a synthetic price series
    np.random.seed(42)
    prices = np.random.randn(500).cumsum() + 100
    series = pd.Series(prices)
    
    kinematics = CryptoKinematics()
    
    # 2. Calculate features on T=0 to T=200
    subset_1 = series.iloc[:200]
    features_1 = kinematics.generate_all_features(subset_1)
    # Check value at index 199 (last known point)
    val_at_199_run1 = features_1.iloc[-1]['phase']
    
    # 3. Calculate features on T=0 to T=201 (One new candle)
    subset_2 = series.iloc[:201]
    features_2 = kinematics.generate_all_features(subset_2)
    # Check value at index 199 (SAME point as before, but now with future knowledge of 201)
    val_at_199_run2 = features_2.iloc[-2]['phase']
    
    print(f"\nValue at Index 199 (Run 1 - End of Data): {val_at_199_run1:.6f}")
    print(f"Value at Index 199 (Run 2 - Future Added): {val_at_199_run2:.6f}")
    
    diff = abs(val_at_199_run1 - val_at_199_run2)
    print(f"Difference: {diff:.6f}")
    
    if diff > 1e-5:
        print("\n[CRITICAL FAIL] Look-Ahead Bias Detected!")
        print("The past changed when the future happened. This feature is non-causal.")
    else:
        print("\n[PASS] Causality Respected.")

if __name__ == "__main__":
    test_causality()
