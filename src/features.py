import numpy as np
import pandas as pd

class CryptoKinematics:
    """
    Feature generation class that transforms raw OHLCV data into robust statistical features.
    
    Principles:
    1.  Strict Causality: No future data leakage. Using rolling windows only.
    2.  Stationarity: Uses Log-Returns as the primary unit.
    3.  Robustness: Focuses on Volatility, Skewness, Kurtosis rather than fragile cycles.
    """
    
    def __init__(self):
        pass

    def generate_all_features(self, close: pd.Series, window_size: int = 100) -> pd.DataFrame:
        """
        Generate all features for the series.
        CAUTION: When used in backtesting, ensure 'close' contains only PAST data up to time T.
        """
        # 1. Base Transformations (Log Returns)
        # ln(P_t / P_{t-1})
        log_returns = np.log(close / close.shift(1)).fillna(0)
        
        # 2. Volatility (Risk)
        # Rolling Standard Deviation of Log Returns
        volatility_short = log_returns.rolling(window=24).std()
        volatility_long = log_returns.rolling(window=window_size).std()
        
        # 3. Higher Moments (Tail Risk)
        skew = log_returns.rolling(window=window_size).skew()
        kurtosis = log_returns.rolling(window=window_size).kurt()
        
        # 4. Momentum (Trend)
        # ROC: P_t / P_{t-N} - 1
        momentum_short = close.pct_change(periods=24)
        momentum_long = close.pct_change(periods=window_size)
        
        # 5. Regime (Hurst) - Optimized for speed/stability if needed, 
        # but for now we stick to standard statistical features which are faster and more robust.
        # Let's add a simple "Efficiency Ratio" (Kaufman) as a proxy for trend quality.
        efficiency_ratio = self.get_efficiency_ratio(close, window=24)
        
        # Combine
        features = pd.DataFrame({
            'log_returns': log_returns,
            'volatility_short': volatility_short,
            'volatility_long': volatility_long,
            'skew': skew,
            'kurtosis': kurtosis,
            'momentum_short': momentum_short,
            'momentum_long': momentum_long,
            'efficiency_ratio': efficiency_ratio
        })
        
        # 6. Normalize (Strictly Causal Rolling Z-Score)
        # We normalize each feature by its OWN rolling history.
        # This ensures that a value of "2.0" today means "2 sigma deviation from the recent mean",
        # which is a stationary signal suitable for ML.
        
        normalized_features = self.rolling_z_score(features, window=window_size*2)
        
        return normalized_features.dropna()

    def rolling_z_score(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Normalize features using a rolling Z-Score to ensure stationarity and causality.
        (Value - RollingMean) / RollingStd
        """
        # Use a large window to capture the "regime" distribution
        rolling_mean = df.rolling(window=window, min_periods=window//2).mean()
        rolling_std = df.rolling(window=window, min_periods=window//2).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, 1e-8)
        
        z_score = (df - rolling_mean) / rolling_std
        
        # Clip outliers to +/- 4 sigma to preserve numerical stability
        return z_score.clip(-4, 4)

    def get_efficiency_ratio(self, close: pd.Series, window: int = 10) -> pd.Series:
        """
        Kaufman Efficiency Ratio: Direction / Volatility
        Abs(P_t - P_{t-n}) / Sum(Abs(P_i - P_{i-1}))
        High values indicate smooth trends, low values indicate chop.
        """
        change = (close - close.shift(window)).abs()
        volatility = (close - close.shift(1)).abs().rolling(window=window).sum()
        
        return change / volatility.replace(0, 1e-8)

