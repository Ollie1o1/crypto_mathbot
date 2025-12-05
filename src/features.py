import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, hilbert

class CryptoKinematics:
    """
    Feature generation class that transforms raw OHLCV data into mathematical features
    based on Calculus, Signal Processing, and Chaos Theory.
    
    Refactored for Stationarity:
    - Uses Log-Returns for Kinematics (Scale Invariance)
    - Implements Z-Score Normalization
    """
    
    def __init__(self):
        pass

    def generate_all_features(self, close: pd.Series) -> pd.DataFrame:
        """
        Generate all features for the entire series.
        """
        # 1. Base Transformations
        log_price = np.log(close)
        smooth_log_price = self.get_smooth_price(log_price)
        
        # 2. Kinematics (on Log Price)
        kinematics = self.get_kinematics(smooth_log_price)
        
        # 3. DSP (on Detrended Log Price)
        dsp = self.get_dsp_features(log_price)
        
        # 4. Regime (on Log Returns)
        regime = self.get_regime_features(close) # Hurst/Entropy handle their own returns calc
        
        # 5. FracDiff (on Log Price)
        fracdiff = self.get_fracdiff(log_price)
        
        # Combine
        features = pd.concat([kinematics, dsp, regime, fracdiff.rename('fracdiff')], axis=1)
        
        # 6. Normalize
        # Z-Score Normalization with rolling window to prevent look-ahead bias in training
        # But for 'generate_all_features' (batch), we can use a large rolling window
        # or just normalize the whole set if it's for training. 
        # Ideally, we use a rolling window.
        
        normalized_features = self.z_score_normalize(features)
        
        return normalized_features

    def z_score_normalize(self, df: pd.DataFrame, window: int = 2000) -> pd.DataFrame:
        """
        Normalize features using a rolling Z-Score to ensure stationarity.
        (Value - Mean) / Std
        """
        rolling_mean = df.rolling(window=window, min_periods=100).mean()
        rolling_std = df.rolling(window=window, min_periods=100).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, 1e-8)
        
        z_score = (df - rolling_mean) / rolling_std
        
        # Clip outliers to +/- 4 sigma
        return z_score.clip(-4, 4)

    def get_smooth_price(self, series: pd.Series, window_length: int = 11) -> pd.Series:
        """
        Apply EMA smoothing.
        """
        return series.ewm(span=window_length).mean()

    def get_kinematics(self, smooth_log_price: pd.Series) -> pd.DataFrame:
        """
        Calculate Velocity and Acceleration on Log-Prices.
        Velocity = diff(log_price) ~= Returns
        Acceleration = diff(Velocity)
        """
        velocity = smooth_log_price.diff()
        acceleration = velocity.diff()
        return pd.DataFrame({'velocity': velocity, 'acceleration': acceleration})

    def get_dsp_features(self, log_price: pd.Series) -> pd.DataFrame:
        """
        Extract Analytic Signal features.
        """
        # Detrend log-price for Hilbert
        # Using a high-pass filter (price - rolling_mean)
        centered = log_price - log_price.rolling(window=20).mean()
        centered = centered.fillna(0)
        
        analytic_signal = hilbert(centered.values)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.angle(analytic_signal)
        
        return pd.DataFrame({
            'amplitude': amplitude_envelope,
            'phase': instantaneous_phase
        }, index=log_price.index)

    def get_regime_features(self, close: pd.Series, window: int = 100) -> pd.DataFrame:
        """
        Calculate Hurst Exponent and Shannon Entropy.
        """
        hurst = close.rolling(window=window).apply(self._calculate_hurst_exponent, raw=True)
        entropy = close.rolling(window=window).apply(self._calculate_shannon_entropy, raw=True)
        
        return pd.DataFrame({'hurst': hurst, 'entropy': entropy})

    def _calculate_hurst_exponent(self, ts: np.ndarray) -> float:
        """
        Calculate the Hurst Exponent.
        """
        lags = range(2, 20)
        # Standard deviation of differences
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        
        # Avoid log(0)
        if np.any(np.array(tau) <= 0):
            return 0.5
            
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0 

    def _calculate_shannon_entropy(self, ts: np.ndarray) -> float:
        """
        Calculate Shannon Entropy of returns.
        """
        returns = np.diff(ts) / ts[:-1]
        # Handle zeros/NaNs
        returns = returns[~np.isnan(returns)]
        if len(returns) == 0:
            return 0.0
            
        hist, _ = np.histogram(returns, bins=10, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist))

    def get_fracdiff(self, log_price: pd.Series, d: float = 0.4, window: int = 20) -> pd.Series:
        """
        Apply Fractional Differentiation to Log-Prices.
        """
        weights = self._get_weights_ffd(d, window)
        res = 0
        for k in range(window):
            res += weights[k] * log_price.shift(k)
            
        return res.dropna()

    def _get_weights_ffd(self, d: float, size: int) -> np.ndarray:
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] * (d - k + 1) / k
            w.append(w_k)
        return np.array(w)
