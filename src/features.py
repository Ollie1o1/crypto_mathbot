import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, hilbert
# from fracdiff.sklearn import Fracdiff # Removed due to install issues

class CryptoKinematics:
    """
    Feature generation class that transforms raw OHLCV data into mathematical features
    based on Calculus, Signal Processing, and Chaos Theory.
    """
    
    def __init__(self):
        pass

    def get_smooth_price(self, close: pd.Series, window_length: int = 11, polyorder: int = 2) -> pd.Series:
        """
        Apply a Savitzky-Golay Filter to the Close price to remove noise.
        """
        # Use EMA for Causal Smoothing (Look-Ahead Bias Free)
        return close.ewm(span=window_length).mean()

    def get_kinematics(self, smooth_price: pd.Series) -> pd.DataFrame:
        """
        Calculate Velocity (1st derivative) and Acceleration (2nd derivative).
        """
        velocity = smooth_price.diff()
        acceleration = velocity.diff()
        return pd.DataFrame({'velocity': velocity, 'acceleration': acceleration})

    def get_zero_crossings(self, acceleration: pd.Series) -> pd.Series:
        """
        Detect Zero-Crossings of Acceleration.
        Returns a signal: 1 (Bottom/Bullish), -1 (Top/Bearish), 0 (Neutral).
        """
        # Sign change detection
        signs = np.sign(acceleration)
        diff_signs = signs.diff()
        
        # If diff is non-zero, a crossing occurred
        # If acceleration goes from + to -, diff is -2 (Top)
        # If acceleration goes from - to +, diff is +2 (Bottom)
        
        signals = pd.Series(0, index=acceleration.index)
        signals[diff_signs == -2] = -1 # Top
        signals[diff_signs == 2] = 1   # Bottom
        
        return signals

    def get_dsp_features(self, close: pd.Series) -> pd.DataFrame:
        """
        Extract Analytic Signal (Hilbert Transform), Instantaneous Phase, and Amplitude.
        """
        # Detrending is often necessary for Hilbert Transform to work well on price
        # We'll use a simple difference or log return for the transform input, 
        # or detrend the price itself. Using log returns is safer for stationarity.
        # However, the prompt asks for "Analytic Signal... of the price". 
        # Hilbert on raw price can be unstable due to trend. 
        # Let's apply it to the detrended price (price - smooth_price) or similar.
        # For now, let's try applying to the centered price.
        
        centered_price = close - close.rolling(window=20).mean()
        centered_price = centered_price.fillna(0)
        
        analytic_signal = hilbert(centered_price.values)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.angle(analytic_signal)
        
        return pd.DataFrame({
            'amplitude': amplitude_envelope,
            'phase': instantaneous_phase
        }, index=close.index)

    def get_regime_features(self, close: pd.Series, window: int = 100) -> pd.DataFrame:
        """
        Calculate Hurst Exponent and Shannon Entropy.
        """
        hurst = close.rolling(window=window).apply(self._calculate_hurst_exponent, raw=True)
        entropy = close.rolling(window=window).apply(self._calculate_shannon_entropy, raw=True)
        
        return pd.DataFrame({'hurst': hurst, 'entropy': entropy})

    def _calculate_hurst_exponent(self, ts: np.ndarray) -> float:
        """
        Calculate the Hurst Exponent of a time series.
        Simplified R/S analysis.
        """
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0 

    def _calculate_shannon_entropy(self, ts: np.ndarray) -> float:
        """
        Calculate Shannon Entropy of returns distribution.
        """
        # Discretize returns into bins
        returns = np.diff(ts) / ts[:-1]
        hist, bin_edges = np.histogram(returns, bins=10, density=True)
        # Filter zero values to avoid log(0)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist))

    def get_fracdiff(self, close: pd.Series, d: float = 0.4, window: int = 20) -> pd.Series:
        """
        Apply Fractional Differentiation using Fixed Width Window (FFD).
        """
        # Custom implementation since fracdiff package is missing
        weights = self._get_weights_ffd(d, window)
        res = 0
        for k in range(window):
            res += weights[k] * close.shift(k)
            
        return res.dropna()

    def _get_weights_ffd(self, d: float, size: int) -> np.ndarray:
        """
        Calculate weights for fractional differentiation.
        """
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] * (d - k + 1) / k
            w.append(w_k)
        return np.array(w)
