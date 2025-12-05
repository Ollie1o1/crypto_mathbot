import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import torch

class SignalGenerator:
    """
    Decision Engine combining ML Classification and Kelly Criterion.
    
    Refactored for Production:
    - Dynamic Volatility Barriers (Triple Barrier Method)
    - Removed Hardcoded Regime Filters (ML learns them)
    - Continuous Kelly Criterion for Sizing
    - GPU Acceleration Support
    """
    
    def __init__(self):
        # Check for GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tree_method = 'hist' if self.device == 'cuda' else 'auto'
        
        print(f"Initializing XGBoost on {self.device.upper()}...")
        
        self.model = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=5, 
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method=tree_method,
            device=self.device
        )
        self.is_trained = False

    def save_model(self, filepath: str):
        self.model.save_model(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        self.model.load_model(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")

    def prepare_features(self, kinematics: pd.DataFrame, dsp: pd.DataFrame, fracdiff: pd.Series, regime: pd.DataFrame) -> pd.DataFrame:
        """
        Combine features for the ML model.
        Now includes Regime features (Hurst, Entropy) so the model can learn them.
        """
        features = pd.concat([kinematics, dsp, fracdiff.rename('fracdiff'), regime], axis=1)
        return features.dropna()

    def train_model(self, features: pd.DataFrame, price: pd.Series, time_horizon: int = 12, barrier_multiplier: float = 2.0):
        """
        Train using Dynamic Triple Barrier Method.
        Barrier Width = Daily Volatility * Multiplier
        """
        # Calculate Volatility (Standard Deviation of Returns)
        returns = price.pct_change()
        volatility = returns.rolling(window=24).std() # 24-hour volatility
        
        labels = []
        valid_indices = []
        
        # Align indices
        common_idx = features.index.intersection(price.index).intersection(volatility.index)
        features = features.loc[common_idx]
        price = price.loc[common_idx]
        volatility = volatility.loc[common_idx]
        
        for i in range(len(price) - time_horizon):
            current_price = price.iloc[i]
            current_vol = volatility.iloc[i]
            
            if pd.isna(current_vol) or current_vol == 0:
                continue
                
            # Dynamic Barrier
            barrier = current_vol * barrier_multiplier
            upper = current_price * (1 + barrier)
            lower = current_price * (1 - barrier)
            
            future_window = price.iloc[i+1 : i+time_horizon+1]
            
            hit_upper = future_window[future_window >= upper].index.min()
            hit_lower = future_window[future_window <= lower].index.min()
            
            if pd.notna(hit_upper) and (pd.isna(hit_lower) or hit_upper < hit_lower):
                labels.append(1)
                valid_indices.append(features.index[i])
            elif pd.notna(hit_lower) and (pd.isna(hit_upper) or hit_lower < hit_upper):
                labels.append(0)
                valid_indices.append(features.index[i])
            else:
                # Timed out (Vertical Barrier) - Label 0 (or ignore)
                # For binary classification, we treat timeout as "No Trade" or 0
                labels.append(0)
                valid_indices.append(features.index[i])
                
        # Align features with labels
        X = features.loc[valid_indices]
        y = np.array(labels)
        
        print(f"Training on {len(X)} samples with Dynamic Barriers...")
        self.model.fit(X, y)
        self.is_trained = True

    def predict_signal(self, current_features: pd.DataFrame) -> float:
        """
        Returns probability of hitting upper barrier.
        """
        if not self.is_trained:
            return 0.5
            
        prob = self.model.predict_proba(current_features)[:, 1][0]
        return prob

    def get_leverage(self, probability: float, volatility: float, target_vol: float = 0.05, max_leverage: float = 4.0) -> float:
        """
        Continuous Kelly Criterion with Volatility Targeting.
        
        1. Kelly Fraction = (p - q) / 1 (assuming 1:1 odds for simplicity, or 2p-1)
           Refined: 2 * Probability - 1
        2. Volatility Scalar = Target_Vol / Current_Vol
        3. Leverage = Kelly * Scalar * Capital_Factor (0.5 for Half-Kelly)
        """
        # 1. Kelly Fraction (Directional Conviction)
        # Prob > 0.5 -> Long, Prob < 0.5 -> Short
        # We return signed leverage to indicate direction
        
        raw_kelly = 2 * probability - 1 # Range [-1, 1]
        
        # Filter weak signals
        if abs(raw_kelly) < 0.1: # Equivalent to Prob between 0.45 and 0.55
            return 0.0
            
        # 2. Volatility Scalar
        if volatility <= 0:
            vol_scalar = 1.0
        else:
            vol_scalar = target_vol / volatility
            
        # 3. Final Leverage (Half-Kelly for safety)
        leverage = raw_kelly * vol_scalar * 0.5
        
        # Cap leverage
        leverage = np.clip(leverage, -max_leverage, max_leverage)
        
        return leverage
