import pandas as pd
import numpy as np
from xgboost import XGBClassifier
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class SignalGenerator:
    """
    Decision Engine with Optimized Hyperparameters and GPU Support.
    """
    
    def __init__(self):
        # Check for GPU
        if HAS_TORCH and torch.cuda.is_available():
            self.device = 'cuda'
            self.tree_method = 'hist'
        else:
            self.device = 'cpu'
            self.tree_method = 'auto'
        
        print(f"Initializing Model on {self.device.upper()}...")
        
        # OPTIMIZED PARAMETERS (Found via Optuna on 3000 candles)
        self.model = XGBClassifier(
            n_estimators=141,
            learning_rate=0.012290387288834912,
            max_depth=7,
            subsample=0.6000064397230108,
            colsample_bytree=0.9602500828990242,
            gamma=0.4205312152212468,
            min_child_weight=10,
            
            # Standard Config
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=-1,
            device=self.device,
            tree_method=self.tree_method
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
        features = pd.concat([kinematics, dsp, regime, fracdiff.rename('fracdiff')], axis=1)
        return features.dropna()

    def train_model(self, features: pd.DataFrame, price: pd.Series, time_horizon: int = 12, barrier_multiplier: float = 2.0):
        # 1. Volatility Calculation
        returns = price.pct_change()
        volatility = returns.rolling(window=24).std()
        
        labels = []
        valid_indices = []
        
        # Align indices
        common_idx = features.index.intersection(price.index).intersection(volatility.index)
        features = features.loc[common_idx]
        price = price.loc[common_idx]
        volatility = volatility.loc[common_idx]
        
        # 2. Triple Barrier Labeling
        # We can speed this up, but the loop is safer for logic verification
        for i in range(len(price) - time_horizon):
            current_vol = volatility.iloc[i]
            if pd.isna(current_vol) or current_vol == 0:
                continue
                
            current_price = price.iloc[i]
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
                labels.append(0)
                valid_indices.append(features.index[i])
                
        X = features.loc[valid_indices]
        y = np.array(labels)
        
        print(f"Training on {len(X)} samples...")
        self.model.fit(X, y)
        self.is_trained = True

    def predict_signal(self, current_features: pd.DataFrame) -> float:
        if not self.is_trained:
            return 0.5
        # Ensure input is on the correct device if using PyTorch tensors, but DataFrame is fine
        return self.model.predict_proba(current_features)[:, 1][0]

    def get_leverage(self, probability: float, volatility: float, target_vol: float = 0.05, max_leverage: float = 4.0) -> float:
        # Kelly Fraction (2p - 1)
        raw_kelly = 2 * probability - 1 
        
        if abs(raw_kelly) < 0.1: 
            return 0.0
            
        # Volatility Scalar
        if volatility <= 0:
            vol_scalar = 1.0
        else:
            vol_scalar = target_vol / volatility
            
        # Half-Kelly
        leverage = raw_kelly * vol_scalar * 0.5
        return np.clip(leverage, -max_leverage, max_leverage)

    def regime_filter(self, entropy: float, hurst: float) -> str:
        """
        Identify Market Regime.
        """
        if pd.isna(hurst) or pd.isna(entropy):
            return 'WAIT'
            
        if hurst > 0.55:
            return 'TREND'
        elif hurst < 0.45:
            return 'MEAN_REVERSION'
        else:
            return 'RANDOM'

    def get_trigger(self, regime: str, hurst: float, acceleration: float, probability: float) -> str:
        """
        Generate Trade Trigger based on Model Probability and Regime.
        """
        # 1. Filter by Regime
        if regime == 'RANDOM' or regime == 'WAIT':
            return 'HOLD'
            
        # 2. Probability Thresholds
        # 0 - 1 Score from XGBoost
        # > 0.55 -> LONG
        # < 0.45 -> SHORT
        
        if probability > 0.55:
            return 'LONG'
        elif probability < 0.45:
            return 'SHORT'
            
        return 'HOLD'