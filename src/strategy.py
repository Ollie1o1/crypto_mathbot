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
    Robust Decision Engine using XGBoost with sensible defaults.
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
        
        # ROBUST DEFAULTS
        # Low learning rate + shallow depth = prevents overfitting
        self.model = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=-1,
            device=self.device,
            tree_method=self.tree_method,
            early_stopping_rounds=50 # Use valid set during training
        )
        self.is_trained = False

    def save_model(self, filepath: str):
        self.model.save_model(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        self.model.load_model(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")

    def prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Pass-through for the new feature set. 
        Ensures we drop NaNs before feeding to model.
        """
        return features.dropna()

    def train_model(self, features: pd.DataFrame, price: pd.Series, time_horizon: int = 1):
        """
        Train the model to predict next-bar return sign (or decent threshold).
        Target: Returns > Transaction Costs (e.g. 0.1%)
        """
        # 1. Create Target
        # Predict: Will price rise more than 0.1% in next 'time_horizon' candles?
        future_returns = price.shift(-time_horizon) / price - 1
        
        # Target: 1 (Buy) if Ret > 0.001 (Cost+Slippage), 0 (Short/Hold) if Ret < -0.001
        # For simplicity, let's try standard directional classification first:
        # 1 if Ret > 0, 0 if Ret < 0
        
        y = (future_returns > 0).astype(int)
        
        # Align
        common_idx = features.index.intersection(y.index)
        X = features.loc[common_idx]
        y = y.loc[common_idx]
        
        # Train Test Split for Early Stopping (Chronological)
        split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]
        
        print(f"Training on {len(X_train)} samples, Validating on {len(X_val)}...")
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        self.is_trained = True
        
        # Feature Importance
        print("\nFeature Importance:")
        try:
            fim = self.model.feature_importances_
            cols = X.columns
            for f, v in sorted(zip(cols, fim), key=lambda x: x[1], reverse=True):
                print(f"{f}: {v:.4f}")
        except:
            pass

    def predict_signal(self, current_features: pd.DataFrame) -> float:
        if not self.is_trained:
            return 0.5
        # Return probability of Class 1 (Up)
        return self.model.predict_proba(current_features)[:, 1][0]

    def get_action(self, probability: float, threshold: float = 0.55) -> str:
        """
        Simple Action Trigger.
        """
        if probability > threshold:
            return 'LONG'
        elif probability < (1 - threshold):
            return 'SHORT'
        return 'HOLD'

    def get_leverage(self, probability: float, volatility: float, target_vol: float = 0.05, max_leverage: float = 3.0) -> float:
        """
        Continuous Kelly-like sizing.
        """
        # Probability Confidence (0.5 to 1.0) -> Scaled to 0 to 1
        confidence = abs(probability - 0.5) * 2
        
        if confidence < 0.1: # Minimum confidence filter
            return 0.0
            
        # Volatility Scalar (Risk Parity)
        if volatility <= 0: volatility = 0.01
        vol_scalar = target_vol / volatility
        
        raw_leverage = confidence * vol_scalar
        
        return np.clip(raw_leverage, 0, max_leverage)