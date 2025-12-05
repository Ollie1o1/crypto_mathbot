import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

class SignalGenerator:
    """
    Decision Engine combining Regime Filtering, ML Classification, and Trigger Logic.
    """
    
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=5, 
            objective='binary:logistic',
            eval_metric='logloss'
        )
        self.is_trained = False

    def regime_filter(self, entropy: float, hurst: float, entropy_threshold: float = 0.8) -> str:
        """
        Layer 1: Regime Filter (Hard Rules).
        Returns 'TRADE' or 'HOLD'.
        """
        # If High Entropy (Chaotic) or Random Walk (Hurst ~ 0.5), stay cash.
        # Hurst < 0.45 (Mean Reversion) or > 0.55 (Trending) are tradeable.
        # 0.45 <= Hurst <= 0.55 is Random Walk.
        
        if entropy > entropy_threshold:
            return 'HOLD'
        
        if 0.45 <= hurst <= 0.55:
            return 'HOLD'
            
        return 'TRADE'

    def prepare_features(self, kinematics: pd.DataFrame, dsp: pd.DataFrame, fracdiff: pd.Series) -> pd.DataFrame:
        """
        Combine features for the ML model.
        """
        features = pd.concat([kinematics, dsp, fracdiff.rename('fracdiff')], axis=1)
        return features.dropna()

    def train_model(self, features: pd.DataFrame, price: pd.Series, time_horizon: int = 12, barrier: float = 0.01):
        """
        Train the XGBoost Classifier using Triple Barrier Method labeling.
        """
        # Create Labels
        # Label 1 if price hits upper barrier before lower barrier within time_horizon
        # Label 0 otherwise
        
        labels = []
        for i in range(len(price) - time_horizon):
            current_price = price.iloc[i]
            upper = current_price * (1 + barrier)
            lower = current_price * (1 - barrier)
            
            future_window = price.iloc[i+1 : i+time_horizon+1]
            
            hit_upper = future_window[future_window >= upper].index.min()
            hit_lower = future_window[future_window <= lower].index.min()
            
            if pd.notna(hit_upper) and (pd.isna(hit_lower) or hit_upper < hit_lower):
                labels.append(1)
            else:
                labels.append(0)
                
        # Align features with labels
        X = features.iloc[:len(labels)]
        y = np.array(labels)
        
        # Train
        self.model.fit(X, y)
        self.is_trained = True

    def predict_signal(self, current_features: pd.DataFrame) -> float:
        """
        Layer 2: ML Classifier Output.
        Returns probability of hitting upper barrier.
        """
        if not self.is_trained:
            return 0.5 # Neutral if not trained
            
        # Expecting a single row dataframe
        prob = self.model.predict_proba(current_features)[:, 1][0]
        return prob

    def get_trigger(self, regime_action: str, hurst: float, acceleration: float, ml_prob: float) -> str:
        """
        Layer 3: Trigger Logic.
        Returns 'LONG', 'SHORT', or 'HOLD'.
        """
        if regime_action == 'HOLD':
            return 'HOLD'
            
        # Trending Regime Logic
        if hurst > 0.55:
            if acceleration > 0 and ml_prob > 0.65:
                return 'LONG'
            if acceleration < 0 and ml_prob < 0.35:
                return 'SHORT'
                
        # Mean Reverting Logic (Optional extension based on prompt implying Trade Grid/Counter-trend)
        # Prompt only specified triggers for Trending Regime explicitly in "Layer 3: Trigger Logic" section.
        # "Execute LONG if: Regime is Trending... Execute SHORT if: Regime is Trending..."
        # So we will stick to that for now.
        
        return 'HOLD'
