import optuna
import pandas as pd
import numpy as np
import asyncio
import argparse
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from src.data import fetch_historical_data
from src.features import CryptoKinematics
import torch

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

async def objective(trial):
    # 1. Hyperparameters to tune
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'gamma': trial.suggest_float('gamma', 0, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist' if torch.cuda.is_available() else 'auto',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 2. Train/Validation Split (Passed from main to avoid re-fetching)
    X_train, X_val, y_train, y_val = DATA['X_train'], DATA['X_val'], DATA['y_train'], DATA['y_val']
    
    # 3. Train Model
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    preds = model.predict_proba(X_val)[:, 1]
    loss = log_loss(y_val, preds)
    
    return loss

DATA = {}

async def main():
    parser = argparse.ArgumentParser(description='Optimize Strategy Hyperparameters')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Symbol to train on')
    parser.add_argument('--limit', type=int, default=2000, help='Number of candles to fetch')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials')
    args = parser.parse_args()
    
    print(f"Fetching {args.limit} candles for {args.symbol}...")
    df = await fetch_historical_data(symbol=args.symbol, limit=args.limit)
    
    print("Generating features...")
    kinematics = CryptoKinematics()
    features = kinematics.generate_all_features(df['close'])
    
    # Prepare Labels (Triple Barrier)
    # Re-using logic from strategy.py roughly, but simplified for optimization
    # We need labels to train.
    
    price = df['close']
    returns = price.pct_change()
    volatility = returns.rolling(window=24).std()
    
    labels = []
    valid_indices = []
    time_horizon = 12
    barrier_multiplier = 2.0
    
    # Align
    common_idx = features.index.intersection(price.index).intersection(volatility.index)
    features = features.loc[common_idx]
    price = price.loc[common_idx]
    volatility = volatility.loc[common_idx]
    
    for i in range(len(price) - time_horizon):
        current_price = price.iloc[i]
        current_vol = volatility.iloc[i]
        
        if pd.isna(current_vol) or current_vol == 0:
            continue
            
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
    
    # Split Data (70% Train, 30% Val) - Time Series Split (No Shuffle)
    split_idx = int(len(X) * 0.7)
    DATA['X_train'] = X.iloc[:split_idx]
    DATA['X_val'] = X.iloc[split_idx:]
    DATA['y_train'] = y[:split_idx]
    DATA['y_val'] = y[split_idx:]
    
    print(f"Starting Optimization ({args.trials} trials)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: asyncio.run(objective(trial)), n_trials=args.trials)
    
    print("\n--- Best Parameters ---")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"Best LogLoss: {study.best_value:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
