# tune_parameters.py
import pandas as pd
import numpy as np
import asyncio
from src.data import fetch_historical_data
from src.features import CryptoKinematics
from src.strategy import SignalGenerator

# MOCK Backtest Function for Speed (No plotting, just numbers)
def fast_backtest(df, strategy, kinematics, eff_threshold, window_size, long_only=False):
    # 1. Re-generate features with specific window
    feats = kinematics.generate_all_features(df['close'], window_size=window_size)
    
    # 2. Simulate
    
    # Alignment
    common = feats.index.intersection(df.index)
    feats = feats.loc[common]
    prices = df['close'].loc[common]
    
    # Predict
    probs = strategy.model.predict_proba(feats)[:, 1]
    
    # Apply Regime Filter (Vectorized)
    # If Efficiency < Threshold, Prob = 0.5
    eff_ratios = feats['efficiency_ratio']
    probs = np.where(eff_ratios < eff_threshold, 0.5, probs)
    
    # Generate Signals
    # Long: > 0.55, Short: < 0.45
    longs = probs > 0.55
    shorts = probs < 0.45
    
    positions = np.zeros(len(probs))
    positions[longs] = 1
    
    if not long_only:
        positions[shorts] = -1
    
    # Calculate Returns
    price_ret = prices.pct_change().fillna(0)
    strategy_ret = positions * price_ret.shift(-1).fillna(0) 
    
    # Fees
    fee = 0.001
    pos_change = np.abs(np.diff(positions, prepend=0))
    fees = pos_change * fee
    
    net_ret = strategy_ret - fees
    
    # Final Metric
    total_return = np.sum(net_ret)
    return total_return

async def main():
    print("Fetching Data (4H Timeframe)...")
    # Note: Fetching 4H data. 6000 candles * 4H = 1000 days (~3 years)
    df = await fetch_historical_data(symbol='BTC/USDT', timeframe='4h', limit=6000)
    
    print(f"Fetched {len(df)} candles.")
    
    # Split Train/Test
    # Adjusted for smaller datasets if necessary
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"Training on {len(train_df)} samples, Testing on {len(test_df)} samples.")
    
    kinematics = CryptoKinematics()
    strategy = SignalGenerator()
    
    # Base Training
    print("Training Base Model...")
    base_feats = kinematics.generate_all_features(train_df['close'], window_size=200)
    strategy.train_model(base_feats, train_df['close'])
    
    print("\n--- Starting Grid Search ---")
    best_score = -100
    best_params = {}
    
    # Grid Search - Expanded
    eff_thresholds = [0.3] # Fix one var to reduce noise/runtime
    windows = [24, 50, 100, 200]
    modes = [True, False] # Long Only?
    
    for w in windows:
        for mode in modes:
            for eff in eff_thresholds:
                score = fast_backtest(test_df, strategy, kinematics, eff_threshold=eff, window_size=w, long_only=mode)
                mode_str = "LONG_ONLY" if mode else "LONG_SHORT"
                print(f"Window: {w} | Mode: {mode_str} | Return: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_params = {'window': w, 'efficiency': eff, 'long_only': mode}
                
    print("\nWINNER:")
    print(best_params)
    print(f"Return: {best_score:.4f}")

if __name__ == "__main__":
    asyncio.run(main())