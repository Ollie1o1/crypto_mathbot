import pandas as pd
import asyncio
import argparse
import os
from src.data import fetch_historical_data
from src.features import CryptoKinematics
from src.strategy import SignalGenerator

async def main():
    parser = argparse.ArgumentParser(description='Train Strategy Model')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Symbol to train on')
    parser.add_argument('--limit', type=int, default=5000, help='Number of candles to fetch')
    parser.add_argument('--save_path', type=str, default='models/xgb_model.json', help='Path to save the model')
    args = parser.parse_args()

    # 1. Fetch Data
    print(f"Fetching {args.limit} candles for {args.symbol}...")
    df = await fetch_historical_data(symbol=args.symbol, limit=args.limit)
    print(f"Fetched {len(df)} candles.")

    # 2. Generate Features (Causal / Sliding Window)
    print("Generating features (Sliding Window for Causality)...")
    kinematics = CryptoKinematics()
    
    # We must generate features exactly as they appear in live trading:
    # by looking only at the past window.
    window_size = 200 # Fixed mismatch (was 1000), aligned with validation
    features_list = []
    
    # We need at least window_size data points to start
    if len(df) < window_size:
        print("Not enough data for window size.")
        return

    # Loop through the dataset
    # Optimization: To speed up, we can assume early features (pre-Hilbert convergence) are noise 
    # and just start training after window_size.
    
    # Actually, we can just use the slow loop for correctness.
    # It's training, run once.
    
    start_idx = window_size
    for i in range(start_idx, len(df)):
        # Slice: T-Window to T
        window = df.iloc[i-window_size : i+1]
        
        # Compute features
        feats = kinematics.generate_all_features(window['close'])
        
        # Take the last row (Time T)
        if not feats.empty:
            features_list.append(feats.iloc[[-1]])
            
        if i % 500 == 0:
            print(f"Generated {i}/{len(df)} feature rows...", end='\r')
            
    print("\nConcatenating features...")
    features = pd.concat(features_list)
    
    # 3. Prepare for Training
    strategy = SignalGenerator()
    
    # Select features used by the ML model
    # Note: Ensure these match exactly what's used in predict_signal
    # All features from generate_all_features are now used (including hurst/entropy)
    ml_features = features.dropna()
    
    # Align price with features
    valid_indices = ml_features.index
    y_price = df['close'].loc[valid_indices]
    
    if len(ml_features) < 100:
        print("Not enough data to train.")
        return

    # 4. Train
    print(f"Training model on {len(ml_features)} samples...")
    strategy.train_model(ml_features, y_price)
    
    # 5. Save
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    strategy.save_model(args.save_path)
    print(f"Training complete. Model saved to {args.save_path}")

if __name__ == "__main__":
    asyncio.run(main())
