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

    # 2. Generate Features
    print("Generating features...")
    kinematics = CryptoKinematics()
    features = kinematics.generate_all_features(df['close'])
    
    # 3. Prepare for Training
    strategy = SignalGenerator()
    
    # Select features used by the ML model
    # Note: Ensure these match exactly what's used in predict_signal
    ml_features = features[['velocity', 'acceleration', 'amplitude', 'phase', 'fracdiff']].dropna()
    
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
