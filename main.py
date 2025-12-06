import asyncio
import pandas as pd
import os
from src.features import CryptoKinematics
from src.strategy import SignalGenerator
from src.execution import ExecutionManager

# Configuration
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
EXCHANGE_ID = 'binance' 
API_KEY = os.getenv('API_KEY', '')
SECRET = os.getenv('SECRET', '')
SANDBOX = True

async def main():
    print("Starting Math-First Trading Bot (Strict Causal Mode)...")
    
    # Initialize Components
    execution = ExecutionManager(EXCHANGE_ID, API_KEY, SECRET, sandbox=SANDBOX)
    kinematics = CryptoKinematics()
    strategy = SignalGenerator()
    
    # Load Model (or train if missing)
    model_path = 'models/optimized_model.json'
    if os.path.exists(model_path):
        strategy.load_model(model_path)
    else:
        print("No model found. Training on historical data...")
        # Fetch enough data for training
        history = await execution.exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=3000)
        df_hist = pd.DataFrame(history, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'], unit='ms')
        df_hist.set_index('timestamp', inplace=True)
        
        # Train
        features = kinematics.generate_all_features(df_hist['close'], window_size=300)
        
        # Align
        common = features.index.intersection(df_hist.index)
        strategy.train_model(features.loc[common], df_hist['close'].loc[common])
        strategy.save_model(model_path)

    try:
        while True:
            print(f"Fetching latest data for {SYMBOL}...")
            # Need enough context for rolling windows (e.g. 500)
            ohlcv = await execution.exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=1000)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Feature Generation
            # This generates features for the whole window, but we only use the last one for prediction
            features = kinematics.generate_all_features(df['close'], window_size=300)
            
            if features.empty:
                print("Not enough data for features.")
                await asyncio.sleep(60)
                continue
                
            current_feat = features.iloc[[-1]]
            idx = current_feat.index[0]
            
            # Inference
            prob = strategy.predict_signal(current_feat)
            
            # Volatility for Sizing
            vol = current_feat['volatility_short'].iloc[0] if 'volatility_short' in current_feat else 0.01
            
            action = strategy.get_action(prob)
            leverage = strategy.get_leverage(prob, vol)
            
            print(f"Time: {idx}, Action: {action}, LVG: {leverage:.2f}x, Prob: {prob:.2f}")
            
            # Execution logic would go here
            # For now just logging
            
            # Wait for next candle
            print("Analysis complete. Waiting 60 seconds...")
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        print("Stopping bot...")
    finally:
        await execution.close()

if __name__ == "__main__":
    asyncio.run(main())
