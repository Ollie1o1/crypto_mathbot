import asyncio
import pandas as pd
import os
from src.features import CryptoKinematics
from src.strategy import SignalGenerator
from src.execution import ExecutionManager

# Configuration
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
EXCHANGE_ID = 'binance' # Example
API_KEY = os.getenv('API_KEY', '')
SECRET = os.getenv('SECRET', '')
SANDBOX = True

async def main():
    print("Starting Math-First Trading Bot...")
    
    # Initialize Components
    execution = ExecutionManager(EXCHANGE_ID, API_KEY, SECRET, sandbox=SANDBOX)
    kinematics = CryptoKinematics()
    strategy = SignalGenerator()
    
    # Initial Training Data Load
    print("Fetching historical data for training (1000 candles)...")
    history = await execution.exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=1000)
    df_history = pd.DataFrame(history, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_history['timestamp'] = pd.to_datetime(df_history['timestamp'], unit='ms')
    df_history.set_index('timestamp', inplace=True)
    
    print("Training initial model...")
    # Feature Engineering for Training
    smooth_price_train = kinematics.get_smooth_price(df_history['close'])
    kin_feats_train = kinematics.get_kinematics(smooth_price_train)
    dsp_feats_train = kinematics.get_dsp_features(df_history['close'])
    # regime_feats_train = kinematics.get_regime_features(df_history['close']) # Not needed for training input, only for filter
    fracdiff_train = kinematics.get_fracdiff(df_history['close'])
    
    features_train = strategy.prepare_features(kin_feats_train, dsp_feats_train, fracdiff_train)
    
    # Align price with features
    common_index = features_train.index.intersection(df_history.index)
    features_train = features_train.loc[common_index]
    price_train = df_history['close'].loc[common_index]
    
    strategy.train_model(features_train, price_train)
    print("Model trained successfully.")
    
    try:
        while True:
            print(f"Fetching latest data for {SYMBOL}...")
            ohlcv = await execution.exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=200)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Feature Engineering
            smooth_price = kinematics.get_smooth_price(df['close'])
            kin_feats = kinematics.get_kinematics(smooth_price)
            dsp_feats = kinematics.get_dsp_features(df['close'])
            regime_feats = kinematics.get_regime_features(df['close'])
            fracdiff = kinematics.get_fracdiff(df['close'])
            
            features = strategy.prepare_features(kin_feats, dsp_feats, fracdiff)
            
            if features.empty:
                print("Not enough data for features.")
                await asyncio.sleep(60)
                continue
                
            current_feat = features.iloc[[-1]]
            idx = current_feat.index[0]
            
            # Signal Generation
            prob = strategy.predict_signal(current_feat)
            acc = features.loc[idx, 'acceleration']
            hurst = regime_feats.loc[idx, 'hurst']
            entropy = regime_feats.loc[idx, 'entropy']
            
            regime_action = strategy.regime_filter(entropy, hurst)
            trigger = strategy.get_trigger(regime_action, hurst, acc, prob)
            
            print(f"Time: {idx}, Trigger: {trigger}, Prob: {prob:.2f}, Hurst: {hurst:.2f}, Entropy: {entropy:.2f}")
            
            # Execution
            if trigger == 'LONG':
                # Calculate size
                ticker = await execution.exchange.fetch_ticker(SYMBOL)
                price = ticker['ask']
                equity = 100 # Mock equity or fetch balance
                # balance = await execution.exchange.fetch_balance()
                # equity = balance['total']['USDT']
                
                # ATR calculation needed for sizing
                high = df['high']
                low = df['low']
                close = df['close']
                tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
                
                size = execution.calculate_position_size(equity, atr)
                size = await execution.check_dust(SYMBOL, size)
                
                if size > 0:
                    await execution.place_post_only_order(SYMBOL, 'buy', size)
                    
            elif trigger == 'SHORT':
                # Similar logic for short
                pass
            
            # Wait for next candle
            print("Analysis complete. Waiting 60 seconds for next cycle...")
            await asyncio.sleep(60) # Reduced to 60s for testing (was 1 hour)
            
    except KeyboardInterrupt:
        print("Stopping bot...")
    finally:
        await execution.close()

if __name__ == "__main__":
    asyncio.run(main())
