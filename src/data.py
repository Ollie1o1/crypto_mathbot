import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
import asyncio

async def fetch_historical_data(symbol='BTC/USDT', timeframe='1h', limit=1000, end_date=None):
    """
    Fetch real historical OHLCV data from Binance with pagination.
    """
    print(f"Fetching {limit} candles of real data for {symbol}...")
    exchange = ccxt.binance()
    try:
        if end_date:
            end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)
        else:
            end_ts = exchange.milliseconds()
            
        # Calculate Start Time
        # Dynamic Timeframe Parsing
        tf_seconds = exchange.parse_timeframe(timeframe)
        total_duration_ms = limit * tf_seconds * 1000
        start_ts = end_ts - total_duration_ms
        
        all_ohlcv = []
        current_since = start_ts
        
        while len(all_ohlcv) < limit:
            fetch_limit = min(limit - len(all_ohlcv), 1000)
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=fetch_limit)
            
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 1 # Next candle timestamp
            
            print(f"Fetched {len(all_ohlcv)} / {limit} candles...", end='\r')
            
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Filter to ensure we don't go past end_date
        if end_date:
             end_dt = pd.to_datetime(end_date)
             df = df[df.index <= end_dt]
             
        return df
    finally:
        await exchange.close()

def generate_synthetic_data(hours=2000, end_date=None):
    """
    Generate synthetic OHLCV data for testing.
    """
    print(f"Generating {hours} hours of Synthetic Data (Random Walk)...")
    
    if end_date:
        end = pd.to_datetime(end_date)
    else:
        end = pd.Timestamp.now()
        
    dates = pd.date_range(end=end, periods=hours, freq='h')
    
    # Generate Trending Data (Sine Wave + Trend + Noise)
    # This ensures Hurst > 0.55 so the bot WILL trade
    t = np.linspace(0, 4*np.pi, hours)
    trend = np.linspace(0, 20, hours) # Strong uptrend
    cycle = 5 * np.sin(t)             # Clear cycles
    noise = np.random.randn(hours) * 0.5
    
    price = 100 + trend + cycle + noise
    df = pd.DataFrame({'close': price}, index=dates)
    return df
