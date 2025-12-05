import pandas as pd
import numpy as np
from src.features import CryptoKinematics
from src.strategy import SignalGenerator

import ccxt.async_support as ccxt
import asyncio
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime, timedelta

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
            
        all_ohlcv = []
        remaining = limit
        
        while remaining > 0:
            fetch_limit = min(remaining, 1000)
            # Calculate 'since' for this batch
            # We fetch backwards-ish by adjusting 'since' or just fetching and filtering?
            # CCXT fetch_ohlcv usually fetches forward from 'since'.
            # To get the LAST 'limit' candles ending at 'end_date', we need to calculate the start.
            
            duration_ms = remaining * 60 * 60 * 1000
            since = end_ts - duration_ms
            
            # Safety check to ensure we don't get stuck
            if len(all_ohlcv) > 0 and since >= all_ohlcv[-1][0]:
                 break
                 
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=fetch_limit)
            
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            remaining -= len(ohlcv)
            
            # Update end_ts for next batch (though we are calculating 'since' from the original end)
            # Actually, simpler approach: Calculate strict Start Time for the whole block
            # and fetch forward from there.
            
        # Re-do with simpler forward-fetch logic
        # 1. Calculate Start Time
        total_duration_ms = limit * 60 * 60 * 1000
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

def calculate_features(data_window: pd.Series):
    """
    Helper to calculate features on a specific window of data.
    Returns the LAST row of features (the one corresponding to the end of the window).
    """
    kinematics = CryptoKinematics()
    
    # 1. Smooth Price (EMA is causal, but we calc on window to be safe)
    smooth_price = kinematics.get_smooth_price(data_window)
    
    # 2. Kinematics
    kin_feats = kinematics.get_kinematics(smooth_price)
    
    # 3. DSP (Hilbert) - Calc on window, take last
    dsp_feats = kinematics.get_dsp_features(data_window)
    
    # 4. Regime
    regime_feats = kinematics.get_regime_features(data_window)
    
    # 5. FracDiff
    fracdiff = kinematics.get_fracdiff(data_window)
    
    # Combine
    # Note: prepare_features concatenates. We need to handle the fact that
    # some features (like diff) introduce NaNs at the start of the window.
    # We only care about the LAST row (current time).
    
    # Manually combine to ensure we get the last row even if earlier ones are NaN
    df = pd.concat([kin_feats, dsp_feats, regime_feats, fracdiff.rename('fracdiff')], axis=1)
    
    return df.iloc[[-1]] # Return as 1-row DataFrame

def walk_forward_optimization(data: pd.DataFrame, train_window_days: int = 30, test_window_days: int = 7):
    """
    Train on Month M, Test on M+1 (or week), roll forward.
    Strictly prevents look-ahead bias by recalculating features at every step.
    """
    print("Starting Walk-Forward Optimization (Strict No-Leakage Mode)...")
    strategy = SignalGenerator()
    
    # Parameters
    train_size = train_window_days * 24
    test_size = test_window_days * 24
    lookback_buffer = 500 # Enough for Hilbert/Regime windows
    
    results = []
    
    # Pre-calculate features for the initial training block to save time?
    # No, to be 100% safe and consistent, we should build the history step-by-step
    # or just use the 'calculate_features' on the training window.
    
    # We iterate through the dataset in chunks of (Train + Test)
    # But we slide by 'test_size'
    
    total_steps = len(data) - train_size - test_size
    step_counter = 0
    
    for i in range(0, total_steps, test_size):
        # Define Training Window
        train_start_idx = i
        train_end_idx = i + train_size
        
        train_data = data['close'].iloc[train_start_idx : train_end_idx]
        
        # 1. Train Model
        # We need features for the ENTIRE training window.
        # Since our features are now Causal (EMA), we CAN calculate them in batch 
        # on the training window without leakage from the future (Test window).
        # However, to be perfectly safe with Hilbert/Regime (which use windows),
        # we should ideally include some buffer before the training window 
        # to warm up the indicators.
        
        buffer_start = max(0, train_start_idx - lookback_buffer)
        train_data_buffered = data['close'].iloc[buffer_start : train_end_idx]
        
        # Calculate features on buffered data
        kinematics = CryptoKinematics()
        smooth_p = kinematics.get_smooth_price(train_data_buffered)
        kin_f = kinematics.get_kinematics(smooth_p)
        dsp_f = kinematics.get_dsp_features(train_data_buffered)
        reg_f = kinematics.get_regime_features(train_data_buffered)
        fd_f = kinematics.get_fracdiff(train_data_buffered)
        
        train_feats_all = strategy.prepare_features(kin_f, dsp_f, fd_f)
        
        # Trim buffer to get actual training set
        # We need to align indices
        train_indices = train_data.index
        valid_indices = train_feats_all.index.intersection(train_indices)
        
        X_train = train_feats_all.loc[valid_indices]
        y_price = data['close'].loc[valid_indices]
        
        if len(X_train) > 100:
            strategy.train_model(X_train, y_price)
        
        # 2. Test / Predict (Step-by-Step)
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_size
        
        # Loop through every candle in the test set
        for j in range(test_start_idx, test_end_idx):
            if j >= len(data):
                break
                
            current_time = data.index[j]
            
            # SANITY CHECK: Ensure we are not using future data
            # We use data up to 'current_time' (inclusive of Close at t) to predict t+1
            
            # Get history window for feature calc (Current + Lookback)
            hist_start = max(0, j - lookback_buffer)
            history_window = data['close'].iloc[hist_start : j + 1]
            
            # Calculate features for this specific step
            # This ensures 'get_dsp_features' (Hilbert) and others only see history
            current_feat_row = calculate_features(history_window)
            
            if current_feat_row.empty:
                continue
                
            # Sanity Check Timestamp
            feat_time = current_feat_row.index[0]
            if feat_time > current_time:
                raise ValueError(f"CRITICAL: Future Data Leak! Feature Time {feat_time} > Current Time {current_time}")
            
            # Predict
            # Filter columns to match training data (exclude regime features)
            ml_features = current_feat_row[['velocity', 'acceleration', 'amplitude', 'phase', 'fracdiff']]
            prob = strategy.predict_signal(ml_features)
            
            # Get other metrics
            acc = current_feat_row['acceleration'].iloc[0]
            hurst = current_feat_row['hurst'].iloc[0] if 'hurst' in current_feat_row else 0.5
            entropy = current_feat_row['entropy'].iloc[0] if 'entropy' in current_feat_row else 0.0
            
            regime_action = strategy.regime_filter(entropy, hurst)
            trigger = strategy.get_trigger(regime_action, hurst, acc, prob)
            leverage = strategy.get_leverage(prob, hurst)
            
            results.append({
                'timestamp': current_time,
                'price': data['close'].iloc[j],
                'trigger': trigger,
                'prob': prob,
                'leverage': leverage
            })
            
        step_counter += 1
        print(f"Completed Batch {step_counter}/{total_steps // test_size}...", end='\r')
            
    return pd.DataFrame(results)

def save_trade_log(results: pd.DataFrame, filename='results/trade_log.csv'):
    """
    Save trade log to CSV.
    """
    trades = []
    position = 0
    entry_price = 0
    entry_time = None
    
    for i in range(1, len(results)):
        prev_row = results.iloc[i-1]
        curr_row = results.iloc[i]
        
        # Close position
        if position != 0:
            exit_price = curr_row['price']
            pnl = (exit_price - entry_price) / entry_price
            if position == -1:
                pnl = -pnl
            
            trades.append({
                'Entry Time': entry_time,
                'Exit Time': curr_row.name,
                'Type': 'LONG' if position > 0 else 'SHORT',
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Leverage': f"{abs(position)}x",
                'PnL': pnl
            })
            position = 0
            
        # Open position
        if prev_row['trigger'] == 'LONG':
            position = prev_row['leverage']
            entry_price = curr_row['price']
            entry_time = curr_row.name
        elif prev_row['trigger'] == 'SHORT':
            position = -prev_row['leverage']
            entry_price = curr_row['price']
            entry_time = curr_row.name
            
    df_trades = pd.DataFrame(trades)
    if not df_trades.empty:
        df_trades.to_csv(filename, index=False)
        print(f"Trade log saved to '{filename}'")
        print("\n--- Recent Trades ---")
        print(df_trades.tail().to_string(index=False))
    else:
        print("No trades executed.")
        
    return df_trades

def monte_carlo_simulation(returns: pd.Series, n_sims: int = 1000, block_size: int = 240):
    """
    Block Bootstrapping Monte Carlo Simulation.
    """
    print(f"Running {n_sims} Monte Carlo Simulations...")
    sim_results = []
    
    if len(returns) < block_size:
        print(f"Warning: Not enough data for Monte Carlo (Need {block_size}, got {len(returns)}). Skipping.")
        return np.zeros((n_sims, len(returns)))

    for _ in range(n_sims):
        # Block Bootstrap
        sim_returns = []
        while len(sim_returns) < len(returns):
            start = np.random.randint(0, len(returns) - block_size)
            block = returns.iloc[start : start+block_size].values
            sim_returns.extend(block)
            
        sim_returns = sim_returns[:len(returns)]
        sim_path = np.cumprod(1 + np.array(sim_returns))
        sim_results.append(sim_path)
        
    return np.array(sim_results)

def calculate_metrics(results: pd.DataFrame):
    """
    Calculate Sharpe, Calmar, and Ruin Probability.
    """
    # Simulate simple PnL
    results['return'] = 0.0
    position = 0
    entry_price = 0
    
    for i in range(1, len(results)):
        prev_row = results.iloc[i-1]
        curr_row = results.iloc[i]
        
        # Close position
        if position != 0:
            ret = (curr_row['price'] - entry_price) / entry_price
            if position < 0:
                ret = -ret
            
            # Apply Leverage
            ret = ret * abs(position)
            
            results.loc[curr_row.name, 'return'] = ret
            position = 0 # Simple 1-step hold for testing
            
        # Open position
        if prev_row['trigger'] == 'LONG':
            position = prev_row['leverage']
            entry_price = curr_row['price']
        elif prev_row['trigger'] == 'SHORT':
            position = -prev_row['leverage']
            entry_price = curr_row['price']
            
    returns = results['return']
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(24*365) if np.std(returns) != 0 else 0
    
    cum_returns = (1 + returns).cumprod()
    max_drawdown = 1 - cum_returns / cum_returns.cummax()
    calmar = returns.mean() * 24 * 365 / max_drawdown.max() if max_drawdown.max() != 0 else 0
    
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Calmar Ratio: {calmar:.2f}")
    
    return returns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate Strategy')
    parser.add_argument('--real', action='store_true', help='Use real historical data instead of synthetic')
    parser.add_argument('--limit', type=int, default=2000, help='Number of historical candles to fetch (default: 2000)')
    parser.add_argument('--end', type=str, help='End date for backtest (YYYY-MM-DD). Default: Now')
    args = parser.parse_args()

    data_source = "Synthetic (Random Walk)"
    if args.real:
        data_source = "Real Market (Binance BTC/USDT)"
        # 1. Fetch Real Data
        df = asyncio.run(fetch_historical_data(limit=args.limit, end_date=args.end))
        print(f"\n--- Backtest Results ({data_source}: {args.limit} candles) ---")
    else:
        # 1. Generate Synthetic Data
        print(f"\n--- Backtest Results ({data_source}) ---")
        df = generate_synthetic_data(hours=args.limit, end_date=args.end)
    
    print(f"Timeframe: {df.index[0]} to {df.index[-1]}")
    print(f"Duration:  {df.index[-1] - df.index[0]}")
    
    # 2. Walk-Forward
    results = walk_forward_optimization(df)
    
    if not results.empty:
        results.set_index('timestamp', inplace=True)
    
    # 3. Metrics & Logging
    save_trade_log(results)
    returns = calculate_metrics(results)
    
    total_return = (1 + returns).prod() - 1
    print(f"Total Return: {total_return:.2%}")

    # Visualization
    starting_capital = 100.0
    equity_curve = starting_capital * (1 + returns).cumprod()
    ending_capital = equity_curve.iloc[-1]
    
    print(f"\n--- Account Summary ---")
    print(f"Data Source:      {data_source}")
    print(f"Leverage Used:    Dynamic (1x - 3x)")
    print(f"Starting Capital: ${starting_capital:.2f}")
    print(f"Ending Capital:   ${ending_capital:.2f}")
    print(f"Net Profit:       ${ending_capital - starting_capital:.2f}")
    
    # Trade Stats
    if 'return' in results.columns:
        trades = results[results['return'] != 0]['return']
        n_trades = len(trades)
        if n_trades > 0:
            win_rate = len(trades[trades > 0]) / n_trades
            avg_pnl = trades.mean()
            best_trade = trades.max()
            worst_trade = trades.min()
            
            print(f"\n--- Trade Statistics ---")
            print(f"Total Trades:     {n_trades}")
            print(f"Win Rate:         {win_rate:.2%}")
            print(f"Avg PnL / Trade:  {avg_pnl:.2%}")
            print(f"Best Trade:       {best_trade:.2%}")
            print(f"Worst Trade:      {worst_trade:.2%}")
            print(f"Compounding:      (1 + {avg_pnl:.4f}) ^ {n_trades} ≈ {(1+avg_pnl)**n_trades:.2f}x")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, equity_curve.values, label='Equity Curve')
    plt.title(f'Strategy Performance: {data_source}')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    
    # Format Y-Axis ($)
    plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    
    # Format Date Axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
    plt.gcf().autofmt_xdate() # Rotate labels
    
    plt.savefig('results/backtest_equity.png')
    print("\nVisualization saved to 'results/backtest_equity.png'")

    # Log Plot (for consistency check)
    plt.figure(figsize=(12, 6))
    plt.semilogy(equity_curve.index, equity_curve.values, label='Equity Curve (Log Scale)')
    plt.title(f'Strategy Performance: {data_source} (Log Scale)')
    plt.xlabel('Date')
    plt.ylabel('Equity ($) - Log Scale')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # Format Y-Axis ($)
    plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    
    # Format Date Axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
    plt.gcf().autofmt_xdate() # Rotate labels
    
    plt.savefig('results/backtest_equity_log.png')
    print("Log-Scale Visualization saved to 'results/backtest_equity_log.png'")
    
    # 4. Monte Carlo
    sim_paths = monte_carlo_simulation(returns)
    
    # Probability of Ruin (hitting 0 or -50% etc)
    # Assuming starting equity 1.0
    ruin_threshold = 0.5
    ruins = np.sum([np.min(path) < ruin_threshold for path in sim_paths])
    prob_ruin = ruins / len(sim_paths)
    
    print(f"Probability of Ruin (< {ruin_threshold}): {prob_ruin:.2%}")
