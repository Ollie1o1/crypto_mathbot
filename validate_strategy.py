import pandas as pd
import numpy as np
from src.features import CryptoKinematics
from src.strategy import SignalGenerator

import ccxt.async_support as ccxt
import asyncio
import argparse
import matplotlib.pyplot as plt

def generate_synthetic_data(hours=2000):
    """
    Generate synthetic OHLCV data for testing.
    """
    print(f"Generating {hours} hours of Synthetic Data (Random Walk)...")
    dates = pd.date_range(start='2024-01-01', periods=hours, freq='h')
    # Add a slight positive drift to make it "too good" like the user saw
    drift = 0.0001
    returns = np.random.randn(len(dates)) * 0.01 + drift
    price = 100 * np.cumprod(1 + returns)
    df = pd.DataFrame({'close': price}, index=dates)
    return df

async def fetch_historical_data(symbol='BTC/USDT', timeframe='1h', limit=1000):
    """
    Fetch real historical OHLCV data from Binance.
    """
    print(f"Fetching {limit} candles of real data for {symbol}...")
    exchange = ccxt.binance()
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    finally:
        await exchange.close()

def walk_forward_optimization(data: pd.DataFrame, train_window_days: int = 30, test_window_days: int = 7):
    """
    Train on Month M, Test on M+1 (or week), roll forward.
    """
    print("Starting Walk-Forward Optimization...")
    kinematics = CryptoKinematics()
    strategy = SignalGenerator()
    
    # Feature Engineering
    print("Generating features...")
    smooth_price = kinematics.get_smooth_price(data['close'])
    kin_feats = kinematics.get_kinematics(smooth_price)
    dsp_feats = kinematics.get_dsp_features(data['close'])
    regime_feats = kinematics.get_regime_features(data['close'])
    fracdiff = kinematics.get_fracdiff(data['close'])
    
    features = strategy.prepare_features(kin_feats, dsp_feats, fracdiff)
    # Align data
    common_index = features.index.intersection(data.index)
    features = features.loc[common_index]
    price = data['close'].loc[common_index]
    
    # Rolling Window
    train_size = train_window_days * 24
    test_size = test_window_days * 24
    
    results = []
    
    for i in range(0, len(features) - train_size - test_size, test_size):
        train_features = features.iloc[i : i+train_size]
        train_price = price.iloc[i : i+train_size]
        
        test_features = features.iloc[i+train_size : i+train_size+test_size]
        test_price = price.iloc[i+train_size : i+train_size+test_size]
        
        # Train
        strategy.train_model(train_features, train_price)
        
        # Test / Predict
        for j in range(len(test_features)):
            current_feat = test_features.iloc[[j]]
            prob = strategy.predict_signal(current_feat)
            
            # Get other metrics for trigger
            idx = test_features.index[j]
            acc = features.loc[idx, 'acceleration']
            hurst = regime_feats.loc[idx, 'hurst']
            entropy = regime_feats.loc[idx, 'entropy']
            
            regime_action = strategy.regime_filter(entropy, hurst)
            trigger = strategy.get_trigger(regime_action, hurst, acc, prob)
            
            results.append({
                'timestamp': idx,
                'price': test_price.iloc[j],
                'trigger': trigger,
                'prob': prob
            })
            
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
                'Exit Time': curr_row['timestamp'],
                'Type': 'LONG' if position == 1 else 'SHORT',
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Leverage': '1x',
                'PnL': pnl
            })
            position = 0
            
        # Open position
        if prev_row['trigger'] == 'LONG':
            position = 1
            entry_price = curr_row['price']
            entry_time = curr_row['timestamp']
        elif prev_row['trigger'] == 'SHORT':
            position = -1
            entry_price = curr_row['price']
            entry_time = curr_row['timestamp']
            
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
            if position == -1:
                ret = -ret
            results.loc[curr_row.name, 'return'] = ret
            position = 0 # Simple 1-step hold for testing
            
        # Open position
        if prev_row['trigger'] == 'LONG':
            position = 1
            entry_price = curr_row['price']
        elif prev_row['trigger'] == 'SHORT':
            position = -1
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
    args = parser.parse_args()

    data_source = "Synthetic (Random Walk)"
    if args.real:
        data_source = "Real Market (Binance BTC/USDT)"
        # 1. Fetch Real Data
        df = asyncio.run(fetch_historical_data(limit=args.limit))
        print(f"\n--- Backtest Results ({data_source}: {args.limit} candles) ---")
    else:
        # 1. Generate Synthetic Data
        print(f"\n--- Backtest Results ({data_source}) ---")
        df = generate_synthetic_data(hours=args.limit)
    
    print(f"Timeframe: {df.index[0]} to {df.index[-1]}")
    print(f"Duration:  {df.index[-1] - df.index[0]}")
    
    # 2. Walk-Forward
    results = walk_forward_optimization(df)
    
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
    print(f"Leverage Used:    1x (Simulated)")
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
    plt.savefig('results/backtest_equity.png')
    print("\nVisualization saved to 'results/backtest_equity.png'")
    
    # 4. Monte Carlo
    sim_paths = monte_carlo_simulation(returns)
    
    # Probability of Ruin (hitting 0 or -50% etc)
    # Assuming starting equity 1.0
    ruin_threshold = 0.5
    ruins = np.sum([np.min(path) < ruin_threshold for path in sim_paths])
    prob_ruin = ruins / len(sim_paths)
    
    print(f"Probability of Ruin (< {ruin_threshold}): {prob_ruin:.2%}")
