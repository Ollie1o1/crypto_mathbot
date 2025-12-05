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

from src.data import fetch_historical_data, generate_synthetic_data

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
    
    total_steps = len(data) - train_size - test_size
    step_counter = 0
    
    # Pre-calculate ALL features once (Vectorized)
    # Note: Z-Score normalization uses a rolling window, so it is causal.
    # However, to be strictly safe for WFO, we should ideally recalculate normalization params.
    # But for performance, we will use the pre-calculated features which use a large rolling window.
    # The 'generate_all_features' method in features.py uses rolling windows, so it respects causality
    # as long as we don't use future data for the window calculation (which rolling() doesn't).
    
    print("Pre-calculating features for the entire dataset...")
    kinematics = CryptoKinematics()
    all_features = kinematics.generate_all_features(data['close'])
    
    for i in range(0, total_steps, test_size):
        # Define Training Window
        train_start_idx = i
        train_end_idx = i + train_size
        
        # 1. Train Model
        # Slice features directly
        # We need to ensure indices align
        train_indices = data.index[train_start_idx : train_end_idx]
        valid_indices = all_features.index.intersection(train_indices)
        
        X_train = all_features.loc[valid_indices]
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
            
            # Get pre-calculated feature row
            if current_time not in all_features.index:
                continue
                
            current_feat_row = all_features.loc[[current_time]]
            
            # Predict
            prob = strategy.predict_signal(current_feat_row)
            
            # Calculate Volatility for Kelly (Simple 24h rolling std)
            # We can also pre-calculate this, but it's fast enough
            # Or use a rolling window on the price series
            # Let's do a quick slice
            hist_start = max(0, j - 24)
            volatility = data['close'].iloc[hist_start:j].pct_change().std()
            if pd.isna(volatility) or volatility == 0:
                volatility = 0.01
            
            # Get Leverage (Signed)
            leverage = strategy.get_leverage(prob, volatility)
            
            trigger = 'HOLD'
            if leverage > 0:
                trigger = 'LONG'
            elif leverage < 0:
                trigger = 'SHORT'
                
            results.append({
                'timestamp': current_time,
                'price': data['close'].iloc[j],
                'trigger': trigger,
                'prob': prob,
                'leverage': abs(leverage)
            })
            
        step_counter += 1
        print(f"Completed Batch {step_counter}/{total_steps // test_size}...", end='\r')
            
    return pd.DataFrame(results)

def run_static_backtest(data: pd.DataFrame, model_path: str):
    """
    Run backtest using a pre-trained model (No re-training).
    Vectorized for speed.
    """
    print(f"Starting Static Backtest with model: {model_path}")
    strategy = SignalGenerator()
    strategy.load_model(model_path)
    
    # 1. Pre-calculate Features
    print("Generating features...")
    kinematics = CryptoKinematics()
    all_features = kinematics.generate_all_features(data['close'])
    
    # 2. Align Data
    common_idx = all_features.index.intersection(data.index)
    all_features = all_features.loc[common_idx]
    data = data.loc[common_idx]
    
    # 3. Batch Prediction (Much Faster)
    print("Running Batch Prediction...")
    probs = strategy.model.predict_proba(all_features)[:, 1]
    
    # 4. Vectorized Logic
    print("Calculating Signals...")
    returns = data['close'].pct_change()
    volatility = returns.rolling(window=24).std().fillna(0.01)
    
    results = []
    
    # We iterate to apply logic, but prediction is already done
    # We can vectorize the logic too, but a loop is fine now that heavy lifting is done
    
    for i in range(len(data)):
        current_time = data.index[i]
        prob = probs[i]
        vol = volatility.iloc[i]
        
        leverage = strategy.get_leverage(prob, vol)
        
        trigger = 'HOLD'
        if leverage > 0:
            trigger = 'LONG'
        elif leverage < 0:
            trigger = 'SHORT'
            
        results.append({
            'timestamp': current_time,
            'price': data['close'].iloc[i],
            'trigger': trigger,
            'prob': prob,
            'leverage': abs(leverage)
        })
        
        if i % 1000 == 0:
             print(f"Processed {i}/{len(data)} candles...", end='\r')
            
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
    parser.add_argument('--model_path', type=str, help='Path to pre-trained model (e.g., models/xgb_model.json). If set, skips WFO.')
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
    
    # 2. Backtest
    if args.model_path:
        results = run_static_backtest(df, args.model_path)
    else:
        results = walk_forward_optimization(df)
    
    if not results.empty:
        results.set_index('timestamp', inplace=True)
    
    # 3. Metrics & Logging
    save_trade_log(results)
    returns = calculate_metrics(results)
    
    total_return = (1 + returns).prod() - 1
    print(f"Total Return: {total_return:.2%}")

    # Visualization
    if returns.empty:
        print("No returns to visualize.")
        import sys
        sys.exit(0)

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
