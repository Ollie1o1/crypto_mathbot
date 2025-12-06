import pandas as pd
import numpy as np
import argparse
import asyncio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from src.features import CryptoKinematics
from src.strategy import SignalGenerator
from src.data import fetch_historical_data

FEE_RATE = 0.001 # 0.1% fee per side (Binance Taker)

def strict_backtest(data: pd.DataFrame, model_path: str = None, lookback_window: int = 500):
    """
    Strict Causal Backtest with Fees.
    Logic:
    1. Iterate time T.
    2. Calc features on data[T-Window : T].
    3. Predict T+1.
    4. Execute with fees.
    """
    print("Starting STRICT CAUSAL Backtest...")
    print(f"Transaction Fee: {FEE_RATE*100:.2f}% per trade")
    
    strategy = SignalGenerator()
    kinematics = CryptoKinematics()
    
    # Train Initial Model if not provided
    if not model_path:
        # Split first 30% for training
        train_size = int(len(data) * 0.3)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        print(f"Training initial model on first {train_size} candles...")
        features_train = kinematics.generate_all_features(train_data['close'], window_size=300)
        
        # Align
        common = features_train.index.intersection(train_data.index)
        strategy.train_model(features_train.loc[common], train_data['close'].loc[common])
    else:
        strategy.load_model(model_path)
        test_data = data # Run on all if model provided (e.g. forward test)
        # But we need warmup period
        # Let's just run from index 500
        pass

    results = []
    
    # We need a rolling window to generate features dynamically
    # Testing Start Index
    start_idx = lookback_window
    
    # If using test_data split
    if not model_path:
        # We perform Walk-Forward? Or just static holdout?
        # Let's do Static Holdout for speed in this prompt, but with strict loop
        full_df = data
        start_idx = int(len(data) * 0.3)
    else:
        full_df = data
    
    total_steps = len(full_df) - start_idx
    
    position = 0 # 0, 1, -1
    entry_price = 0.0
    
    # Equity Curve Tracking
    equity = 100.0
    equity_curve = [equity]
    timestamps = [full_df.index[start_idx-1]]
    
    print(f"Running simulation on {total_steps} candles...")
    
    for i in range(total_steps):
        current_idx = start_idx + i
        
        # 1. Causal Slice
        # We need enough history for rolling windows (e.g. 200)
        # Slicing is somewhat slow, but essential for causality proof
        hist_start = max(0, current_idx - lookback_window)
        current_slice = full_df.iloc[hist_start : current_idx + 1] # Includes current candle CLOSE
        
        current_time = current_slice.index[-1]
        current_price = current_slice['close'].iloc[-1]
        
        # 2. Generate Features (Latest row only)
        # We generate the whole window then take the last row
        features = kinematics.generate_all_features(current_slice['close'], window_size=300)
        
        if features.empty:
            continue
            
        current_feat = features.iloc[[-1]] # DataFrame 1 row
        
        # 3. Predict & Sizing
        prob = strategy.predict_signal(current_feat)
        
        # Calc Volatility for Sizing
        # Using feature volatility or calc fresh
        vol = current_feat['volatility_short'].iloc[0] if 'volatility_short' in current_feat else 0.01
        
        target_action = strategy.get_action(prob, threshold=0.55)
        leverage = strategy.get_leverage(prob, vol)
        
        # 4. Execution Engine (with Fees)
        # Simple Reversal Logic
        
        new_position = 0
        if target_action == 'LONG': new_position = 1
        elif target_action == 'SHORT': new_position = -1
        
        # Apply Leverage Size
        new_position = new_position * leverage
        
        # Check Change
        # We iterate to apply PnL
        
        # PnL from PREVIOUS Step
        if i > 0:
            # Price Change
            prev_price = full_df['close'].iloc[current_idx-1]
            price_change = (current_price - prev_price) / prev_price
            
            # Position PnL
            step_pnl = position * price_change
            equity = equity * (1 + step_pnl)
            
        # Trade Execution Cost
        # If position changed significantly
        if abs(new_position - position) > 0.01:
            # We traded
            # Fee logic: Fee paid on NOTIONAL value traded
            # Notional Traded = abs(new_pos - old_pos) * Equity
            # Fee = Notional * Rate
            
            trade_size = abs(new_position - position)
            fee_cost = trade_size * FEE_RATE
            
            # Reduce Equity
            equity = equity * (1 - fee_cost)
            
            results.append({
                'timestamp': current_time,
                'action': 'TRADE',
                'size': trade_size,
                'cost': fee_cost,
                'equity': equity
            })
            
        # Update State
        position = new_position
        
        equity_curve.append(equity)
        timestamps.append(current_time)
        
        if i % 500 == 0:
            print(f"Simulated {i}/{total_steps} | Eq: {equity:.2f} | Pos: {position:.2f}", end='\r')
            
    # Metrics
    equity_series = pd.Series(equity_curve, index=timestamps)
    returns = equity_series.pct_change().dropna()
    
    total_ret = (equity - 100) / 100
    sharpe = returns.mean() / returns.std() * np.sqrt(24*365) if returns.std() > 0 else 0
    max_drawdown = (equity_series / equity_series.cummax() - 1).min()
    
    print("\n\n--- Strict Backtest Results ---")
    print(f"Total Return: {total_ret:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Final Equity: ${equity:.2f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(equity_series)
    plt.title(f"Strict Causal Backtest (Fees={FEE_RATE*100}%)")
    plt.ylabel("Equity ($)")
    plt.yscale('log')
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig('results/strict_backtest.png')
    print("Plot saved to results/strict_backtest.png")
    
    return equity_series

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=3000)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--timeframe', type=str, default='1h')
    parser.add_argument('--symbol', type=str, default='BTC/USDT')
    args = parser.parse_args()
    
    df = asyncio.run(fetch_historical_data(symbol=args.symbol, timeframe=args.timeframe, limit=args.limit))
    strict_backtest(df, args.model_path)
