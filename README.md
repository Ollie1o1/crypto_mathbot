# Math-First Quantitative Trading Bot

A Market-Neutral / Directional Hybrid Trading System in Python for cryptocurrency markets, designed for capital-constrained accounts. This bot prioritizes fee minimization (Maker-Only) and ruin prevention using Calculus, Chaos Theory, and Signal Processing.

## Project Structure

```
cryptoLEV/
├── src/                # Core source code
│   ├── features.py     # Math & Physics feature engineering
│   ├── strategy.py     # Signal generation & ML logic
│   ├── execution.py    # Order management & risk control
│   └── data.py         # Data fetching & generation
├── results/            # Backtest reports, logs, and charts
├── logs/               # Application logs
├── main.py             # Live trading entry point
├── train_model.py      # Model training script
├── validate_strategy.py # Backtesting & Validation script
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

## Features

- **Kinematics**: Velocity & Acceleration on **Log-Returns** (Scale Invariant).
- **Signal Processing**: Hilbert Transform for Phase/Amplitude on Detrended Series.
- **Regime Detection**: Hurst Exponent and Shannon Entropy (Fed to ML).
- **Machine Learning**: XGBoost Classifier with **Dynamic Volatility Barriers**.
- **Execution**: Continuous **Kelly Criterion** with Volatility Targeting.
- **Performance**: **GPU Acceleration** and **Vectorized Backtesting** (1000x Speedup).
- **Optimization**: Automated Hyperparameter Tuning with **Optuna**.

## Project Structure

- `src/`: Core logic (Features, Strategy, Execution).
- `strategies/`: Strategy configurations and definitions.
- `utils/`: Utility functions.
- `models/`: Saved ML models.

## Requirements

- Python 3.10+
- See `requirements.txt` for Python packages.

## Installation

1.  Clone the repository.
2.  Create and activate a virtual environment (Recommended):
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    
    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Configuration
Configure your API keys in a `.env` file (optional for backtesting, required for live trading).

### 2. Workflow: Train -> Save -> Validate (Recommended)

To avoid re-training on every run and to ensure consistent results, use the **Train-Save-Load** workflow.

**Step 1: Train the Model**
Fetch historical data, train the XGBoost model, and save it.
```bash
# Train on 5000 hours of real data
# Train on 5000 hours of real data
python train_model.py --symbol "BTC/USDT" --limit 5000 --save_path "models/xgb_prod.json"
```

**Step 2: Validate the Model**
Load the saved model and test it on a different (or same) dataset.
```bash
# Validate on the last 2000 hours using the saved model
python validate_strategy.py --real --limit 2000 --model_path "models/xgb_prod.json"
```

### 3. Hyperparameter Tuning (Optional)
Optimize the XGBoost model parameters using Optuna.
```bash
python optimize.py --limit 2000 --trials 50
```

### 4. Quick Backtest (Walk-Forward)
Run the validation script directly to perform a Walk-Forward Optimization (Train/Test rolling window) on the fly.

```bash
# Default (2000 hours ~ 83 days)
python validate_strategy.py

# Test on 5000 hours (~208 days)
python validate_strategy.py --limit 5000
```

**Output:**
- **Console Report**: Shows Data Source, Timeframe, Duration, Leverage, and Account Summary.
- **Trade Log**: Saves a CSV of all trades to `results/trade_log.csv`.
- **Visualization**: Saves a plot of the equity curve to `results/backtest_equity.png`.

**Understanding Results:**
- **Synthetic Data (Default)**: Shows high returns (e.g., >900%) because it uses a **Random Walk with Positive Drift**. This is "Too Good To Be True" by design to verify the math works on trending data. **Do not expect these returns on real markets.**
- **Real Data (`--real`)**: Uses actual Binance BTC/USDT history.
    - **0 Trades?** This is normal on short timeframes. The bot is designed to be **Capital-Constrained & Risk-Averse**. It only trades when:
        1.  **Trend is Strong** (Hurst > 0.55)
        2.  **Momentum is Accelerating**
        3.  **AI Confidence is High** (> 65%)
    - If conditions aren't perfect, it stays in **CASH** to prevent ruin.

## Performance & Validation

**Important: Anti-Cheating Protocol**
We have implemented a **Strict Causal Backtester** in `validate_strategy.py`.
- Previous versions (and most public bots) use vectorized feature generation which can accidentally "leak" future data (e.g., Hilbert Transform on the whole dataset).
- This bot now recalculates features **step-by-step** using a sliding window, exactly mimicking live trading.
- **Result**: Backtests are slower but 100% realistic. No more "4,000,000% returns" artifacts.

### 4. Live Trading
Run the main bot to start the execution loop:
```bash
python main.py
```
- The bot will **automatically load** `models/optimized_model.json` if it exists.
- **Note**: Ensure you retrain your model using the updated `train_model.py` to ensure the model learns from strictly causal features.
- If no model is found, it will train a fresh one.
