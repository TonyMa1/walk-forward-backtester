# Walk Forward Optimization (WFO) Backtester

A Python implementation of Walk Forward Optimization (WFO) for trading strategy backtesting with robust position management and dynamic parameter optimization.

## Overview

This project implements Walk Forward Optimization (WFO) for backtesting and optimizing trading strategies. WFO is a technique that uses rolling windows of data to optimize strategy parameters on in-sample data and validate them on out-of-sample data, reducing the risk of overfitting.

### Key Features

- Walk Forward Optimization with configurable rolling windows
- Grid search for parameter optimization
- Multiple trading strategy implementations:
  - Moving Average Crossover with position tracking
  - ATR-Based Dynamic Stop Loss with trailing stops
- Performance metrics including Sharpe ratio, returns, and win rate
- Parameter stability analysis
- Comprehensive type hints and documentation

## Project Structure

```
walk_forward_opt/
├── src/
│   └── walk_forward_opt/
│       ├── strategies/      # Trading strategy implementations
│       ├── backtesting/     # WFO and backtesting logic
│       ├── optimization/    # Parameter optimization algorithms
│       └── utils/           # Utility functions and metrics
└── tests/
    └── walk_forward_opt/
        ├── test_strategies/
        ├── test_backtesting/
        ├── test_optimization/
        └── test_utils/
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/TonyMa1/walk-forward-backtester.git
cd walk-forward-backtester
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the sample demonstration:

```bash
python -m walk_forward_opt
```

This will run the Walk Forward Optimization process with sample data and display performance metrics for both the MA Crossover and ATR Stop Loss strategies.

### Using in Your Own Code

```python
import pandas as pd
from walk_forward_opt.backtesting.wfo import WalkForwardOptimization
from walk_forward_opt.strategies.ma_crossover import MACrossoverStrategy
from walk_forward_opt.strategies.atr_stop_loss import ATRStopLossStrategy

# Load your price data (or use the sample data generator)
from walk_forward_opt.__main__ import generate_sample_data
data = generate_sample_data(n_days=1000, volatility=0.02)
# Or load your own data:
# data = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)['close']

# Initialize a strategy
ma_strategy = MACrossoverStrategy()

# Initialize the WFO backtester
wfo = WalkForwardOptimization(
    data=data,
    train_size=252,  # 1 year of trading days
    test_size=126,   # 6 months of trading days
    step_size=126    # Move forward 6 months at a time
)

# Run optimization with the MA Crossover strategy
results = wfo.run_optimization(
    strategy=ma_strategy.generate_signals,
    optimize_params=ma_strategy.optimize_parameters,
    short_ma_range=(5, 50),
    long_ma_range=(20, 200),
    step=5
)

# Analyze results
analysis = wfo.analyze_results(results)
print(f"Mean Training Sharpe Ratio: {analysis['mean_train_metric']:.2f}")
print(f"Mean Testing Sharpe Ratio: {analysis['mean_test_metric']:.2f}")
print(f"Total Test Returns: {analysis['total_test_returns']:.2%}")
```

### Customizing Strategies

You can modify existing strategies or create your own by implementing:

1. A signal generation function that takes price data and parameters
2. A parameter optimization function that finds the best parameters

See the existing implementations in `src/walk_forward_opt/strategies/` for examples.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

This project uses Ruff for linting and formatting:

```bash
ruff check .
ruff format .
```

### Type Checking

```bash
mypy src/
```

## Understanding the Output

When you run the backtester, it outputs:

- **Mean Training Sharpe Ratio**: Average performance on in-sample data
- **Mean Testing Sharpe Ratio**: Average performance on out-of-sample data
- **Total Test Returns**: Cumulative returns across all test windows
- **Parameter Stability**: How consistent the optimized parameters are across windows (lower is more stable)

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 