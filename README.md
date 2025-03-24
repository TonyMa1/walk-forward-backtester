# Walk Forward Optimization (WFO) Backtester

A Python implementation of Walk Forward Optimization (WFO) for trading strategy backtesting, incorporating Bayesian Optimization for parameter optimization.

## Overview

This project implements Walk Forward Optimization (WFO) for backtesting and optimizing trading strategies. It enhances the basic WFO process by incorporating Bayesian Optimization for parameter optimization and explores dynamic stop-loss strategies.

### Key Features

- Walk Forward Optimization implementation with rolling windows
- Bayesian Optimization for parameter optimization
- Multiple trading strategy implementations:
  - Moving Average Crossover
  - ATR-Based Dynamic Stop Loss
- Comprehensive testing suite
- Type hints and documentation

## Project Structure

```
walk_forward_opt/
├── src/
│   └── walk_forward_opt/
│       ├── strategies/      # Trading strategy implementations
│       ├── backtesting/     # WFO and backtesting logic
│       ├── optimization/    # Parameter optimization algorithms
│       └── utils/          # Utility functions
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
git clone https://github.com/yourusername/walk-forward-opt.git
cd walk-forward-opt
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Example usage of the WFO backtester:

```python
from walk_forward_opt.backtesting import WalkForwardOptimization
from walk_forward_opt.strategies import trading_strategy_ma_crossover

# Initialize WFO with your data
wfo = WalkForwardOptimization(data=your_data, train_size=252, test_size=126)

# Run optimization
results = wfo.run_optimization(strategy=trading_strategy_ma_crossover)
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

This project uses Ruff for linting and formatting. To run:

```bash
ruff check .
ruff format .
```

### Type Checking

```bash
mypy src/
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 