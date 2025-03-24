"""Main script for demonstrating Walk Forward Optimization.

This script provides examples of using Walk Forward Optimization with different
trading strategies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from walk_forward_opt.backtesting.wfo import WalkForwardOptimization
from walk_forward_opt.strategies.ma_crossover import MACrossoverStrategy
from walk_forward_opt.strategies.atr_stop_loss import ATRStopLossStrategy

def generate_sample_data(n_days: int = 1000, volatility: float = 0.02) -> pd.Series:
    """Generate sample price data for testing.
    
    Args:
        n_days: Number of days of data to generate
        volatility: Daily price volatility
        
    Returns:
        pd.Series: Sample price data
    """
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    prices = 100 * (1 + np.random.normal(0, volatility, n_days).cumsum())
    return pd.Series(prices, index=dates)

def main() -> None:
    """Run examples of Walk Forward Optimization with different strategies."""
    # Generate sample data
    data = generate_sample_data()
    
    # Example 1: MA Crossover Strategy
    print("\n=== MA Crossover Strategy ===")
    ma_strategy = MACrossoverStrategy()
    wfo = WalkForwardOptimization(
        data=data,
        train_size=252,  # 1 year of trading days
        test_size=126,   # 6 months of trading days
        step_size=126    # 6-month step
    )
    
    results = wfo.run_optimization(
        strategy=ma_strategy.generate_signals,
        optimize_params=ma_strategy.optimize_parameters,
        short_ma_range=(5, 50),
        long_ma_range=(20, 200),
        step=5
    )
    
    analysis = wfo.analyze_results(results)
    print(f"Mean Training Sharpe Ratio: {analysis['mean_train_metric']:.2f}")
    print(f"Mean Testing Sharpe Ratio: {analysis['mean_test_metric']:.2f}")
    print(f"Total Test Returns: {analysis['total_test_returns']:.2%}")
    print(f"Parameter Stability:")
    for param, stability in analysis['parameter_stability'].items():
        print(f"  {param}: {stability:.2f}")
    
    # Example 2: ATR Stop Loss Strategy
    print("\n=== ATR Stop Loss Strategy ===")
    atr_strategy = ATRStopLossStrategy()
    wfo = WalkForwardOptimization(
        data=data,
        train_size=252,
        test_size=126,
        step_size=126
    )
    
    results = wfo.run_optimization(
        strategy=atr_strategy.generate_signals,
        optimize_params=atr_strategy.optimize_parameters,
        atr_multiplier_range=(0.5, 3.0),
        atr_window_range=(5, 30),
        min_holding_period=5
    )
    
    analysis = wfo.analyze_results(results)
    print(f"Mean Training Sharpe Ratio: {analysis['mean_train_metric']:.2f}")
    print(f"Mean Testing Sharpe Ratio: {analysis['mean_test_metric']:.2f}")
    print(f"Total Test Returns: {analysis['total_test_returns']:.2%}")
    print(f"Parameter Stability:")
    for param, stability in analysis['parameter_stability'].items():
        print(f"  {param}: {stability:.2f}")

if __name__ == "__main__":
    main() 