"""Tests for the Moving Average Crossover strategy.

This module contains tests for the Moving Average Crossover strategy
implementation, including signal generation and parameter optimization.
"""

from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import pytest

from walk_forward_opt.strategies.ma_crossover import (
    generate_signals,
    optimize_parameters,
    MACrossoverStrategy
)

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch

@pytest.fixture
def sample_data() -> pd.Series:
    """Create sample price data for testing."""
    # Create a simple price series with a known trend
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    prices = np.linspace(100, 200, 100) + np.random.normal(0, 1, 100)
    return pd.Series(prices, index=dates)

def test_generate_signals(sample_data: pd.Series) -> None:
    """Test the generate_signals function.
    
    Args:
        sample_data: Sample price data
    """
    # Test with valid parameters
    signals = generate_signals(sample_data, short_ma=5, long_ma=20)
    
    # Verify signal properties
    assert len(signals) == len(sample_data)
    assert all(s in [-1, 0, 1] for s in signals)
    assert all(s == 0 for s in signals[:20])  # No signals before long MA is available
    
    # Test with invalid parameters
    with pytest.raises(ValueError):
        generate_signals(sample_data, short_ma=20, long_ma=10)

def test_optimize_parameters(sample_data: pd.Series) -> None:
    """Test the optimize_parameters function.
    
    Args:
        sample_data: Sample price data
    """
    # Test parameter optimization
    best_params, best_sharpe = optimize_parameters(
        sample_data,
        short_ma_range=(5, 10),
        long_ma_range=(20, 30),
        step=5
    )
    
    # Verify optimization results
    assert isinstance(best_params, dict)
    assert 'short_ma' in best_params
    assert 'long_ma' in best_params
    assert best_params['short_ma'] < best_params['long_ma']
    assert isinstance(best_sharpe, float)
    assert not np.isnan(best_sharpe)

def test_macrossover_strategy_initialization() -> None:
    """Test MACrossoverStrategy initialization."""
    # Test valid initialization
    strategy = MACrossoverStrategy(short_ma=5, long_ma=20)
    assert strategy.short_ma == 5
    assert strategy.long_ma == 20
    
    # Test invalid initialization
    with pytest.raises(ValueError):
        MACrossoverStrategy(short_ma=20, long_ma=10)

def test_macrossover_strategy_generate_signals(sample_data: pd.Series) -> None:
    """Test MACrossoverStrategy signal generation.
    
    Args:
        sample_data: Sample price data
    """
    strategy = MACrossoverStrategy(short_ma=5, long_ma=20)
    signals = strategy.generate_signals(sample_data)
    
    # Verify signal properties
    assert len(signals) == len(sample_data)
    assert all(s in [-1, 0, 1] for s in signals)
    assert all(s == 0 for s in signals[:20])  # No signals before long MA is available

def test_macrossover_strategy_optimize_parameters(sample_data: pd.Series) -> None:
    """Test MACrossoverStrategy parameter optimization.
    
    Args:
        sample_data: Sample price data
    """
    strategy = MACrossoverStrategy()
    best_params, best_sharpe = strategy.optimize_parameters(
        sample_data,
        short_ma_range=(5, 10),
        long_ma_range=(20, 30),
        step=5
    )
    
    # Verify optimization results
    assert isinstance(best_params, dict)
    assert 'short_ma' in best_params
    assert 'long_ma' in best_params
    assert best_params['short_ma'] < best_params['long_ma']
    assert isinstance(best_sharpe, float)
    assert not np.isnan(best_sharpe)

def test_macrossover_strategy_with_trending_data() -> None:
    """Test MA Crossover strategy with strongly trending data."""
    # Create strongly trending data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    prices = np.linspace(100, 200, 100)  # Strong upward trend
    data = pd.Series(prices, index=dates)
    
    strategy = MACrossoverStrategy(short_ma=5, long_ma=20)
    signals = strategy.generate_signals(data)
    
    # In a strong uptrend, we should see more buy signals than sell signals
    buy_signals = sum(1 for s in signals if s == 1)
    sell_signals = sum(1 for s in signals if s == -1)
    assert buy_signals > sell_signals

def test_macrossover_strategy_with_choppy_data() -> None:
    """Test MA Crossover strategy with choppy (sideways) data."""
    # Create choppy data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    prices = 100 + np.random.normal(0, 5, 100)  # Random walk around 100
    data = pd.Series(prices, index=dates)
    
    strategy = MACrossoverStrategy(short_ma=5, long_ma=20)
    signals = strategy.generate_signals(data)
    
    # In choppy data, we should see a mix of buy and sell signals
    buy_signals = sum(1 for s in signals if s == 1)
    sell_signals = sum(1 for s in signals if s == -1)
    assert buy_signals > 0
    assert sell_signals > 0 