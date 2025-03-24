"""Tests for the ATR-based Dynamic Stop Loss strategy.

This module contains tests for the ATR-based Dynamic Stop Loss strategy
implementation, including signal generation and parameter optimization.
"""

from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import pytest

from walk_forward_opt.strategies.atr_stop_loss import (
    calculate_atr,
    generate_signals,
    backtest_strategy,
    optimize_parameters,
    ATRStopLossStrategy
)

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch

@pytest.fixture
def sample_data() -> pd.Series:
    """Create sample price data for testing."""
    # Create a simple price series with a known trend and volatility
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    prices = np.linspace(100, 200, 100) + np.random.normal(0, 2, 100)
    return pd.Series(prices, index=dates)

def test_calculate_atr(sample_data: pd.Series) -> None:
    """Test the calculate_atr function.
    
    Args:
        sample_data: Sample price data
    """
    # Test ATR calculation
    atr = calculate_atr(sample_data, window=14)
    
    # Verify ATR properties
    assert len(atr) == len(sample_data)
    assert all(atr >= 0)  # ATR should be non-negative
    assert not np.any(np.isnan(atr[14:]))  # No NaN values after window period
    assert np.all(np.isnan(atr[:13]))  # NaN values before window period

def test_generate_signals(sample_data: pd.Series) -> None:
    """Test the generate_signals function.
    
    Args:
        sample_data: Sample price data
    """
    # Test with valid parameters
    signals = generate_signals(
        sample_data,
        atr_multiplier=2.0,
        atr_window=14,
        min_holding_period=5
    )
    
    # Verify signal properties
    assert len(signals) == len(sample_data)
    assert all(s in [-1, 0, 1] for s in signals)
    assert all(s == 0 for s in signals[:14])  # No signals before ATR is available
    
    # Test with invalid parameters
    with pytest.raises(ValueError):
        generate_signals(sample_data, atr_multiplier=-1, atr_window=14)

def test_backtest_strategy(sample_data: pd.Series) -> None:
    """Test the backtest_strategy function.
    
    Args:
        sample_data: Sample price data
    """
    # Test strategy backtesting
    sharpe = backtest_strategy(
        atr_multiplier=2.0,
        atr_window=14,
        data=sample_data,
        min_holding_period=5
    )
    
    # Verify backtest results
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)

def test_optimize_parameters(sample_data: pd.Series) -> None:
    """Test the optimize_parameters function.
    
    Args:
        sample_data: Sample price data
    """
    # Test parameter optimization
    best_params, best_sharpe = optimize_parameters(
        sample_data,
        min_atr_multiplier=0.5,
        min_holding_period=5
    )
    
    # Verify optimization results
    assert isinstance(best_params, dict)
    assert 'atr_multiplier' in best_params
    assert 'atr_window' in best_params
    assert best_params['atr_multiplier'] >= 0.5
    assert 5 <= best_params['atr_window'] <= 60
    assert isinstance(best_sharpe, float)
    assert not np.isnan(best_sharpe)

def test_atr_stop_loss_strategy_initialization() -> None:
    """Test ATRStopLossStrategy initialization."""
    # Test valid initialization
    strategy = ATRStopLossStrategy(
        atr_multiplier=2.0,
        atr_window=14,
        min_holding_period=5
    )
    assert strategy.atr_multiplier == 2.0
    assert strategy.atr_window == 14
    assert strategy.min_holding_period == 5

def test_atr_stop_loss_strategy_generate_signals(sample_data: pd.Series) -> None:
    """Test ATRStopLossStrategy signal generation.
    
    Args:
        sample_data: Sample price data
    """
    strategy = ATRStopLossStrategy(
        atr_multiplier=2.0,
        atr_window=14,
        min_holding_period=5
    )
    signals = strategy.generate_signals(sample_data)
    
    # Verify signal properties
    assert len(signals) == len(sample_data)
    assert all(s in [-1, 0, 1] for s in signals)
    assert all(s == 0 for s in signals[:14])  # No signals before ATR is available

def test_atr_stop_loss_strategy_optimize_parameters(sample_data: pd.Series) -> None:
    """Test ATRStopLossStrategy parameter optimization.
    
    Args:
        sample_data: Sample price data
    """
    strategy = ATRStopLossStrategy()
    best_params, best_sharpe = strategy.optimize_parameters(
        sample_data,
        min_atr_multiplier=0.5,
        min_holding_period=5
    )
    
    # Verify optimization results
    assert isinstance(best_params, dict)
    assert 'atr_multiplier' in best_params
    assert 'atr_window' in best_params
    assert best_params['atr_multiplier'] >= 0.5
    assert 5 <= best_params['atr_window'] <= 60
    assert isinstance(best_sharpe, float)
    assert not np.isnan(best_sharpe)

def test_atr_stop_loss_strategy_with_volatile_data() -> None:
    """Test ATR Stop Loss strategy with highly volatile data."""
    # Create highly volatile data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    prices = 100 + np.random.normal(0, 10, 100)  # High volatility
    data = pd.Series(prices, index=dates)
    
    strategy = ATRStopLossStrategy(
        atr_multiplier=2.0,
        atr_window=14,
        min_holding_period=5
    )
    signals = strategy.generate_signals(data)
    
    # In volatile data, we should see frequent stop losses
    sell_signals = sum(1 for s in signals if s == -1)
    assert sell_signals > 0

def test_atr_stop_loss_strategy_with_stable_data() -> None:
    """Test ATR Stop Loss strategy with stable data."""
    # Create stable data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    prices = 100 + np.random.normal(0, 0.1, 100)  # Low volatility
    data = pd.Series(prices, index=dates)
    
    strategy = ATRStopLossStrategy(
        atr_multiplier=2.0,
        atr_window=14,
        min_holding_period=5
    )
    signals = strategy.generate_signals(data)
    
    # In stable data, we should see fewer stop losses
    sell_signals = sum(1 for s in signals if s == -1)
    assert sell_signals < 20  # Arbitrary threshold 