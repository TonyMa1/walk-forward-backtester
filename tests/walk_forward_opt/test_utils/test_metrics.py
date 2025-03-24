"""Tests for the metrics module.

This module contains tests for the trading performance metrics calculation
functions in the metrics module.
"""

from typing import TYPE_CHECKING
import numpy as np
import pytest

from walk_forward_opt.utils.metrics import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_win_rate
)

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch

@pytest.fixture
def sample_data() -> np.ndarray:
    """Create sample price data for testing."""
    return np.array([100.0, 101.0, 102.0, 101.5, 102.5, 103.0, 102.0, 101.0, 100.5, 101.0])

@pytest.fixture
def sample_signals() -> list[int]:
    """Create sample trading signals for testing."""
    return [0, 1, 0, -1, 1, 0, -1, 0, 1, 0]

def test_calculate_returns(sample_data: np.ndarray, sample_signals: list[int]) -> None:
    """Test the calculate_returns function.
    
    Args:
        sample_data: Sample price data
        sample_signals: Sample trading signals
    """
    # Calculate returns
    returns = calculate_returns(sample_signals, sample_data)
    
    # Expected returns calculation:
    # Period 1: 1 * (101.0 - 100.0) = 1.0
    # Period 3: -1 * (101.5 - 102.0) = 0.5
    # Period 4: 1 * (102.5 - 101.5) = 1.0
    # Period 6: -1 * (102.0 - 103.0) = 1.0
    # Period 8: 1 * (101.0 - 100.5) = 0.5
    expected_returns = 4.0
    
    assert np.isclose(returns, expected_returns)

def test_calculate_returns_empty() -> None:
    """Test calculate_returns with empty data."""
    with pytest.raises(ValueError):
        calculate_returns([], [])

def test_calculate_returns_mismatched_lengths() -> None:
    """Test calculate_returns with mismatched data lengths."""
    with pytest.raises(ValueError):
        calculate_returns([1, 0], [100.0, 101.0, 102.0])

def test_calculate_sharpe_ratio() -> None:
    """Test the calculate_sharpe_ratio function."""
    # Create sample returns with known properties
    returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
    
    # Calculate Sharpe ratio
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
    
    # Verify the result is reasonable
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)
    assert not np.isinf(sharpe)

def test_calculate_sharpe_ratio_zero_std() -> None:
    """Test calculate_sharpe_ratio with zero standard deviation."""
    returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe == 0.0

def test_calculate_max_drawdown() -> None:
    """Test the calculate_max_drawdown function."""
    # Create sample returns that will produce a known maximum drawdown
    returns = np.array([0.01, 0.02, -0.05, 0.01, 0.02])
    
    # Calculate maximum drawdown
    max_dd = calculate_max_drawdown(returns)
    
    # Verify the result is reasonable
    assert isinstance(max_dd, float)
    assert 0 <= max_dd <= 1
    assert not np.isnan(max_dd)

def test_calculate_win_rate(sample_data: np.ndarray, sample_signals: list[int]) -> None:
    """Test the calculate_win_rate function.
    
    Args:
        sample_data: Sample price data
        sample_signals: Sample trading signals
    """
    # Calculate win rate
    win_rate = calculate_win_rate(sample_signals, sample_data)
    
    # Expected win rate calculation:
    # Total trades: 5 (from sample_signals)
    # Winning trades: 3 (periods 1, 4, and 8)
    expected_win_rate = 0.6
    
    assert np.isclose(win_rate, expected_win_rate)

def test_calculate_win_rate_no_trades() -> None:
    """Test calculate_win_rate with no trades."""
    signals = [0] * 10
    data = np.ones(10)
    win_rate = calculate_win_rate(signals, data)
    assert win_rate == 0.0

def test_calculate_win_rate_mismatched_lengths() -> None:
    """Test calculate_win_rate with mismatched data lengths."""
    with pytest.raises(ValueError):
        calculate_win_rate([1, 0], [100.0, 101.0, 102.0]) 