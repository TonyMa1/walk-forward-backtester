"""Utility functions for calculating trading performance metrics.

This module provides functions for calculating various trading performance metrics
such as returns, Sharpe ratio, and other risk-adjusted measures.
"""

from typing import List, Union
import numpy as np
import pandas as pd

from .types import DataArray, SignalArray

def calculate_returns(signals: SignalArray, data: DataArray) -> float:
    """Calculate strategy returns based on signals and price data.
    
    Args:
        signals: List of trading signals (1 for buy, -1 for sell, 0 for hold)
        data: Price data array
        
    Returns:
        float: Total strategy returns
    """
    if len(signals) != len(data):
        raise ValueError("Signals and data must have the same length")
    
    # Convert to numpy arrays for vectorized operations
    signals = np.array(signals)
    data = np.array(data)
    
    # Calculate price changes
    price_changes = np.diff(data)
    
    # Calculate returns (shift signals by 1 to align with price changes)
    returns = signals[:-1] * price_changes
    
    return float(np.sum(returns))

def calculate_sharpe_ratio(returns: Union[List[float], np.ndarray], risk_free_rate: float = 0.02) -> float:
    """Calculate the Sharpe ratio for a series of returns.
    
    Args:
        returns: List or array of returns
        risk_free_rate: Annual risk-free rate (default: 0.02 or 2%)
        
    Returns:
        float: Sharpe ratio
    """
    returns = np.array(returns)
    
    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / 252)  # Convert annual to daily
    
    # Calculate mean and standard deviation
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    
    if std_excess_return == 0:
        return 0.0
    
    # Annualize the Sharpe ratio
    sharpe_ratio = np.sqrt(252) * (mean_excess_return / std_excess_return)
    
    return float(sharpe_ratio)

def calculate_max_drawdown(returns: Union[List[float], np.ndarray]) -> float:
    """Calculate the maximum drawdown from a series of returns.
    
    Args:
        returns: List or array of returns
        
    Returns:
        float: Maximum drawdown as a percentage
    """
    returns = np.array(returns)
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + returns)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_returns)
    
    # Calculate drawdowns
    drawdowns = (running_max - cumulative_returns) / running_max
    
    # Get maximum drawdown
    max_drawdown = np.max(drawdowns)
    
    return float(max_drawdown)

def calculate_win_rate(signals: SignalArray, data: DataArray) -> float:
    """Calculate the win rate of a trading strategy.
    
    Args:
        signals: List of trading signals (1 for buy, -1 for sell, 0 for hold)
        data: Price data array
        
    Returns:
        float: Win rate as a percentage
    """
    if len(signals) != len(data):
        raise ValueError("Signals and data must have the same length")
    
    # Convert to numpy arrays for vectorized operations
    signals = np.array(signals)
    data = np.array(data)
    
    # Calculate price changes
    price_changes = np.diff(data)
    
    # Calculate returns for each trade
    returns = signals[:-1] * price_changes
    
    # Count winning trades
    winning_trades = np.sum(returns > 0)
    total_trades = np.sum(np.abs(signals[:-1]))  # Count non-zero signals
    
    if total_trades == 0:
        return 0.0
    
    return float(winning_trades / total_trades) 