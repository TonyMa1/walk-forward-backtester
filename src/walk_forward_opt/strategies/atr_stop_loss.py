"""ATR-based Dynamic Stop Loss strategy implementation.

This module implements a trading strategy that uses the Average True Range (ATR)
indicator to set dynamic stop-loss levels based on market volatility.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

from ..utils.types import DataArray, SignalArray, OptimizationResult
from ..utils.metrics import calculate_sharpe_ratio

def calculate_atr(data: DataArray, window: int) -> np.ndarray:
    """Calculate Average True Range (ATR) for the given price data.
    
    Args:
        data: Price data array
        window: ATR calculation window
        
    Returns:
        np.ndarray: ATR values
    """
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    high = data
    low = data
    close = data.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    true_range = pd.DataFrame({
        'tr1': tr1,
        'tr2': tr2,
        'tr3': tr3
    }).max(axis=1)
    
    return true_range.rolling(window=window).mean().values

def generate_signals(
    data: DataArray,
    atr_multiplier: float,
    atr_window: int,
    min_holding_period: int = 5
) -> SignalArray:
    """Generate trading signals using ATR-based stop loss.
    
    Args:
        data: Price data array
        atr_multiplier: Multiplier for ATR to set stop loss
        atr_window: Window for ATR calculation
        min_holding_period: Minimum number of periods to hold a position
        
    Returns:
        SignalArray: List of trading signals
    """
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    # Calculate ATR
    atr = calculate_atr(data, atr_window)
    
    # Initialize signals
    signals = [0] * len(data)
    
    # Trading state variables
    in_position = False
    position_start = 0
    stop_loss = 0
    
    # Simple trend detection using short-term moving average
    short_ma = data.rolling(window=5).mean()
    
    for i in range(atr_window + 5, len(data)):
        current_price = data.iloc[i]
        
        # If not in a position, look for entry points
        if not in_position:
            # Entry signal: price above short MA and ATR is valid
            if current_price > short_ma.iloc[i] and not np.isnan(atr[i]):
                signals[i] = 1
                in_position = True
                position_start = i
                stop_loss = current_price - (atr[i] * atr_multiplier)
        
        # If in a position
        else:
            # Check minimum holding period
            if (i - position_start) >= min_holding_period:
                # Check stop loss
                if current_price < stop_loss:
                    signals[i] = -1
                    in_position = False
                else:
                    # Update stop loss (trailing stop)
                    new_stop = current_price - (atr[i] * atr_multiplier)
                    stop_loss = max(stop_loss, new_stop)
                    signals[i] = 1  # Maintain position signal
            else:
                # Within minimum holding period, maintain position
                signals[i] = 1
    
    return signals

def backtest_strategy(
    atr_multiplier: float,
    atr_window: int,
    data: DataArray,
    min_holding_period: int = 5
) -> float:
    """Backtest the ATR stop loss strategy with given parameters.
    
    Args:
        atr_multiplier: Multiplier for ATR to set stop loss
        atr_window: Window for ATR calculation
        data: Price data array
        min_holding_period: Minimum number of periods to hold a position
        
    Returns:
        float: Sharpe ratio of the strategy
    """
    signals = generate_signals(data, atr_multiplier, atr_window, min_holding_period)
    returns = np.diff(data) * np.array(signals[:-1])
    return calculate_sharpe_ratio(returns)

def optimize_parameters(
    data: DataArray,
    atr_multiplier_range: Tuple[float, float] = (0.5, 3.0),
    atr_window_range: Tuple[int, int] = (5, 30),
    min_holding_period: int = 5
) -> OptimizationResult:
    """Optimize ATR stop loss strategy parameters using grid search.
    
    Args:
        data: Price data array
        atr_multiplier_range: Tuple of (min, max) for ATR multiplier
        atr_window_range: Tuple of (min, max) for ATR window
        min_holding_period: Minimum number of periods to hold a position
        
    Returns:
        OptimizationResult: Tuple of (best_params, best_sharpe)
    """
    best_sharpe = float('-inf')
    best_params = {}
    
    # Generate parameter grid
    multipliers = np.linspace(
        atr_multiplier_range[0],
        atr_multiplier_range[1],
        num=20
    )
    windows = range(atr_window_range[0], atr_window_range[1] + 1, 5)
    
    for multiplier in multipliers:
        for window in windows:
            sharpe = backtest_strategy(
                multiplier,
                window,
                data,
                min_holding_period
            )
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {
                    'atr_multiplier': multiplier,
                    'atr_window': window
                }
    
    return best_params, best_sharpe

class ATRStopLossStrategy:
    """ATR-based Dynamic Stop Loss strategy class.
    
    This class implements a trading strategy that uses the Average True Range (ATR)
    indicator to set dynamic stop-loss levels based on market volatility.
    """
    
    def __init__(
        self,
        atr_multiplier: float = 2.0,
        atr_window: int = 14,
        min_holding_period: int = 5
    ):
        """Initialize the ATR Stop Loss strategy.
        
        Args:
            atr_multiplier: Multiplier for ATR to set stop loss
            atr_window: Window for ATR calculation
            min_holding_period: Minimum number of periods to hold a position
        """
        self.atr_multiplier = atr_multiplier
        self.atr_window = atr_window
        self.min_holding_period = min_holding_period
    
    def generate_signals(self, data: DataArray, **kwargs) -> SignalArray:
        """Generate trading signals using current parameters.
        
        Args:
            data: Price data array
            **kwargs: Optional parameter overrides
            
        Returns:
            SignalArray: List of trading signals
        """
        # Update parameters if provided
        atr_multiplier = kwargs.get('atr_multiplier', self.atr_multiplier)
        atr_window = kwargs.get('atr_window', self.atr_window)
        min_holding_period = kwargs.get('min_holding_period', self.min_holding_period)
        
        return generate_signals(
            data,
            atr_multiplier,
            atr_window,
            min_holding_period
        )
    
    def optimize_parameters(
        self,
        data: DataArray,
        atr_multiplier_range: Tuple[float, float] = (0.5, 3.0),
        atr_window_range: Tuple[int, int] = (5, 30),
        min_holding_period: int = 5
    ) -> OptimizationResult:
        """Optimize strategy parameters.
        
        Args:
            data: Price data array
            atr_multiplier_range: Tuple of (min, max) for ATR multiplier
            atr_window_range: Tuple of (min, max) for ATR window
            min_holding_period: Minimum number of periods to hold a position
            
        Returns:
            OptimizationResult: Tuple of (best_params, best_sharpe)
        """
        best_params, best_sharpe = optimize_parameters(
            data,
            atr_multiplier_range,
            atr_window_range,
            min_holding_period
        )
        
        # Update instance parameters with best values
        self.atr_multiplier = best_params['atr_multiplier']
        self.atr_window = best_params['atr_window']
        self.min_holding_period = min_holding_period
        
        return best_params, best_sharpe 