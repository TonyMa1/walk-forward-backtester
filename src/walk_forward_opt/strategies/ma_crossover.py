"""Moving Average Crossover strategy implementation.

This module implements a Moving Average Crossover strategy that generates
trading signals based on the intersection of short and long-term moving averages.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

from ..utils.types import DataArray, SignalArray, OptimizationResult
from ..utils.metrics import calculate_sharpe_ratio

def generate_signals(data: DataArray, short_ma: int, long_ma: int) -> SignalArray:
    """Generate trading signals using Moving Average Crossover strategy.
    
    Args:
        data: Price data array
        short_ma: Short-term moving average period
        long_ma: Long-term moving average period
        
    Returns:
        SignalArray: List of trading signals (1 for buy, -1 for sell, 0 for hold)
    """
    if short_ma >= long_ma:
        raise ValueError("Short MA period must be less than long MA period")
    
    # Convert to pandas Series for easier MA calculation
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    # Calculate moving averages
    short_ma_series = data.rolling(window=short_ma).mean()
    long_ma_series = data.rolling(window=long_ma).mean()
    
    # Generate signals
    signals = [0] * len(data)
    position = 0  # 0 = no position, 1 = long, -1 = short
    
    for i in range(long_ma, len(data)):
        # Crossover conditions
        buy_signal = short_ma_series.iloc[i] > long_ma_series.iloc[i] and short_ma_series.iloc[i-1] <= long_ma_series.iloc[i-1]
        sell_signal = short_ma_series.iloc[i] < long_ma_series.iloc[i] and short_ma_series.iloc[i-1] >= long_ma_series.iloc[i-1]
        
        if buy_signal and position <= 0:
            position = 1
            signals[i] = 1
        elif sell_signal and position >= 0:
            position = -1
            signals[i] = -1
        else:
            # Maintain position
            signals[i] = position
    
    return signals

def optimize_parameters(
    data: DataArray,
    short_ma_range: Tuple[int, int] = (5, 50),
    long_ma_range: Tuple[int, int] = (20, 200),
    step: int = 5
) -> OptimizationResult:
    """Optimize MA Crossover strategy parameters using grid search.
    
    Args:
        data: Price data array
        short_ma_range: Tuple of (min, max) for short MA period
        long_ma_range: Tuple of (min, max) for long MA period
        step: Step size for parameter grid
        
    Returns:
        OptimizationResult: Tuple of (best_params, best_sharpe)
    """
    best_sharpe = float('-inf')
    best_params = {}
    
    # Generate parameter grid
    short_periods = range(short_ma_range[0], short_ma_range[1] + 1, step)
    long_periods = range(long_ma_range[0], long_ma_range[1] + 1, step)
    
    for short_ma in short_periods:
        for long_ma in long_periods:
            if short_ma >= long_ma:
                continue
                
            # Generate signals
            signals = generate_signals(data, short_ma, long_ma)
            
            # Calculate returns
            returns = np.diff(data) * np.array(signals[:-1])
            
            # Calculate Sharpe ratio
            sharpe = calculate_sharpe_ratio(returns)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {
                    'short_ma': short_ma,
                    'long_ma': long_ma
                }
    
    return best_params, best_sharpe

class MACrossoverStrategy:
    """Moving Average Crossover strategy class.
    
    This class implements the Moving Average Crossover strategy with methods
    for generating signals and optimizing parameters.
    """
    
    def __init__(self, short_ma: int = 20, long_ma: int = 50):
        """Initialize the MA Crossover strategy.
        
        Args:
            short_ma: Short-term moving average period
            long_ma: Long-term moving average period
        """
        if short_ma >= long_ma:
            raise ValueError("Short MA period must be less than long MA period")
        
        self.short_ma = short_ma
        self.long_ma = long_ma
    
    def generate_signals(self, data: DataArray, **kwargs) -> SignalArray:
        """Generate trading signals using current parameters.
        
        Args:
            data: Price data array
            **kwargs: Optional parameter overrides
            
        Returns:
            SignalArray: List of trading signals
        """
        # Update parameters if provided
        short_ma = kwargs.get('short_ma', self.short_ma)
        long_ma = kwargs.get('long_ma', self.long_ma)
        
        return generate_signals(data, short_ma, long_ma)
    
    def optimize_parameters(
        self,
        data: DataArray,
        short_ma_range: Tuple[int, int] = (5, 50),
        long_ma_range: Tuple[int, int] = (20, 200),
        step: int = 5
    ) -> OptimizationResult:
        """Optimize strategy parameters.
        
        Args:
            data: Price data array
            short_ma_range: Tuple of (min, max) for short MA period
            long_ma_range: Tuple of (min, max) for long MA period
            step: Step size for parameter grid
            
        Returns:
            OptimizationResult: Tuple of (best_params, best_sharpe)
        """
        best_params, best_sharpe = optimize_parameters(data, short_ma_range, long_ma_range, step)
        
        # Update instance parameters with best values
        self.short_ma = best_params['short_ma']
        self.long_ma = best_params['long_ma']
        
        return best_params, best_sharpe 