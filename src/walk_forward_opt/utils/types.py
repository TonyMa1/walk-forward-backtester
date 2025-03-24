"""Type definitions for the Walk Forward Optimization project.

This module contains type definitions and aliases used throughout the project
to ensure type safety and improve code readability.
"""

from typing import List, Tuple, Union, Callable, Any, Dict, TypeVar, Protocol

import numpy as np
import pandas as pd

# Type aliases for common data structures
DataArray = Union[List[float], np.ndarray, pd.Series]
SignalArray = List[int]  # 1 for buy, -1 for sell, 0 for hold
ParameterDict = Dict[str, Any]

# Type for strategy functions
StrategyFunction = Callable[[DataArray, Any], SignalArray]

# Type for objective functions used in optimization
ObjectiveFunction = Callable[[Any], float]

# Type for optimization results
OptimizationResult = Tuple[ParameterDict, float]

# Type variable for generic data types
T = TypeVar('T', bound=Union[float, int])

class StrategyProtocol(Protocol):
    """Protocol defining the interface for trading strategies."""
    
    def generate_signals(self, data: DataArray, **params: Any) -> SignalArray:
        """Generate trading signals for the given data and parameters."""
        ...
    
    def optimize_parameters(self, data: DataArray, **kwargs: Any) -> OptimizationResult:
        """Optimize strategy parameters for the given data."""
        ... 