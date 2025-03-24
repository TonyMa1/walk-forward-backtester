"""Walk Forward Optimization (WFO) implementation.

This module implements the Walk Forward Optimization process for backtesting
trading strategies with parameter optimization.
"""

from typing import List, Tuple, Dict, Any, Callable, Optional
import numpy as np
import pandas as pd

from ..utils.types import DataArray, SignalArray, OptimizationResult
from ..utils.metrics import calculate_returns, calculate_sharpe_ratio

class WalkForwardOptimization:
    """Walk Forward Optimization class for backtesting trading strategies.
    
    This class implements the Walk Forward Optimization process, which involves:
    1. Splitting historical data into training and testing periods
    2. Optimizing strategy parameters on training data
    3. Testing the optimized strategy on subsequent testing data
    4. Rolling the windows forward and repeating
    """
    
    def __init__(
        self,
        data: DataArray,
        train_size: int = 252,
        test_size: int = 126,
        step_size: Optional[int] = None
    ):
        """Initialize the Walk Forward Optimization process.
        
        Args:
            data: Historical price data
            train_size: Size of training window in periods
            test_size: Size of testing window in periods
            step_size: Size of step between windows (default: test_size)
        """
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
        
        self.data = data
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        
        if train_size <= 0 or test_size <= 0 or self.step_size <= 0:
            raise ValueError("Window sizes must be positive")
        
        if train_size <= test_size:
            raise ValueError("Training window must be larger than testing window")
    
    def split_data(self) -> List[Tuple[pd.Series, pd.Series]]:
        """Split data into training and testing windows.
        
        Returns:
            List of tuples containing (train_data, test_data) for each window
        """
        splits = []
        total_size = len(self.data)
        
        for i in range(0, total_size - self.train_size - self.test_size + 1, self.step_size):
            train_start = i
            train_end = i + self.train_size
            test_start = train_end
            test_end = test_start + self.test_size
            
            train_data = self.data[train_start:train_end]
            test_data = self.data[test_start:test_end]
            
            splits.append((train_data, test_data))
        
        return splits
    
    def run_optimization(
        self,
        strategy: Callable[[DataArray, Any], SignalArray],
        optimize_params: Callable[[DataArray], OptimizationResult],
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Run the Walk Forward Optimization process.
        
        Args:
            strategy: Function that generates trading signals
            optimize_params: Function that optimizes strategy parameters
            **kwargs: Additional arguments to pass to strategy and optimize_params
            
        Returns:
            List of dictionaries containing optimization results for each window
        """
        splits = self.split_data()
        results = []
        
        for i, (train_data, test_data) in enumerate(splits):
            print(f"\nProcessing window {i+1}/{len(splits)}:")
            print(f"  Train data: {len(train_data)} periods from {train_data.index[0]} to {train_data.index[-1]}")
            print(f"  Test data: {len(test_data)} periods from {test_data.index[0]} to {test_data.index[-1]}")
            
            # Optimize parameters on training data
            best_params, train_metric = optimize_params(train_data, **kwargs)
            print(f"  Optimized parameters: {best_params}")
            print(f"  Training Sharpe: {train_metric:.4f}")
            
            # Apply optimized strategy to test data
            test_signals = strategy(test_data, **best_params)
            non_zero_signals = sum(1 for s in test_signals if s != 0)
            print(f"  Test signals generated: {non_zero_signals} (non-zero out of {len(test_signals)})")
            
            # Ensure we have valid signals
            if non_zero_signals == 0:
                print(f"  Warning: No trading signals generated for test window {i+1}")
            
            test_returns = calculate_returns(test_signals, test_data)
            test_sharpe = calculate_sharpe_ratio(test_returns)
            print(f"  Test Sharpe: {test_sharpe:.4f}")
            print(f"  Test Returns: {np.sum(test_returns):.4f}")
            
            # Store results
            window_result = {
                'train_metric': train_metric,
                'test_metric': test_sharpe,
                'parameters': best_params,
                'test_returns': test_returns,
                'test_signals': test_signals
            }
            
            results.append(window_result)
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the results of the Walk Forward Optimization process.
        
        Args:
            results: List of optimization results for each window
            
        Returns:
            Dictionary containing analysis metrics
        """
        if not results:
            return {
                'mean_train_metric': 0.0,
                'std_train_metric': 0.0,
                'mean_test_metric': 0.0,
                'std_test_metric': 0.0,
                'total_test_returns': 0.0,
                'sharpe_ratio': 0.0,
                'parameter_stability': {}
            }
        
        train_metrics = [r['train_metric'] for r in results]
        test_metrics = [r['test_metric'] for r in results]
        
        # Calculate aggregate metrics
        analysis = {
            'mean_train_metric': float(np.mean(train_metrics)),
            'std_train_metric': float(np.std(train_metrics)),
            'mean_test_metric': float(np.mean(test_metrics)),
            'std_test_metric': float(np.std(test_metrics)),
        }
        
        # Calculate total test returns and Sharpe ratio
        all_returns = []
        for result in results:
            returns = result['test_returns']
            if isinstance(returns, np.ndarray) and returns.size > 0:
                all_returns.extend(returns.tolist())
        
        if all_returns:
            all_returns = np.array(all_returns)
            analysis['total_test_returns'] = float(np.sum(all_returns))
            analysis['sharpe_ratio'] = float(calculate_sharpe_ratio(all_returns))
        else:
            analysis['total_test_returns'] = 0.0
            analysis['sharpe_ratio'] = 0.0
        
        # Calculate parameter stability
        analysis['parameter_stability'] = self._calculate_parameter_stability(results)
        
        return analysis
    
    def _calculate_parameter_stability(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate the stability of optimized parameters across windows.
        
        Args:
            results: List of optimization results for each window
            
        Returns:
            Dictionary containing parameter stability metrics
        """
        if not results:
            return {}
        
        # Get all parameter names
        param_names = results[0]['parameters'].keys()
        param_values = {name: [] for name in param_names}
        
        # Collect parameter values across windows
        for result in results:
            for name in param_names:
                param_values[name].append(result['parameters'][name])
        
        # Calculate coefficient of variation for each parameter
        stability = {}
        for name, values in param_values.items():
            mean = np.mean(values)
            std = np.std(values)
            # Avoid division by zero
            stability[name] = float(std / mean) if mean != 0 else float('inf')
        
        return stability 