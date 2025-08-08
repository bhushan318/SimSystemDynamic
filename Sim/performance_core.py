# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 01:00:30 2025

@author: NagabhushanamTattaga
"""

# performance_core.py - Phase 1: CPU Numba JIT Optimization

import numpy as np
import numba
from numba import jit, njit, prange, types
from numba.typed import Dict, List
import time
import warnings
from typing import Tuple, Dict as PyDict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Performance configuration
@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    enable_jit: bool = True
    enable_parallel: bool = True
    enable_fastmath: bool = True
    cache_kernels: bool = True
    parallel_threshold: int = 100  # Minimum size for parallel execution
    
    # JIT compilation options
    nopython: bool = True
    nogil: bool = True
    error_model: str = 'numpy'

class ExecutionMode(Enum):
    """Execution mode selection"""
    AUTO = "auto"
    CPU_SERIAL = "cpu_serial"
    CPU_PARALLEL = "cpu_parallel"
    GPU_SINGLE = "gpu_single"
    GPU_MULTI = "gpu_multi"

# Global performance configuration
PERF_CONFIG = PerformanceConfig()

# ===============================================================================
# Core JIT-Optimized Computation Kernels
# ===============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def vectorized_stock_update(stock_values: np.ndarray, 
                           net_flows: np.ndarray, 
                           dt: float, 
                           min_values: np.ndarray, 
                           max_values: np.ndarray) -> np.ndarray:
    """
    Ultra-fast vectorized stock update with bounds checking
    
    Args:
        stock_values: Current stock values [n_stocks]
        net_flows: Net flow rates [n_stocks] 
        dt: Time step
        min_values: Minimum bounds [n_stocks]
        max_values: Maximum bounds [n_stocks]
    
    Returns:
        Updated stock values [n_stocks]
    """
    n_stocks = stock_values.shape[0]
    result = np.empty(n_stocks, dtype=np.float64)
    
    for i in prange(n_stocks):
        # Euler step with bounds checking
        new_value = stock_values[i] + net_flows[i] * dt
        
        # Apply bounds (clip operation)
        if new_value < min_values[i]:
            result[i] = min_values[i]
        elif new_value > max_values[i]:
            result[i] = max_values[i]
        else:
            result[i] = new_value
    
    return result

@njit(parallel=True, fastmath=True, cache=True)
def parallel_flow_computation(from_stocks: np.ndarray,
                             to_stocks: np.ndarray,
                             flow_rates: np.ndarray,
                             flow_connections: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel computation of flow effects on stocks
    
    Args:
        from_stocks: Source stock values [n_stocks]
        to_stocks: Target stock values [n_stocks] 
        flow_rates: Flow rates [n_flows]
        flow_connections: Connection matrix [n_flows, 2] (from_idx, to_idx)
    
    Returns:
        Tuple of (inflow_totals, outflow_totals) for each stock
    """
    n_stocks = from_stocks.shape[0]
    n_flows = flow_rates.shape[0]
    
    inflows = np.zeros(n_stocks, dtype=np.float64)
    outflows = np.zeros(n_stocks, dtype=np.float64)
    
    for i in prange(n_flows):
        from_idx = flow_connections[i, 0]
        to_idx = flow_connections[i, 1]
        rate = flow_rates[i]
        
        if from_idx >= 0:  # Valid from stock
            outflows[from_idx] += rate
        if to_idx >= 0:    # Valid to stock  
            inflows[to_idx] += rate
    
    return inflows, outflows

@njit(parallel=True, fastmath=True, cache=True)
def vectorized_rk4_step(y: np.ndarray, 
                       k1: np.ndarray, k2: np.ndarray, 
                       k3: np.ndarray, k4: np.ndarray, 
                       dt: float) -> np.ndarray:
    """
    Vectorized RK4 integration step
    
    Args:
        y: Current state [n_vars]
        k1, k2, k3, k4: RK4 slope estimates [n_vars]
        dt: Time step
    
    Returns:
        Next state [n_vars]
    """
    n_vars = y.shape[0]
    result = np.empty(n_vars, dtype=np.float64)
    
    for i in prange(n_vars):
        result[i] = y[i] + dt * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6.0
    
    return result

@njit(fastmath=True, cache=True)
def compute_derivative_single(stock_values: np.ndarray,
                             auxiliary_values: np.ndarray,
                             parameters: np.ndarray,
                             flow_formulas: np.ndarray,
                             time: float) -> np.ndarray:
    """
    Compute derivatives for single time step (used in integration)
    
    Args:
        stock_values: Current stock values [n_stocks]
        auxiliary_values: Current auxiliary values [n_aux]
        parameters: Model parameters [n_params]
        flow_formulas: Pre-compiled flow rate coefficients [n_flows, n_vars]
        time: Current time
    
    Returns:
        Derivatives [n_stocks]
    """
    n_stocks = stock_values.shape[0]
    n_flows = flow_formulas.shape[0]
    
    derivatives = np.zeros(n_stocks, dtype=np.float64)
    
    # Simple linear flow computation (can be extended for non-linear)
    for flow_idx in range(n_flows):
        # Compute flow rate from linear combination
        rate = 0.0
        for var_idx in range(stock_values.shape[0]):
            rate += flow_formulas[flow_idx, var_idx] * stock_values[var_idx]
        
        # Apply flow effect (simplified - assumes flow affects stock 0)
        if flow_idx < n_stocks:
            derivatives[flow_idx] += rate
    
    return derivatives

# ===============================================================================
# Multi-dimensional Array Operations
# ===============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def multidim_flow_application(from_values: np.ndarray,
                             to_values: np.ndarray, 
                             flow_matrix: np.ndarray,
                             dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply multi-dimensional flows using matrix operations
    
    Args:
        from_values: Source stock array [dims...]
        to_values: Target stock array [dims...]
        flow_matrix: Flow transition matrix [n_from, n_to]
        dt: Time step
    
    Returns:
        Tuple of (new_from_values, new_to_values)
    """
    # Flatten for matrix operations
    from_flat = from_values.flatten()
    to_flat = to_values.flatten()
    
    n_from = from_flat.shape[0]
    n_to = to_flat.shape[0]
    
    # Ensure matrix dimensions match
    if flow_matrix.shape[0] != n_from or flow_matrix.shape[1] != n_to:
        # Return original values if dimensions don't match
        return from_values.copy(), to_values.copy()
    
    # Compute flow effects
    outflows = np.zeros(n_from, dtype=np.float64)
    inflows = np.zeros(n_to, dtype=np.float64)
    
    for i in prange(n_from):
        for j in range(n_to):
            flow_rate = flow_matrix[i, j] * from_flat[i] * dt
            outflows[i] += flow_rate
            inflows[j] += flow_rate
    
    # Apply changes
    new_from_flat = from_flat - outflows
    new_to_flat = to_flat + inflows
    
    # Reshape back to original dimensions
    new_from = new_from_flat.reshape(from_values.shape)
    new_to = new_to_flat.reshape(to_values.shape)
    
    return new_from, new_to

@njit(parallel=True, fastmath=True, cache=True)
def cross_dimensional_flow(values: np.ndarray,
                          from_indices: np.ndarray,
                          to_indices: np.ndarray,
                          rates: np.ndarray) -> np.ndarray:
    """
    Apply cross-dimensional flows (e.g., aging, migration)
    
    Args:
        values: Multi-dimensional stock array [dim1, dim2, ...]
        from_indices: Source indices [n_flows, n_dims]
        to_indices: Target indices [n_flows, n_dims] 
        rates: Flow rates [n_flows]
    
    Returns:
        Updated values array
    """
    result = values.copy()
    n_flows = rates.shape[0]
    
    for flow_idx in range(n_flows):
        rate = rates[flow_idx]
        
        # Extract multi-dimensional indices
        from_idx = tuple(from_indices[flow_idx])
        to_idx = tuple(to_indices[flow_idx])
        
        # Apply flow
        flow_amount = result[from_idx] * rate
        result[from_idx] -= flow_amount
        result[to_idx] += flow_amount
    
    return result

# ===============================================================================
# Formula Evaluation Optimization  
# ===============================================================================

@njit(fastmath=True, cache=True)
def evaluate_linear_formula(coefficients: np.ndarray,
                           variables: np.ndarray,
                           constant: float) -> float:
    """
    Evaluate linear formula: sum(coeff[i] * var[i]) + constant
    
    Args:
        coefficients: Variable coefficients [n_vars]
        variables: Variable values [n_vars]
        constant: Constant term
    
    Returns:
        Formula result
    """
    result = constant
    for i in range(coefficients.shape[0]):
        result += coefficients[i] * variables[i]
    return result

@njit(parallel=True, fastmath=True, cache=True)
def batch_formula_evaluation(coefficients_matrix: np.ndarray,
                            variables_matrix: np.ndarray,
                            constants: np.ndarray) -> np.ndarray:
    """
    Evaluate multiple linear formulas in parallel
    
    Args:
        coefficients_matrix: Coefficient matrix [n_formulas, n_vars]
        variables_matrix: Variables matrix [n_instances, n_vars]
        constants: Constant terms [n_formulas]
    
    Returns:
        Results matrix [n_instances, n_formulas]
    """
    n_instances = variables_matrix.shape[0]
    n_formulas = coefficients_matrix.shape[0]
    
    results = np.empty((n_instances, n_formulas), dtype=np.float64)
    
    for instance_idx in prange(n_instances):
        for formula_idx in range(n_formulas):
            result = constants[formula_idx]
            for var_idx in range(coefficients_matrix.shape[1]):
                result += (coefficients_matrix[formula_idx, var_idx] * 
                          variables_matrix[instance_idx, var_idx])
            results[instance_idx, formula_idx] = result
    
    return results

# ===============================================================================
# System Dynamics Specific Functions
# ===============================================================================

@njit(fastmath=True, cache=True)
def step_function(height: float, step_time: float, current_time: float) -> float:
    """JIT-optimized STEP function"""
    return height if current_time >= step_time else 0.0

@njit(fastmath=True, cache=True)
def pulse_function(height: float, start_time: float, 
                  duration: float, current_time: float) -> float:
    """JIT-optimized PULSE function"""
    return height if start_time <= current_time <= start_time + duration else 0.0

@njit(fastmath=True, cache=True)
def ramp_function(slope: float, start_time: float, 
                 end_time: float, current_time: float) -> float:
    """JIT-optimized RAMP function"""
    if current_time < start_time:
        return 0.0
    elif current_time > end_time:
        return slope * (end_time - start_time)
    else:
        return slope * (current_time - start_time)

@njit(fastmath=True, cache=True)
def lookup_interpolation(input_value: float, 
                        x_values: np.ndarray, 
                        y_values: np.ndarray) -> float:
    """JIT-optimized linear interpolation for lookup tables"""
    n = x_values.shape[0]
    
    # Handle boundary cases
    if input_value <= x_values[0]:
        return y_values[0]
    if input_value >= x_values[n-1]:
        return y_values[n-1]
    
    # Find interpolation interval
    for i in range(n-1):
        if x_values[i] <= input_value <= x_values[i+1]:
            # Linear interpolation
            x0, x1 = x_values[i], x_values[i+1]
            y0, y1 = y_values[i], y_values[i+1]
            t = (input_value - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    
    return y_values[0]  # Fallback

# ===============================================================================
# Performance Monitoring and Benchmarking
# ===============================================================================

class JITPerformanceMonitor:
    """Monitor JIT compilation and execution performance"""
    
    def __init__(self):
        self.compilation_times = {}
        self.execution_times = {}
        self.compiled_functions = set()
    
    def benchmark_function(self, func, *args, warmup_runs=3, benchmark_runs=10):
        """Benchmark JIT function performance"""
        func_name = func.__name__
        
        # Warmup runs to trigger JIT compilation
        print(f"🔥 Warming up {func_name}...")
        compile_start = time.time()
        
        for _ in range(warmup_runs):
            try:
                _ = func(*args)
            except Exception as e:
                print(f"⚠️ Warmup failed for {func_name}: {e}")
                return None
        
        compile_time = time.time() - compile_start
        self.compilation_times[func_name] = compile_time
        self.compiled_functions.add(func_name)
        
        # Benchmark runs
        print(f"📊 Benchmarking {func_name}...")
        execution_times = []
        
        for _ in range(benchmark_runs):
            start_time = time.time()
            try:
                result = func(*args)
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
            except Exception as e:
                print(f"⚠️ Benchmark failed for {func_name}: {e}")
                return None
        
        avg_execution_time = np.mean(execution_times)
        std_execution_time = np.std(execution_times)
        
        self.execution_times[func_name] = {
            'mean': avg_execution_time,
            'std': std_execution_time,
            'runs': execution_times
        }
        
        print(f"✅ {func_name}: {avg_execution_time*1000:.2f}ms ± {std_execution_time*1000:.2f}ms")
        
        return {
            'function': func_name,
            'compilation_time': compile_time,
            'avg_execution_time': avg_execution_time,
            'std_execution_time': std_execution_time
        }
    
    def get_performance_report(self):
        """Get comprehensive performance report"""
        return {
            'compiled_functions': len(self.compiled_functions),
            'total_compilation_time': sum(self.compilation_times.values()),
            'compilation_times': self.compilation_times,
            'execution_times': self.execution_times
        }

# ===============================================================================
# Integration with Existing Code
# ===============================================================================

def optimize_existing_simulation(model):
    """Apply JIT optimizations to existing simulation model"""
    
    print("🚀 Applying CPU Numba JIT optimizations...")
    
    # Add performance monitor
    performance_monitor = JITPerformanceMonitor()
    
    # Pre-compile critical functions with actual model data
    if hasattr(model, 'stocks') and model.stocks:
        n_stocks = len(model.stocks)
        
        # Create dummy data for compilation
        dummy_stocks = np.random.rand(n_stocks) * 100
        dummy_flows = np.random.rand(n_stocks) * 10
        dummy_bounds_min = np.zeros(n_stocks)
        dummy_bounds_max = np.full(n_stocks, np.inf)
        
        # Pre-compile and benchmark critical functions
        print("\n🔥 Pre-compiling JIT kernels...")
        
        benchmarks = []
        benchmarks.append(performance_monitor.benchmark_function(
            vectorized_stock_update, dummy_stocks, dummy_flows, 0.1, 
            dummy_bounds_min, dummy_bounds_max
        ))
        
        # Create dummy flow connections
        dummy_connections = np.array([[i, (i+1) % n_stocks] for i in range(n_stocks)])
        benchmarks.append(performance_monitor.benchmark_function(
            parallel_flow_computation, dummy_stocks, dummy_stocks, 
            dummy_flows, dummy_connections
        ))
        
        # Add optimized methods to model
        model._jit_optimized = True
        model._performance_monitor = performance_monitor
        
        print(f"\n✅ JIT optimization completed!")
        print(f"   Compiled functions: {len(performance_monitor.compiled_functions)}")
        
        return model, performance_monitor
    
    else:
        print("⚠️ Model has no stocks - skipping optimization")
        return model, None

# ===============================================================================
# Testing and Validation
# ===============================================================================

def test_jit_performance():
    """Test JIT performance improvements"""
    
    print("🧪 Testing JIT Performance Improvements")
    print("=" * 50)
    
    monitor = JITPerformanceMonitor()
    
    # Test data sizes
    test_sizes = [100, 1000, 5000, 10000]
    
    for size in test_sizes:
        print(f"\n📊 Testing with {size} stocks...")
        
        # Create test data
        stocks = np.random.rand(size) * 1000
        flows = np.random.rand(size) * 50
        bounds_min = np.zeros(size)
        bounds_max = np.full(size, 10000.0)
        
        # Benchmark stock update
        result = monitor.benchmark_function(
            vectorized_stock_update, stocks, flows, 0.1, bounds_min, bounds_max,
            warmup_runs=2, benchmark_runs=5
        )
        
        if result:
            throughput = size / result['avg_execution_time']
            print(f"   Throughput: {throughput:.0f} stocks/second")
    
    print(f"\n📈 Performance Summary:")
    report = monitor.get_performance_report()
    print(f"   Total compilation time: {report['total_compilation_time']:.3f}s")
    print(f"   Compiled functions: {report['compiled_functions']}")

if __name__ == "__main__":
    test_jit_performance()