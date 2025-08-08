# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 01:40:10 2025

@author: NagabhushanamTattaga
"""

# gpu_kernels.py - Phase 3: GPU Kernels & Automatic Selection

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import math

# GPU Libraries with fallback
try:
    import cupy as cp
    from cupy import cuda
    import cupyx
    from cupyx.scipy import sparse as cp_sparse
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    # Mock for compatibility
    class MockCuPy:
        @staticmethod
        def array(x): return np.array(x)
        @staticmethod
        def asarray(x): return np.asarray(x)
        @staticmethod
        def asnumpy(x): return np.asarray(x)
        @staticmethod
        def zeros_like(x): return np.zeros_like(x)
        @staticmethod
        def empty(shape, dtype=np.float64): return np.empty(shape, dtype)
    cp = MockCuPy()

# Import from previous phases
try:
    from performance_core import (vectorized_stock_update, parallel_flow_computation, 
                                vectorized_rk4_step, PERF_CONFIG)
    from gpu_infrastructure import PerformanceSystem, ExecutionContext, GPUMemoryPool
    CPU_KERNELS_AVAILABLE = True
except ImportError:
    CPU_KERNELS_AVAILABLE = False
    print("⚠️ CPU kernels not available - install from Phase 1")

# ===============================================================================
# GPU Kernel Definitions
# ===============================================================================

# Raw CUDA kernels for maximum performance
STOCK_UPDATE_KERNEL = r'''
extern "C" __global__
void stock_update_kernel(const double* __restrict__ stock_values,
                        const double* __restrict__ net_flows,
                        double* __restrict__ result,
                        const double* __restrict__ min_values,
                        const double* __restrict__ max_values,
                        double dt,
                        int n_stocks) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_stocks) {
        double new_value = stock_values[idx] + net_flows[idx] * dt;
        
        // Apply bounds
        if (new_value < min_values[idx]) {
            result[idx] = min_values[idx];
        } else if (new_value > max_values[idx]) {
            result[idx] = max_values[idx];
        } else {
            result[idx] = new_value;
        }
    }
}
'''

FLOW_COMPUTATION_KERNEL = r'''
extern "C" __global__
void flow_computation_kernel(const double* __restrict__ from_stocks,
                           const double* __restrict__ to_stocks,
                           const double* __restrict__ flow_rates,
                           const int* __restrict__ flow_connections,
                           double* __restrict__ inflows,
                           double* __restrict__ outflows,
                           int n_stocks, int n_flows) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_flows) {
        int from_idx = flow_connections[idx * 2];
        int to_idx = flow_connections[idx * 2 + 1];
        double rate = flow_rates[idx];
        
        if (from_idx >= 0 && from_idx < n_stocks) {
            atomicAdd(&outflows[from_idx], rate);
        }
        if (to_idx >= 0 && to_idx < n_stocks) {
            atomicAdd(&inflows[to_idx], rate);
        }
    }
}
'''

MULTIDIM_FLOW_KERNEL = r'''
extern "C" __global__
void multidim_flow_kernel(const double* __restrict__ from_values,
                         double* __restrict__ to_values,
                         const double* __restrict__ flow_matrix,
                         double dt,
                         int n_from, int n_to) {
    
    int to_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (to_idx < n_to) {
        double inflow_sum = 0.0;
        
        for (int from_idx = 0; from_idx < n_from; from_idx++) {
            double flow_rate = flow_matrix[from_idx * n_to + to_idx];
            inflow_sum += flow_rate * from_values[from_idx] * dt;
        }
        
        to_values[to_idx] += inflow_sum;
    }
}
'''

RK4_KERNEL = r'''
extern "C" __global__
void rk4_step_kernel(const double* __restrict__ y,
                    const double* __restrict__ k1,
                    const double* __restrict__ k2,
                    const double* __restrict__ k3,
                    const double* __restrict__ k4,
                    double* __restrict__ result,
                    double dt,
                    int n_vars) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_vars) {
        result[idx] = y[idx] + dt * (k1[idx] + 2.0*k2[idx] + 2.0*k3[idx] + k4[idx]) / 6.0;
    }
}
'''

# ===============================================================================
# GPU Kernel Wrapper Classes
# ===============================================================================

class GPUKernelManager:
    """Manages compiled GPU kernels for optimal performance"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.compiled_kernels = {}
        self.kernel_cache = {}
        self.compilation_times = {}
        
        if GPU_AVAILABLE:
            self._compile_kernels()
    
    def _compile_kernels(self):
        """Compile all CUDA kernels"""
        
        print(f"🔧 Compiling GPU kernels for device {self.device_id}...")
        
        kernels_to_compile = {
            'stock_update': STOCK_UPDATE_KERNEL,
            'flow_computation': FLOW_COMPUTATION_KERNEL,
            'multidim_flow': MULTIDIM_FLOW_KERNEL,
            'rk4_step': RK4_KERNEL
        }
        
        with cp.cuda.Device(self.device_id):
            for kernel_name, kernel_code in kernels_to_compile.items():
                try:
                    start_time = time.time()
                    
                    # Compile kernel
                    compiled_kernel = cp.RawKernel(kernel_code, f"{kernel_name}_kernel")
                    
                    compile_time = time.time() - start_time
                    
                    self.compiled_kernels[kernel_name] = compiled_kernel
                    self.compilation_times[kernel_name] = compile_time
                    
                    print(f"   ✅ {kernel_name}: {compile_time:.3f}s")
                    
                except Exception as e:
                    print(f"   ❌ {kernel_name}: {e}")
                    warnings.warn(f"Failed to compile {kernel_name} kernel: {e}")
        
        print(f"✅ Compiled {len(self.compiled_kernels)}/{len(kernels_to_compile)} kernels")
    
    def get_optimal_block_size(self, kernel_name: str, problem_size: int) -> Tuple[int, int]:
        """Calculate optimal CUDA block and grid sizes"""
        
        if kernel_name not in self.compiled_kernels:
            return 256, (problem_size + 255) // 256  # Default fallback
        
        kernel = self.compiled_kernels[kernel_name]
        
        # Get device properties
        device_props = cp.cuda.runtime.getDeviceProperties(self.device_id)
        max_threads_per_block = device_props['maxThreadsPerBlock']
        
        # Calculate optimal block size (power of 2, up to max)
        block_size = min(max_threads_per_block, 2 ** int(math.log2(problem_size) + 1))
        block_size = max(32, min(1024, block_size))  # Clamp to reasonable range
        
        # Calculate grid size
        grid_size = (problem_size + block_size - 1) // block_size
        
        return block_size, grid_size

# ===============================================================================
# High-Level GPU Operations
# ===============================================================================

class GPUStockOperations:
    """GPU-accelerated stock operations"""
    
    def __init__(self, kernel_manager: GPUKernelManager, memory_pool: GPUMemoryPool):
        self.kernel_manager = kernel_manager
        self.memory_pool = memory_pool
        self.device_id = kernel_manager.device_id
    
    def update_stocks_gpu(self, stock_values: np.ndarray, 
                         net_flows: np.ndarray,
                         dt: float,
                         min_values: Optional[np.ndarray] = None,
                         max_values: Optional[np.ndarray] = None) -> np.ndarray:
        """GPU-accelerated stock update with bounds checking"""
        
        if not GPU_AVAILABLE or 'stock_update' not in self.kernel_manager.compiled_kernels:
            raise RuntimeError("GPU stock update not available")
        
        n_stocks = len(stock_values)
        
        # Handle bounds
        if min_values is None:
            min_values = np.zeros(n_stocks)
        if max_values is None:
            max_values = np.full(n_stocks, np.inf)
        
        with cp.cuda.Device(self.device_id):
            # Allocate GPU memory
            gpu_stocks = self.memory_pool.get_array(stock_values.shape, stock_values.dtype)
            gpu_flows = self.memory_pool.get_array(net_flows.shape, net_flows.dtype)
            gpu_result = self.memory_pool.get_array(stock_values.shape, stock_values.dtype)
            gpu_min = self.memory_pool.get_array(min_values.shape, min_values.dtype)
            gpu_max = self.memory_pool.get_array(max_values.shape, max_values.dtype)
            
            # Copy data to GPU
            gpu_stocks[:] = cp.asarray(stock_values)
            gpu_flows[:] = cp.asarray(net_flows)
            gpu_min[:] = cp.asarray(min_values)
            gpu_max[:] = cp.asarray(max_values)
            
            # Calculate kernel launch parameters
            block_size, grid_size = self.kernel_manager.get_optimal_block_size('stock_update', n_stocks)
            
            # Launch kernel
            kernel = self.kernel_manager.compiled_kernels['stock_update']
            kernel(
                (grid_size,), (block_size,),
                (gpu_stocks, gpu_flows, gpu_result, gpu_min, gpu_max, dt, n_stocks)
            )
            
            # Copy result back to CPU
            result = cp.asnumpy(gpu_result)
            
            # Return GPU memory to pool
            self.memory_pool.return_array(gpu_stocks)
            self.memory_pool.return_array(gpu_flows)
            self.memory_pool.return_array(gpu_result)
            self.memory_pool.return_array(gpu_min)
            self.memory_pool.return_array(gpu_max)
            
            return result
    
    def compute_flows_gpu(self, from_stocks: np.ndarray,
                         to_stocks: np.ndarray,
                         flow_rates: np.ndarray,
                         flow_connections: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated flow computation"""
        
        if not GPU_AVAILABLE or 'flow_computation' not in self.kernel_manager.compiled_kernels:
            raise RuntimeError("GPU flow computation not available")
        
        n_stocks = len(from_stocks)
        n_flows = len(flow_rates)
        
        with cp.cuda.Device(self.device_id):
            # Allocate GPU memory
            gpu_from = self.memory_pool.get_array(from_stocks.shape, from_stocks.dtype)
            gpu_to = self.memory_pool.get_array(to_stocks.shape, to_stocks.dtype)
            gpu_rates = self.memory_pool.get_array(flow_rates.shape, flow_rates.dtype)
            gpu_connections = self.memory_pool.get_array(flow_connections.shape, flow_connections.dtype)
            gpu_inflows = self.memory_pool.get_array((n_stocks,), np.float64)
            gpu_outflows = self.memory_pool.get_array((n_stocks,), np.float64)
            
            # Copy data to GPU
            gpu_from[:] = cp.asarray(from_stocks)
            gpu_to[:] = cp.asarray(to_stocks)
            gpu_rates[:] = cp.asarray(flow_rates)
            gpu_connections[:] = cp.asarray(flow_connections.astype(np.int32))
            gpu_inflows.fill(0)
            gpu_outflows.fill(0)
            
            # Launch kernel
            block_size, grid_size = self.kernel_manager.get_optimal_block_size('flow_computation', n_flows)
            kernel = self.kernel_manager.compiled_kernels['flow_computation']
            
            kernel(
                (grid_size,), (block_size,),
                (gpu_from, gpu_to, gpu_rates, gpu_connections, 
                 gpu_inflows, gpu_outflows, n_stocks, n_flows)
            )
            
            # Copy results back
            inflows = cp.asnumpy(gpu_inflows)
            outflows = cp.asnumpy(gpu_outflows)
            
            # Return memory to pool
            for gpu_array in [gpu_from, gpu_to, gpu_rates, gpu_connections, gpu_inflows, gpu_outflows]:
                self.memory_pool.return_array(gpu_array)
            
            return inflows, outflows

class GPUIntegrationMethods:
    """GPU-accelerated integration methods"""
    
    def __init__(self, kernel_manager: GPUKernelManager, memory_pool: GPUMemoryPool):
        self.kernel_manager = kernel_manager
        self.memory_pool = memory_pool
        self.device_id = kernel_manager.device_id
    
    def rk4_step_gpu(self, y: np.ndarray, k1: np.ndarray, k2: np.ndarray, 
                    k3: np.ndarray, k4: np.ndarray, dt: float) -> np.ndarray:
        """GPU-accelerated RK4 integration step"""
        
        if not GPU_AVAILABLE or 'rk4_step' not in self.kernel_manager.compiled_kernels:
            raise RuntimeError("GPU RK4 not available")
        
        n_vars = len(y)
        
        with cp.cuda.Device(self.device_id):
            # Allocate GPU memory
            gpu_arrays = []
            for array in [y, k1, k2, k3, k4]:
                gpu_array = self.memory_pool.get_array(array.shape, array.dtype)
                gpu_array[:] = cp.asarray(array)
                gpu_arrays.append(gpu_array)
            
            gpu_result = self.memory_pool.get_array(y.shape, y.dtype)
            
            # Launch kernel
            block_size, grid_size = self.kernel_manager.get_optimal_block_size('rk4_step', n_vars)
            kernel = self.kernel_manager.compiled_kernels['rk4_step']
            
            kernel(
                (grid_size,), (block_size,),
                (*gpu_arrays, gpu_result, dt, n_vars)
            )
            
            # Copy result back
            result = cp.asnumpy(gpu_result)
            
            # Return memory to pool
            for gpu_array in gpu_arrays + [gpu_result]:
                self.memory_pool.return_array(gpu_array)
            
            return result

# ===============================================================================
# Automatic CPU/GPU Selection Engine
# ===============================================================================

class PerformanceSelector:
    """Intelligent CPU/GPU selection based on problem characteristics"""
    
    def __init__(self, performance_system: 'PerformanceSystem'):
        self.perf_system = performance_system
        self.execution_history = {}
        self.performance_thresholds = {
            'gpu_min_size': 100,        # Minimum problem size for GPU
            'gpu_memory_safety': 0.8,   # Use 80% of available GPU memory
            'cpu_parallel_threshold': 50, # Minimum size for CPU parallelization
            'gpu_speedup_threshold': 2.0  # Minimum speedup to prefer GPU
        }
        
        # Adaptive thresholds (learned from execution history)
        self.adaptive_thresholds = self.performance_thresholds.copy()
    
    def select_execution_strategy(self, operation: str, 
                                problem_size: int,
                                memory_requirement: int,
                                data_types: List[type] = None) -> Dict[str, Any]:
        """Select optimal execution strategy based on problem characteristics"""
        
        strategy = {
            'execution_mode': 'cpu_serial',
            'device_id': None,
            'expected_speedup': 1.0,
            'memory_estimate': memory_requirement,
            'reason': 'default',
            'fallback_mode': 'cpu_serial'
        }
        
        # Check GPU availability and suitability
        if (self.perf_system.capabilities.has_cuda and 
            problem_size >= self.adaptive_thresholds['gpu_min_size']):
            
            gpu_strategy = self._evaluate_gpu_strategy(operation, problem_size, memory_requirement)
            if gpu_strategy['suitable']:
                strategy.update({
                    'execution_mode': 'gpu',
                    'device_id': gpu_strategy['device_id'],
                    'expected_speedup': gpu_strategy['expected_speedup'],
                    'reason': gpu_strategy['reason'],
                    'fallback_mode': 'cpu_parallel' if self.perf_system.capabilities.cpu_threads >= 4 else 'cpu_serial'
                })
                return strategy
        
        # CPU strategy selection
        if (problem_size >= self.adaptive_thresholds['cpu_parallel_threshold'] and
            self.perf_system.capabilities.cpu_threads >= 4):
            
            strategy.update({
                'execution_mode': 'cpu_parallel',
                'expected_speedup': min(self.perf_system.capabilities.cpu_threads, 
                                      problem_size / 50),
                'reason': 'cpu_parallel_optimal'
            })
        
        return strategy
    
    def _evaluate_gpu_strategy(self, operation: str, problem_size: int, 
                              memory_requirement: int) -> Dict[str, Any]:
        """Evaluate GPU execution strategy"""
        
        gpu_eval = {
            'suitable': False,
            'device_id': 0,
            'expected_speedup': 1.0,
            'reason': 'gpu_not_suitable'
        }
        
        # Find suitable GPU
        for gpu in self.perf_system.capabilities.gpus:
            available_memory = gpu.memory_free
            required_memory = memory_requirement * self.performance_thresholds['gpu_memory_safety']
            
            if available_memory > required_memory:
                # Estimate speedup based on operation type and problem size
                speedup = self._estimate_gpu_speedup(operation, problem_size, gpu)
                
                if speedup >= self.adaptive_thresholds['gpu_speedup_threshold']:
                    gpu_eval.update({
                        'suitable': True,
                        'device_id': gpu.device_id,
                        'expected_speedup': speedup,
                        'reason': f'gpu_speedup_{speedup:.1f}x'
                    })
                    break
            else:
                gpu_eval['reason'] = 'gpu_insufficient_memory'
        
        return gpu_eval
    
    def _estimate_gpu_speedup(self, operation: str, problem_size: int, gpu) -> float:
        """Estimate GPU speedup for specific operation"""
        
        # Base speedup estimates for different operations
        base_speedups = {
            'stock_update': 50,
            'flow_computation': 30,
            'matrix_multiplication': 100,
            'integration_step': 20,
            'multidim_flow': 80
        }
        
        base_speedup = base_speedups.get(operation, 10)
        
        # Scale by problem size (larger problems benefit more from GPU)
        size_factor = min(5.0, math.log10(problem_size))
        
        # Scale by GPU compute capability
        compute_factor = gpu.compute_capability[0] / 6.0  # Normalize to compute 6.0
        
        # Scale by multiprocessor count
        mp_factor = min(2.0, gpu.multiprocessor_count / 28)  # Normalize to RTX 3080
        
        estimated_speedup = base_speedup * size_factor * compute_factor * mp_factor
        
        # Apply historical learning if available
        if operation in self.execution_history:
            historical_speedups = [h['actual_speedup'] for h in self.execution_history[operation] 
                                 if h['problem_size'] > problem_size * 0.5]
            if historical_speedups:
                avg_historical = np.mean(historical_speedups)
                estimated_speedup = (estimated_speedup + avg_historical) / 2
        
        return min(1000, max(1.0, estimated_speedup))  # Clamp to reasonable range
    
    def record_execution_result(self, operation: str, strategy: Dict[str, Any], 
                               actual_runtime: float, baseline_runtime: float):
        """Record execution results for adaptive learning"""
        
        actual_speedup = baseline_runtime / actual_runtime if actual_runtime > 0 else 1.0
        
        execution_record = {
            'timestamp': time.time(),
            'operation': operation,
            'strategy': strategy,
            'actual_speedup': actual_speedup,
            'expected_speedup': strategy['expected_speedup'],
            'speedup_error': abs(actual_speedup - strategy['expected_speedup']),
            'problem_size': strategy.get('problem_size', 0)
        }
        
        if operation not in self.execution_history:
            self.execution_history[operation] = []
        
        self.execution_history[operation].append(execution_record)
        
        # Keep only recent history (last 100 executions per operation)
        if len(self.execution_history[operation]) > 100:
            self.execution_history[operation] = self.execution_history[operation][-100:]
        
        # Update adaptive thresholds
        self._update_adaptive_thresholds()
    
    def _update_adaptive_thresholds(self):
        """Update adaptive thresholds based on execution history"""
        
        # Analyze recent performance trends
        for operation, history in self.execution_history.items():
            if len(history) < 10:  # Need sufficient data
                continue
            
            recent_history = history[-20:]  # Last 20 executions
            
            # Calculate average speedup error for GPU executions
            gpu_executions = [h for h in recent_history if h['strategy']['execution_mode'] == 'gpu']
            
            if gpu_executions:
                avg_speedup_error = np.mean([h['speedup_error'] for h in gpu_executions])
                
                # If GPU consistently underperforms, increase threshold
                if avg_speedup_error > 5.0:  # Large prediction error
                    self.adaptive_thresholds['gpu_speedup_threshold'] *= 1.1
                    self.adaptive_thresholds['gpu_min_size'] = int(self.adaptive_thresholds['gpu_min_size'] * 1.2)
                
                # If GPU consistently overperforms, decrease threshold
                elif avg_speedup_error < 1.0:  # Small prediction error
                    self.adaptive_thresholds['gpu_speedup_threshold'] *= 0.95
                    self.adaptive_thresholds['gpu_min_size'] = int(self.adaptive_thresholds['gpu_min_size'] * 0.9)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance analysis summary"""
        
        summary = {
            'total_operations': sum(len(history) for history in self.execution_history.values()),
            'operations_tracked': list(self.execution_history.keys()),
            'adaptive_thresholds': self.adaptive_thresholds,
            'original_thresholds': self.performance_thresholds,
            'operation_stats': {}
        }
        
        for operation, history in self.execution_history.items():
            if history:
                speedups = [h['actual_speedup'] for h in history]
                gpu_executions = [h for h in history if h['strategy']['execution_mode'] == 'gpu']
                
                summary['operation_stats'][operation] = {
                    'total_executions': len(history),
                    'gpu_executions': len(gpu_executions),
                    'avg_speedup': np.mean(speedups),
                    'max_speedup': np.max(speedups),
                    'gpu_success_rate': len(gpu_executions) / len(history) if history else 0
                }
        
        return summary

# ===============================================================================
# Unified High-Performance Execution Engine
# ===============================================================================

class UnifiedPerformanceEngine:
    """Complete high-performance execution engine with automatic optimization"""
    
    def __init__(self):
        self.performance_system = None
        self.kernel_manager = None
        self.gpu_operations = None
        self.gpu_integration = None
        self.selector = None
        self.initialized = False
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize complete performance engine"""
        
        print("🚀 Initializing Unified Performance Engine...")
        
        # Initialize performance system
        self.performance_system = PerformanceSystem()
        init_result = self.performance_system.initialize()
        
        # Initialize GPU components if available
        if init_result['gpu_available']:
            try:
                device_id = 0  # Use first GPU
                memory_pool = self.performance_system.execution_context.memory_pools[device_id]
                
                self.kernel_manager = GPUKernelManager(device_id)
                self.gpu_operations = GPUStockOperations(self.kernel_manager, memory_pool)
                self.gpu_integration = GPUIntegrationMethods(self.kernel_manager, memory_pool)
                
                print("✅ GPU acceleration initialized")
                
            except Exception as e:
                print(f"⚠️ GPU initialization failed: {e}")
                init_result['gpu_available'] = False
        
        # Initialize performance selector
        self.selector = PerformanceSelector(self.performance_system)
        
        self.initialized = True
        
        print("✅ Unified Performance Engine ready")
        
        return {
            **init_result,
            'gpu_kernels_compiled': len(self.kernel_manager.compiled_kernels) if self.kernel_manager else 0,
            'performance_selector': True
        }
    
    def execute_stock_update(self, stock_values: np.ndarray, net_flows: np.ndarray, 
                           dt: float, **kwargs) -> np.ndarray:
        """Execute stock update with automatic optimization"""
        
        if not self.initialized:
            raise RuntimeError("Performance engine not initialized")
        
        problem_size = len(stock_values)
        memory_req = stock_values.nbytes * 4  # Estimate memory requirement
        
        # Select execution strategy
        strategy = self.selector.select_execution_strategy(
            'stock_update', problem_size, memory_req
        )
        
        start_time = time.time()
        
        try:
            if strategy['execution_mode'] == 'gpu' and self.gpu_operations:
                result = self.gpu_operations.update_stocks_gpu(stock_values, net_flows, dt, **kwargs)
            elif strategy['execution_mode'] == 'cpu_parallel' and CPU_KERNELS_AVAILABLE:
                # Use CPU parallel kernels
                min_vals = kwargs.get('min_values', np.zeros_like(stock_values))
                max_vals = kwargs.get('max_values', np.full_like(stock_values, np.inf))
                result = vectorized_stock_update(stock_values, net_flows, dt, min_vals, max_vals)
            else:
                # Fallback to basic NumPy
                result = stock_values + net_flows * dt
                if 'min_values' in kwargs:
                    result = np.maximum(result, kwargs['min_values'])
                if 'max_values' in kwargs:
                    result = np.minimum(result, kwargs['max_values'])
            
            execution_time = time.time() - start_time
            
            # Record performance for learning
            baseline_time = self._estimate_baseline_time('stock_update', problem_size)
            self.selector.record_execution_result('stock_update', strategy, execution_time, baseline_time)
            
            return result
            
        except Exception as e:
            # Fallback execution
            warnings.warn(f"Optimized execution failed, using fallback: {e}")
            return stock_values + net_flows * dt
    
    def execute_integration_step(self, y: np.ndarray, derivatives_func: Callable, 
                                dt: float, method: str = 'rk4') -> np.ndarray:
        """Execute integration step with automatic optimization"""
        
        if not self.initialized:
            raise RuntimeError("Performance engine not initialized")
        
        problem_size = len(y)
        memory_req = y.nbytes * 8  # Estimate for RK4 workspace
        
        strategy = self.selector.select_execution_strategy(
            'integration_step', problem_size, memory_req
        )
        
        start_time = time.time()
        
        try:
            if method == 'rk4':
                # Compute RK4 slopes
                k1 = derivatives_func(y)
                k2 = derivatives_func(y + k1 * dt / 2)
                k3 = derivatives_func(y + k2 * dt / 2)
                k4 = derivatives_func(y + k3 * dt)
                
                if strategy['execution_mode'] == 'gpu' and self.gpu_integration:
                    result = self.gpu_integration.rk4_step_gpu(y, k1, k2, k3, k4, dt)
                elif CPU_KERNELS_AVAILABLE:
                    result = vectorized_rk4_step(y, k1, k2, k3, k4, dt)
                else:
                    result = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            else:
                # Euler step
                dy = derivatives_func(y)
                result = y + dy * dt
            
            execution_time = time.time() - start_time
            baseline_time = self._estimate_baseline_time('integration_step', problem_size)
            self.selector.record_execution_result('integration_step', strategy, execution_time, baseline_time)
            
            return result
            
        except Exception as e:
            warnings.warn(f"Optimized integration failed, using fallback: {e}")
            dy = derivatives_func(y)
            return y + dy * dt
    
    def _estimate_baseline_time(self, operation: str, problem_size: int) -> float:
        """Estimate baseline execution time for performance comparison"""
        
        # Simple heuristic estimates (in seconds)
        base_times = {
            'stock_update': 1e-7,      # ~100ns per element
            'integration_step': 5e-7,   # ~500ns per element
            'flow_computation': 2e-7,   # ~200ns per element
        }
        
        base_time = base_times.get(operation, 1e-7)
        return base_time * problem_size
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        if not self.initialized:
            return {'error': 'Engine not initialized'}
        
        report = {
            'system_info': self.performance_system.get_system_info(),
            'performance_selector': self.selector.get_performance_summary(),
            'gpu_status': {},
            'execution_stats': {}
        }
        
        # Add GPU-specific information
        if self.kernel_manager:
            report['gpu_status'] = {
                'compiled_kernels': list(self.kernel_manager.compiled_kernels.keys()),
                'compilation_times': self.kernel_manager.compilation_times,
                'memory_pools': self.performance_system.execution_context.get_execution_stats()['memory_pools']
            }
        
        return report

# ===============================================================================
# Testing and Benchmarking
# ===============================================================================

def test_gpu_kernels():
    """Comprehensive test of GPU kernels and automatic selection"""
    
    print("🧪 Testing GPU Kernels & Automatic Selection")
    print("=" * 50)
    
    # Initialize performance engine
    engine = UnifiedPerformanceEngine()
    init_result = engine.initialize()
    
    print(f"\n📊 Initialization Results:")
    for key, value in init_result.items():
        print(f"   {key}: {value}")
    
    # Test different problem sizes
    test_sizes = [100, 1000, 5000, 10000, 50000]
    
    for size in test_sizes:
        print(f"\n🔬 Testing with {size} stocks...")
        
        # Create test data
        stock_values = np.random.rand(size) * 1000
        net_flows = np.random.rand(size) * 50 - 25
        dt = 0.1
        
        try:
            # Test stock update
            start_time = time.time()
            result = engine.execute_stock_update(stock_values, net_flows, dt)
            execution_time = time.time() - start_time
            
            print(f"   ✅ Stock update: {execution_time*1000:.2f}ms")
            print(f"   📈 Throughput: {size/execution_time:.0f} stocks/second")
            
            # Verify correctness
            expected = stock_values + net_flows * dt
            if np.allclose(result, expected, rtol=1e-10):
                print(f"   ✅ Results verified")
            else:
                print(f"   ⚠️ Results differ from expected")
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
    
    # Get final performance report
    print(f"\n📈 Performance Report:")
    report = engine.get_performance_report()
    
    if 'performance_selector' in report:
        selector_stats = report['performance_selector']
        print(f"   Total operations: {selector_stats['total_operations']}")
        print(f"   Operations tracked: {selector_stats['operations_tracked']}")
        
        for op, stats in selector_stats.get('operation_stats', {}).items():
            print(f"   {op}: {stats['total_executions']} executions, "
                  f"{stats['avg_speedup']:.1f}x avg speedup")
    
    return engine

if __name__ == "__main__":
    test_gpu_kernels()