# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 01:40:26 2025

@author: NagabhushanamTattaga
"""

# final_optimization.py - Phase 4: Final Optimization & Multi-GPU Support

import numpy as np
import time
import threading
import asyncio
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import json

# GPU Libraries
try:
    import cupy as cp
    from cupy import cuda
    import cupyx
    from cupyx.scipy import ndimage, sparse
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

# Previous phase imports
try:
    from gpu_kernels import UnifiedPerformanceEngine, GPUKernelManager, PerformanceSelector
    from gpu_infrastructure import PerformanceSystem, GPUMemoryPool
    from performance_core import PerformanceConfig
    PREVIOUS_PHASES_AVAILABLE = True
except ImportError:
    PREVIOUS_PHASES_AVAILABLE = False
    print("⚠️ Previous phases not available - ensure Phases 1-3 are implemented")

# ===============================================================================
# Memory Transfer Optimization
# ===============================================================================

class OptimizedMemoryTransfer:
    """Optimized GPU-CPU memory transfers with streaming and prefetching"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.streams = []
        self.pinned_memory_pool = {}
        self.transfer_stats = {
            'h2d_transfers': 0,
            'd2h_transfers': 0,
            'h2d_bandwidth': 0.0,
            'd2h_bandwidth': 0.0,
            'async_transfers': 0
        }
        
        if GPU_AVAILABLE:
            self._initialize_streams()
    
    def _initialize_streams(self):
        """Initialize CUDA streams for asynchronous transfers"""
        try:
            with cp.cuda.Device(self.device_id):
                # Create multiple streams for overlapped computation and transfer
                self.streams = [cp.cuda.Stream() for _ in range(4)]
                print(f"✅ Initialized {len(self.streams)} CUDA streams for device {self.device_id}")
        except Exception as e:
            warnings.warn(f"Failed to initialize CUDA streams: {e}")
    
    def allocate_pinned_memory(self, shape: Tuple[int, ...], dtype=np.float64) -> np.ndarray:
        """Allocate pinned (page-locked) memory for faster transfers"""
        
        if not GPU_AVAILABLE:
            return np.empty(shape, dtype=dtype)
        
        key = (shape, dtype)
        if key in self.pinned_memory_pool:
            return self.pinned_memory_pool[key]
        
        try:
            # Allocate pinned memory using CuPy
            with cp.cuda.Device(self.device_id):
                pinned_memory = cp.cuda.alloc_pinned_memory(np.prod(shape) * np.dtype(dtype).itemsize)
                array = np.frombuffer(pinned_memory, dtype=dtype).reshape(shape)
                self.pinned_memory_pool[key] = array
                return array
        except Exception as e:
            warnings.warn(f"Pinned memory allocation failed: {e}")
            return np.empty(shape, dtype=dtype)
    
    def async_h2d_transfer(self, host_array: np.ndarray, 
                          stream_id: int = 0) -> 'cp.ndarray':
        """Asynchronous host-to-device transfer"""
        
        if not GPU_AVAILABLE or not self.streams:
            return cp.asarray(host_array)
        
        stream = self.streams[stream_id % len(self.streams)]
        
        with cp.cuda.Device(self.device_id):
            start_time = time.time()
            
            # Allocate device memory
            device_array = cp.empty(host_array.shape, dtype=host_array.dtype)
            
            # Asynchronous copy
            device_array.set(host_array, stream=stream)
            
            # Update statistics
            transfer_time = time.time() - start_time
            if transfer_time > 0:
                bandwidth = host_array.nbytes / transfer_time / 1e9  # GB/s
                self.transfer_stats['h2d_bandwidth'] = (
                    self.transfer_stats['h2d_bandwidth'] + bandwidth) / 2
            
            self.transfer_stats['h2d_transfers'] += 1
            self.transfer_stats['async_transfers'] += 1
            
            return device_array
    
    def async_d2h_transfer(self, device_array: 'cp.ndarray', 
                          stream_id: int = 0) -> np.ndarray:
        """Asynchronous device-to-host transfer"""
        
        if not GPU_AVAILABLE or not hasattr(device_array, 'device'):
            return np.asarray(device_array)
        
        stream = self.streams[stream_id % len(self.streams)]
        
        start_time = time.time()
        
        # Use pinned memory for faster transfer
        pinned_result = self.allocate_pinned_memory(device_array.shape, device_array.dtype)
        
        # Asynchronous copy
        device_array.get(out=pinned_result, stream=stream)
        
        # Wait for transfer completion
        stream.synchronize()
        
        # Update statistics
        transfer_time = time.time() - start_time
        if transfer_time > 0:
            bandwidth = device_array.nbytes / transfer_time / 1e9  # GB/s
            self.transfer_stats['d2h_bandwidth'] = (
                self.transfer_stats['d2h_bandwidth'] + bandwidth) / 2
        
        self.transfer_stats['d2h_transfers'] += 1
        
        return pinned_result.copy()  # Return a copy to avoid pinned memory issues
    
    def batch_transfer_h2d(self, host_arrays: List[np.ndarray]) -> List['cp.ndarray']:
        """Batch multiple host-to-device transfers for efficiency"""
        
        if not GPU_AVAILABLE:
            return [cp.asarray(arr) for arr in host_arrays]
        
        device_arrays = []
        
        with cp.cuda.Device(self.device_id):
            for i, host_array in enumerate(host_arrays):
                stream_id = i % len(self.streams)
                device_array = self.async_h2d_transfer(host_array, stream_id)
                device_arrays.append(device_array)
            
            # Synchronize all streams
            for stream in self.streams:
                stream.synchronize()
        
        return device_arrays
    
    def get_transfer_stats(self) -> Dict[str, Any]:
        """Get memory transfer statistics"""
        return {
            **self.transfer_stats,
            'pinned_memory_allocations': len(self.pinned_memory_pool),
            'active_streams': len(self.streams)
        }

# ===============================================================================
# Multi-GPU Distributed Computing
# ===============================================================================

class MultiGPUManager:
    """Manages distributed computing across multiple GPUs"""
    
    def __init__(self, device_ids: List[int] = None):
        self.device_ids = device_ids or []
        self.gpu_pools = {}
        self.memory_transfers = {}
        self.load_balancer = None
        self.synchronization_events = {}
        
        if GPU_AVAILABLE:
            self._detect_and_initialize_gpus()
    
    def _detect_and_initialize_gpus(self):
        """Detect and initialize all available GPUs"""
        
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            
            if not self.device_ids:
                self.device_ids = list(range(device_count))
            
            print(f"🔧 Initializing Multi-GPU manager for {len(self.device_ids)} devices...")
            
            for device_id in self.device_ids:
                if device_id < device_count:
                    # Initialize memory pool and transfer manager for each GPU
                    self.gpu_pools[device_id] = GPUMemoryPool(device_id)
                    self.memory_transfers[device_id] = OptimizedMemoryTransfer(device_id)
                    
                    # Create synchronization events
                    with cp.cuda.Device(device_id):
                        self.synchronization_events[device_id] = cp.cuda.Event()
                    
                    print(f"   ✅ GPU {device_id} initialized")
                else:
                    print(f"   ⚠️ GPU {device_id} not available (only {device_count} devices)")
            
            self.load_balancer = GPULoadBalancer(list(self.gpu_pools.keys()))
            
        except Exception as e:
            warnings.warn(f"Multi-GPU initialization failed: {e}")
            self.device_ids = []
    
    def distribute_computation(self, operation: str, data: np.ndarray, 
                             computation_func: Callable, **kwargs) -> np.ndarray:
        """Distribute computation across multiple GPUs"""
        
        if len(self.device_ids) <= 1:
            # Single GPU or CPU fallback
            return computation_func(data, **kwargs)
        
        # Determine optimal data partitioning
        partition_strategy = self.load_balancer.get_partition_strategy(
            len(data), operation, self.device_ids
        )
        
        # Split data according to strategy
        data_chunks = self._partition_data(data, partition_strategy)
        
        # Distribute computation
        results = []
        futures = []
        
        with ThreadPoolExecutor(max_workers=len(self.device_ids)) as executor:
            for device_id, chunk in zip(self.device_ids, data_chunks):
                if len(chunk) > 0:
                    future = executor.submit(
                        self._execute_on_device, 
                        device_id, computation_func, chunk, **kwargs
                    )
                    futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    warnings.warn(f"Multi-GPU computation failed on device: {e}")
        
        # Combine results
        return self._combine_results(results, partition_strategy)
    
    def _partition_data(self, data: np.ndarray, 
                       strategy: Dict[str, Any]) -> List[np.ndarray]:
        """Partition data for multi-GPU processing"""
        
        partition_sizes = strategy['partition_sizes']
        chunks = []
        
        start_idx = 0
        for size in partition_sizes:
            end_idx = start_idx + size
            chunk = data[start_idx:end_idx]
            chunks.append(chunk)
            start_idx = end_idx
        
        return chunks
    
    def _execute_on_device(self, device_id: int, computation_func: Callable, 
                          data_chunk: np.ndarray, **kwargs) -> np.ndarray:
        """Execute computation on specific GPU device"""
        
        with cp.cuda.Device(device_id):
            # Transfer data to GPU
            gpu_data = self.memory_transfers[device_id].async_h2d_transfer(data_chunk)
            
            # Execute computation
            gpu_result = computation_func(gpu_data, **kwargs)
            
            # Transfer result back to CPU
            cpu_result = self.memory_transfers[device_id].async_d2h_transfer(gpu_result)
            
            return cpu_result
    
    def _combine_results(self, results: List[np.ndarray], 
                        strategy: Dict[str, Any]) -> np.ndarray:
        """Combine results from multiple GPUs"""
        
        if strategy['combination_method'] == 'concatenate':
            return np.concatenate(results, axis=0)
        elif strategy['combination_method'] == 'sum':
            return np.sum(results, axis=0)
        elif strategy['combination_method'] == 'mean':
            return np.mean(results, axis=0)
        else:
            return np.concatenate(results, axis=0)  # Default
    
    def synchronize_all_devices(self):
        """Synchronize all GPU devices"""
        for device_id in self.device_ids:
            with cp.cuda.Device(device_id):
                cp.cuda.Stream.null.synchronize()
    
    def get_multi_gpu_stats(self) -> Dict[str, Any]:
        """Get multi-GPU statistics"""
        
        stats = {
            'device_count': len(self.device_ids),
            'active_devices': self.device_ids,
            'memory_pools': {},
            'transfer_stats': {},
            'load_balancer': {}
        }
        
        for device_id in self.device_ids:
            if device_id in self.gpu_pools:
                stats['memory_pools'][device_id] = self.gpu_pools[device_id].get_memory_info()
            
            if device_id in self.memory_transfers:
                stats['transfer_stats'][device_id] = self.memory_transfers[device_id].get_transfer_stats()
        
        if self.load_balancer:
            stats['load_balancer'] = self.load_balancer.get_balancing_stats()
        
        return stats

class GPULoadBalancer:
    """Intelligent load balancing across multiple GPUs"""
    
    def __init__(self, device_ids: List[int]):
        self.device_ids = device_ids
        self.device_capabilities = {}
        self.load_history = {device_id: [] for device_id in device_ids}
        self.balancing_stats = {
            'total_partitions': 0,
            'avg_load_balance': 0.0,
            'device_utilization': {device_id: 0.0 for device_id in device_ids}
        }
        
        self._analyze_device_capabilities()
    
    def _analyze_device_capabilities(self):
        """Analyze relative capabilities of each GPU device"""
        
        for device_id in self.device_ids:
            try:
                with cp.cuda.Device(device_id):
                    props = cp.cuda.runtime.getDeviceProperties(device_id)
                    meminfo = cp.cuda.runtime.memGetInfo()
                    
                    self.device_capabilities[device_id] = {
                        'memory_total': meminfo[1],
                        'memory_free': meminfo[0],
                        'multiprocessor_count': props['multiProcessorCount'],
                        'max_threads_per_block': props['maxThreadsPerBlock'],
                        'compute_capability': (props['major'], props['minor']),
                        'relative_performance': self._estimate_relative_performance(props)
                    }
            except Exception as e:
                warnings.warn(f"Failed to analyze device {device_id}: {e}")
                self.device_capabilities[device_id] = {'relative_performance': 1.0}
    
    def _estimate_relative_performance(self, props: Dict[str, Any]) -> float:
        """Estimate relative performance of GPU device"""
        
        # Simple heuristic based on compute capability and SM count
        compute_score = props['major'] * 10 + props['minor']
        sm_count = props['multiProcessorCount']
        
        # Normalize to a baseline (e.g., GTX 1080 = 1.0)
        baseline_compute = 61  # Compute 6.1
        baseline_sm = 20
        
        relative_performance = (compute_score / baseline_compute) * (sm_count / baseline_sm)
        
        return max(0.1, min(10.0, relative_performance))  # Clamp to reasonable range
    
    def get_partition_strategy(self, total_size: int, operation: str, 
                             device_ids: List[int]) -> Dict[str, Any]:
        """Determine optimal data partitioning strategy"""
        
        # Calculate relative workload distribution
        total_performance = sum(
            self.device_capabilities[device_id]['relative_performance'] 
            for device_id in device_ids
        )
        
        partition_sizes = []
        for device_id in device_ids:
            device_performance = self.device_capabilities[device_id]['relative_performance']
            partition_ratio = device_performance / total_performance
            partition_size = int(total_size * partition_ratio)
            partition_sizes.append(partition_size)
        
        # Adjust for any rounding errors
        total_assigned = sum(partition_sizes)
        if total_assigned != total_size:
            partition_sizes[-1] += total_size - total_assigned
        
        self.balancing_stats['total_partitions'] += 1
        
        strategy = {
            'partition_sizes': partition_sizes,
            'device_assignments': device_ids,
            'combination_method': self._get_combination_method(operation),
            'load_balance_ratio': max(partition_sizes) / max(1, min(partition_sizes))
        }
        
        return strategy
    
    def _get_combination_method(self, operation: str) -> str:
        """Determine how to combine results from multiple GPUs"""
        
        if operation in ['stock_update', 'integration_step']:
            return 'concatenate'
        elif operation in ['reduction', 'sum']:
            return 'sum'
        elif operation in ['average', 'mean']:
            return 'mean'
        else:
            return 'concatenate'  # Default
    
    def record_execution_time(self, device_id: int, execution_time: float):
        """Record execution time for load balancing optimization"""
        
        self.load_history[device_id].append(execution_time)
        
        # Keep only recent history
        if len(self.load_history[device_id]) > 100:
            self.load_history[device_id] = self.load_history[device_id][-100:]
        
        # Update utilization statistics
        if self.load_history[device_id]:
            avg_time = np.mean(self.load_history[device_id])
            total_avg = np.mean([
                np.mean(times) for times in self.load_history.values() if times
            ])
            
            if total_avg > 0:
                self.balancing_stats['device_utilization'][device_id] = avg_time / total_avg
    
    def get_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics"""
        return self.balancing_stats.copy()

# ===============================================================================
# Adaptive Threshold Optimization
# ===============================================================================

class AdaptiveThresholdOptimizer:
    """Automatically optimizes performance thresholds based on execution history"""
    
    def __init__(self):
        self.threshold_history = {}
        self.performance_metrics = {}
        self.optimization_parameters = {
            'learning_rate': 0.1,
            'stability_threshold': 0.05,
            'min_samples': 20,
            'confidence_interval': 0.95
        }
        
        # Initial thresholds (will be optimized)
        self.current_thresholds = {
            'gpu_min_size': 100,
            'cpu_parallel_threshold': 50,
            'gpu_memory_safety_margin': 0.8,
            'multi_gpu_threshold': 10000,
            'async_transfer_threshold': 1000
        }
    
    def record_performance_sample(self, threshold_name: str, threshold_value: float,
                                 problem_size: int, execution_time: float, 
                                 expected_speedup: float, actual_speedup: float):
        """Record performance sample for threshold optimization"""
        
        sample = {
            'timestamp': time.time(),
            'threshold_value': threshold_value,
            'problem_size': problem_size,
            'execution_time': execution_time,
            'expected_speedup': expected_speedup,
            'actual_speedup': actual_speedup,
            'speedup_error': abs(actual_speedup - expected_speedup),
            'efficiency': actual_speedup / max(1.0, expected_speedup)
        }
        
        if threshold_name not in self.threshold_history:
            self.threshold_history[threshold_name] = []
        
        self.threshold_history[threshold_name].append(sample)
        
        # Keep only recent samples
        if len(self.threshold_history[threshold_name]) > 1000:
            self.threshold_history[threshold_name] = self.threshold_history[threshold_name][-1000:]
    
    def optimize_threshold(self, threshold_name: str) -> Optional[float]:
        """Optimize specific threshold using collected performance data"""
        
        if (threshold_name not in self.threshold_history or 
            len(self.threshold_history[threshold_name]) < self.optimization_parameters['min_samples']):
            return None
        
        samples = self.threshold_history[threshold_name]
        recent_samples = samples[-100:]  # Use recent samples for optimization
        
        # Analyze performance trends
        current_threshold = self.current_thresholds[threshold_name]
        
        # Find optimal threshold value
        threshold_values = [s['threshold_value'] for s in recent_samples]
        efficiencies = [s['efficiency'] for s in recent_samples]
        
        # Group samples by threshold value and calculate average efficiency
        threshold_efficiency = {}
        for threshold_val, efficiency in zip(threshold_values, efficiencies):
            if threshold_val not in threshold_efficiency:
                threshold_efficiency[threshold_val] = []
            threshold_efficiency[threshold_val].append(efficiency)
        
        # Find threshold with best average efficiency
        best_threshold = current_threshold
        best_efficiency = 0.0
        
        for threshold_val, efficiency_list in threshold_efficiency.items():
            if len(efficiency_list) >= 5:  # Need minimum samples
                avg_efficiency = np.mean(efficiency_list)
                if avg_efficiency > best_efficiency:
                    best_efficiency = avg_efficiency
                    best_threshold = threshold_val
        
        # Apply conservative update
        learning_rate = self.optimization_parameters['learning_rate']
        new_threshold = current_threshold + learning_rate * (best_threshold - current_threshold)
        
        # Ensure threshold stays within reasonable bounds
        bounds = self._get_threshold_bounds(threshold_name)
        new_threshold = max(bounds['min'], min(bounds['max'], new_threshold))
        
        return new_threshold
    
    def _get_threshold_bounds(self, threshold_name: str) -> Dict[str, float]:
        """Get reasonable bounds for threshold values"""
        
        bounds_map = {
            'gpu_min_size': {'min': 10, 'max': 10000},
            'cpu_parallel_threshold': {'min': 10, 'max': 1000},
            'gpu_memory_safety_margin': {'min': 0.5, 'max': 0.95},
            'multi_gpu_threshold': {'min': 1000, 'max': 1000000},
            'async_transfer_threshold': {'min': 100, 'max': 100000}
        }
        
        return bounds_map.get(threshold_name, {'min': 1, 'max': 1000000})
    
    def update_all_thresholds(self) -> Dict[str, float]:
        """Update all thresholds based on collected performance data"""
        
        updated_thresholds = {}
        
        for threshold_name in self.current_thresholds:
            new_threshold = self.optimize_threshold(threshold_name)
            if new_threshold is not None:
                old_value = self.current_thresholds[threshold_name]
                self.current_thresholds[threshold_name] = new_threshold
                updated_thresholds[threshold_name] = {
                    'old_value': old_value,
                    'new_value': new_threshold,
                    'change_percent': ((new_threshold - old_value) / old_value) * 100
                }
        
        return updated_thresholds
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive threshold optimization report"""
        
        report = {
            'current_thresholds': self.current_thresholds.copy(),
            'sample_counts': {
                name: len(history) for name, history in self.threshold_history.items()
            },
            'optimization_opportunities': {},
            'performance_trends': {}
        }
        
        for threshold_name, samples in self.threshold_history.items():
            if len(samples) >= 10:
                recent_efficiency = np.mean([s['efficiency'] for s in samples[-20:]])
                report['performance_trends'][threshold_name] = {
                    'recent_efficiency': recent_efficiency,
                    'sample_count': len(samples),
                    'avg_speedup_error': np.mean([s['speedup_error'] for s in samples[-20:]])
                }
        
        return report

# ===============================================================================
# Complete Optimized Performance System
# ===============================================================================

class FinalOptimizedPerformanceSystem:
    """Complete optimized performance system with all enhancements"""
    
    def __init__(self):
        self.base_engine = None
        self.multi_gpu_manager = None
        self.memory_transfer = None
        self.threshold_optimizer = None
        self.performance_monitor = None
        self.initialized = False
        
        # Performance tracking
        self.execution_stats = {
            'total_executions': 0,
            'gpu_executions': 0,
            'multi_gpu_executions': 0,
            'cpu_executions': 0,
            'avg_speedup': 0.0,
            'total_time_saved': 0.0
        }
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize complete optimized performance system"""
        
        print("🚀 Initializing Final Optimized Performance System...")
        
        # Initialize base engine
        if PREVIOUS_PHASES_AVAILABLE:
            self.base_engine = UnifiedPerformanceEngine()
            base_result = self.base_engine.initialize()
        else:
            print("⚠️ Base performance engine not available")
            base_result = {'gpu_available': False}
        
        # Initialize multi-GPU support
        if base_result.get('gpu_available', False):
            try:
                # Detect all available GPUs
                device_count = cp.cuda.runtime.getDeviceCount()
                if device_count > 1:
                    self.multi_gpu_manager = MultiGPUManager()
                    print(f"✅ Multi-GPU support initialized ({device_count} devices)")
                else:
                    print("ℹ️  Single GPU detected - multi-GPU features disabled")
                
                # Initialize optimized memory transfers
                self.memory_transfer = OptimizedMemoryTransfer()
                
            except Exception as e:
                print(f"⚠️ Multi-GPU initialization failed: {e}")
        
        # Initialize adaptive threshold optimizer
        self.threshold_optimizer = AdaptiveThresholdOptimizer()
        
        # Initialize performance monitor
        self.performance_monitor = SystemPerformanceMonitor()
        
        self.initialized = True
        
        # Create initialization summary
        summary = {
            'base_engine': base_result.get('gpu_available', False),
            'multi_gpu': self.multi_gpu_manager is not None,
            'gpu_count': len(self.multi_gpu_manager.device_ids) if self.multi_gpu_manager else 0,
            'optimized_transfers': self.memory_transfer is not None,
            'adaptive_thresholds': True,
            'performance_monitoring': True
        }
        
        print("✅ Final Optimized Performance System ready")
        self._print_initialization_summary(summary)
        
        return summary
    
    def execute_optimized(self, operation: str, *args, **kwargs) -> Any:
        """Execute operation with full optimization stack"""
        
        if not self.initialized:
            raise RuntimeError("Performance system not initialized")
        
        start_time = time.time()
        
        # Determine problem size and complexity
        problem_size = self._estimate_problem_size(*args)
        complexity = self._estimate_complexity(operation, *args)
        
        # Select optimal execution strategy
        strategy = self._select_optimal_strategy(operation, problem_size, complexity)
        
        try:
            # Execute with selected strategy
            if strategy['execution_mode'] == 'multi_gpu' and self.multi_gpu_manager:
                result = self._execute_multi_gpu(operation, strategy, *args, **kwargs)
            elif strategy['execution_mode'] == 'single_gpu' and self.base_engine:
                result = self._execute_single_gpu(operation, *args, **kwargs)
            else:
                result = self._execute_cpu(operation, *args, **kwargs)
            
            # Record performance metrics
            execution_time = time.time() - start_time
            self._record_execution_metrics(operation, strategy, execution_time, problem_size)
            
            return result
            
        except Exception as e:
            # Fallback to simpler execution mode
            warnings.warn(f"Optimized execution failed, using fallback: {e}")
            return self._execute_fallback(operation, *args, **kwargs)
    
    def _select_optimal_strategy(self, operation: str, problem_size: int, 
                               complexity: int) -> Dict[str, Any]:
        """Select optimal execution strategy using adaptive thresholds"""
        
        thresholds = self.threshold_optimizer.current_thresholds
        
        strategy = {
            'execution_mode': 'cpu',
            'expected_speedup': 1.0,
            'resource_requirements': problem_size * 8,  # bytes
            'complexity_score': complexity
        }
        
        # Multi-GPU decision
        if (self.multi_gpu_manager and 
            problem_size >= thresholds['multi_gpu_threshold'] and
            complexity >= 3):
            
            strategy.update({
                'execution_mode': 'multi_gpu',
                'expected_speedup': len(self.multi_gpu_manager.device_ids) * 0.7,  # 70% efficiency
                'device_count': len(self.multi_gpu_manager.device_ids)
            })
        
        # Single GPU decision
        elif (self.base_engine and 
              problem_size >= thresholds['gpu_min_size'] and
              complexity >= 2):
            
            strategy.update({
                'execution_mode': 'single_gpu',
                'expected_speedup': min(100, problem_size / 100),
                'device_id': 0
            })
        
        # CPU parallel decision
        elif problem_size >= thresholds['cpu_parallel_threshold']:
            strategy.update({
                'execution_mode': 'cpu_parallel',
                'expected_speedup': min(8, problem_size / 50)
            })
        
        return strategy
    
    def _execute_multi_gpu(self, operation: str, strategy: Dict[str, Any], 
                          *args, **kwargs) -> Any:
        """Execute operation using multi-GPU"""
        
        self.execution_stats['multi_gpu_executions'] += 1
        
        # Use multi-GPU manager for distributed computation
        if operation == 'stock_update' and len(args) >= 2:
            data = args[0]  # stock_values
            return self.multi_gpu_manager.distribute_computation(
                operation, data, self._stock_update_kernel, *args[1:], **kwargs
            )
        else:
            # Fallback to single GPU
            return self._execute_single_gpu(operation, *args, **kwargs)
    
    def _execute_single_gpu(self, operation: str, *args, **kwargs) -> Any:
        """Execute operation using single GPU with optimizations"""
        
        self.execution_stats['gpu_executions'] += 1
        
        if self.base_engine:
            if operation == 'stock_update':
                return self.base_engine.execute_stock_update(*args, **kwargs)
            elif operation == 'integration_step':
                return self.base_engine.execute_integration_step(*args, **kwargs)
        
        # Fallback
        return self._execute_cpu(operation, *args, **kwargs)
    
    def _execute_cpu(self, operation: str, *args, **kwargs) -> Any:
        """Execute operation using CPU"""
        
        self.execution_stats['cpu_executions'] += 1
        
        # Basic CPU implementations
        if operation == 'stock_update' and len(args) >= 3:
            stock_values, net_flows, dt = args[:3]
            return stock_values + net_flows * dt
        elif operation == 'integration_step' and len(args) >= 3:
            y, derivatives_func, dt = args[:3]
            dy = derivatives_func(y)
            return y + dy * dt
        
        return args[0] if args else None  # Fallback
    
    def _execute_fallback(self, operation: str, *args, **kwargs) -> Any:
        """Fallback execution using simplest method"""
        return self._execute_cpu(operation, *args, **kwargs)
    
    def _stock_update_kernel(self, stock_values: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Stock update kernel for multi-GPU distribution"""
        if len(args) >= 2:
            net_flows, dt = args[:2]
            return stock_values + net_flows * dt
        return stock_values
    
    def _estimate_problem_size(self, *args) -> int:
        """Estimate problem size from arguments"""
        total_size = 0
        for arg in args:
            if isinstance(arg, np.ndarray):
                total_size += arg.size
            elif hasattr(arg, '__len__'):
                total_size += len(arg)
        return total_size
    
    def _estimate_complexity(self, operation: str, *args) -> int:
        """Estimate computational complexity (1-5 scale)"""
        complexity_map = {
            'stock_update': 1,
            'flow_computation': 2,
            'integration_step': 3,
            'matrix_multiplication': 4,
            'optimization': 5
        }
        return complexity_map.get(operation, 2)
    
    def _record_execution_metrics(self, operation: str, strategy: Dict[str, Any],
                                 execution_time: float, problem_size: int):
        """Record execution metrics for performance optimization"""
        
        self.execution_stats['total_executions'] += 1
        
        # Calculate actual speedup (estimated baseline)
        baseline_time = problem_size * 1e-7  # Simple estimate
        actual_speedup = baseline_time / execution_time if execution_time > 0 else 1.0
        
        # Update running averages
        self.execution_stats['avg_speedup'] = (
            (self.execution_stats['avg_speedup'] * (self.execution_stats['total_executions'] - 1) + 
             actual_speedup) / self.execution_stats['total_executions']
        )
        
        time_saved = max(0, baseline_time - execution_time)
        self.execution_stats['total_time_saved'] += time_saved
        
        # Record for threshold optimization
        self.threshold_optimizer.record_performance_sample(
            'gpu_min_size' if 'gpu' in strategy['execution_mode'] else 'cpu_parallel_threshold',
            self.threshold_optimizer.current_thresholds['gpu_min_size'],
            problem_size, execution_time, 
            strategy['expected_speedup'], actual_speedup
        )
    
    def optimize_thresholds(self) -> Dict[str, Any]:
        """Optimize all performance thresholds"""
        
        if not self.threshold_optimizer:
            return {'error': 'Threshold optimizer not available'}
        
        return self.threshold_optimizer.update_all_thresholds()
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive performance system report"""
        
        report = {
            'execution_stats': self.execution_stats.copy(),
            'system_status': {
                'initialized': self.initialized,
                'multi_gpu_available': self.multi_gpu_manager is not None,
                'optimized_transfers': self.memory_transfer is not None
            }
        }
        
        # Add component-specific reports
        if self.base_engine:
            report['base_engine'] = self.base_engine.get_performance_report()
        
        if self.multi_gpu_manager:
            report['multi_gpu'] = self.multi_gpu_manager.get_multi_gpu_stats()
        
        if self.memory_transfer:
            report['memory_transfers'] = self.memory_transfer.get_transfer_stats()
        
        if self.threshold_optimizer:
            report['threshold_optimization'] = self.threshold_optimizer.get_optimization_report()
        
        return report
    
    def _print_initialization_summary(self, summary: Dict[str, Any]):
        """Print initialization summary"""
        
        print(f"\n🎯 FINAL OPTIMIZATION SUMMARY")
        print(f"{'='*50}")
        print(f"Base Engine: {'✅' if summary['base_engine'] else '❌'}")
        print(f"Multi-GPU: {'✅' if summary['multi_gpu'] else '❌'} ({summary['gpu_count']} devices)")
        print(f"Optimized Transfers: {'✅' if summary['optimized_transfers'] else '❌'}")
        print(f"Adaptive Thresholds: {'✅' if summary['adaptive_thresholds'] else '❌'}")
        print(f"Performance Monitoring: {'✅' if summary['performance_monitoring'] else '❌'}")
        print(f"{'='*50}")

class SystemPerformanceMonitor:
    """System-wide performance monitoring"""
    
    def __init__(self):
        self.monitoring_active = False
        self.performance_log = []
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring_active = True
        print("📊 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        print("📊 Performance monitoring stopped")

# ===============================================================================
# Testing and Integration
# ===============================================================================

def test_final_optimization():
    """Comprehensive test of final optimization system"""
    
    print("🧪 Testing Final Optimization System")
    print("=" * 50)
    
    # Initialize system
    system = FinalOptimizedPerformanceSystem()
    init_result = system.initialize()
    
    print(f"\n📊 Initialization Results:")
    for key, value in init_result.items():
        print(f"   {key}: {value}")
    
    # Test different operations and sizes
    test_operations = [
        ('stock_update', [1000, 10000, 100000]),
        ('integration_step', [500, 5000, 50000])
    ]
    
    for operation, sizes in test_operations:
        print(f"\n🔬 Testing {operation}...")
        
        for size in sizes:
            try:
                # Create test data
                if operation == 'stock_update':
                    data = (np.random.rand(size) * 1000, 
                           np.random.rand(size) * 50 - 25, 
                           0.1)
                else:
                    data = (np.random.rand(size), 
                           lambda x: x * 0.1, 
                           0.1)
                
                # Execute with optimization
                start_time = time.time()
                result = system.execute_optimized(operation, *data)
                execution_time = time.time() - start_time
                
                print(f"   Size {size}: {execution_time*1000:.2f}ms")
                
            except Exception as e:
                print(f"   Size {size}: Failed - {e}")
    
    # Test threshold optimization
    print(f"\n🎯 Testing Threshold Optimization...")
    optimization_result = system.optimize_thresholds()
    
    if optimization_result:
        print(f"   Thresholds updated: {len(optimization_result)}")
        for threshold_name, changes in optimization_result.items():
            change_pct = changes['change_percent']
            print(f"   {threshold_name}: {change_pct:+.1f}% change")
    
    # Get final report
    print(f"\n📈 Final Performance Report:")
    report = system.get_comprehensive_report()
    
    stats = report['execution_stats']
    print(f"   Total executions: {stats['total_executions']}")
    print(f"   GPU executions: {stats['gpu_executions']}")
    print(f"   Multi-GPU executions: {stats['multi_gpu_executions']}")
    print(f"   Average speedup: {stats['avg_speedup']:.1f}x")
    print(f"   Total time saved: {stats['total_time_saved']:.3f}s")
    
    return system

if __name__ == "__main__":
    test_final_optimization()