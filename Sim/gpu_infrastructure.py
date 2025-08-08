# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 01:39:58 2025

@author: NagabhushanamTattaga
"""

# gpu_infrastructure.py - Phase 2: GPU Infrastructure & Hardware Detection

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import psutil
import platform
import subprocess
import json

# GPU Libraries (with fallback handling)
try:
    import cupy as cp
    import cupyx
    from cupyx.scipy import sparse as cp_sparse
    GPU_AVAILABLE = True
    print("✅ CuPy detected - GPU acceleration available")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ CuPy not available - GPU acceleration disabled")
    # Create mock cp for compatibility
    class MockCuPy:
        @staticmethod
        def array(x): return np.array(x)
        @staticmethod
        def zeros_like(x): return np.zeros_like(x)
        @staticmethod
        def asarray(x): return np.asarray(x)
    cp = MockCuPy()

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("⚠️ pynvml not available - limited GPU monitoring")

# ===============================================================================
# Hardware Detection and Capabilities
# ===============================================================================

@dataclass
class GPUDevice:
    """GPU device information"""
    device_id: int
    name: str
    memory_total: int  # bytes
    memory_free: int   # bytes
    compute_capability: Tuple[int, int]
    multiprocessor_count: int
    max_threads_per_block: int
    warp_size: int
    is_available: bool = True

@dataclass 
class SystemCapabilities:
    """Complete system hardware capabilities"""
    cpu_cores: int
    cpu_threads: int
    ram_total: int  # bytes
    ram_available: int  # bytes
    
    gpu_count: int = 0
    gpus: List[GPUDevice] = field(default_factory=list)
    
    # Computed capabilities
    has_gpu: bool = False
    has_cuda: bool = False
    optimal_execution_mode: str = "cpu"
    
    def __post_init__(self):
        self.has_gpu = self.gpu_count > 0
        self.has_cuda = self.has_gpu and GPU_AVAILABLE

class HardwareDetector:
    """Comprehensive hardware detection and capability assessment"""
    
    def __init__(self):
        self._gpu_initialized = False
        self._system_caps = None
    
    def detect_system_capabilities(self) -> SystemCapabilities:
        """Detect and assess complete system capabilities"""
        
        print("🔍 Detecting system hardware capabilities...")
        
        # CPU Detection
        cpu_info = self._detect_cpu()
        
        # Memory Detection  
        memory_info = self._detect_memory()
        
        # GPU Detection
        gpu_info = self._detect_gpus()
        
        # Create capabilities object
        caps = SystemCapabilities(
            cpu_cores=cpu_info['cores'],
            cpu_threads=cpu_info['threads'],
            ram_total=memory_info['total'],
            ram_available=memory_info['available'],
            gpu_count=len(gpu_info),
            gpus=gpu_info
        )
        
        # Determine optimal execution mode
        caps.optimal_execution_mode = self._determine_optimal_mode(caps)
        
        self._system_caps = caps
        self._print_capabilities_summary(caps)
        
        return caps
    
    def _detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU capabilities"""
        try:
            return {
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True),
                'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
                'architecture': platform.machine()
            }
        except Exception as e:
            warnings.warn(f"CPU detection failed: {e}")
            return {'cores': 1, 'threads': 1, 'frequency': 0, 'architecture': 'unknown'}
    
    def _detect_memory(self) -> Dict[str, int]:
        """Detect system memory"""
        try:
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'percent_used': memory.percent
            }
        except Exception as e:
            warnings.warn(f"Memory detection failed: {e}")
            return {'total': 0, 'available': 0, 'percent_used': 0}
    
    def _detect_gpus(self) -> List[GPUDevice]:
        """Detect and characterize GPU devices"""
        
        if not GPU_AVAILABLE:
            return []
        
        gpus = []
        
        try:
            # Initialize CUDA context
            cp.cuda.runtime.getDeviceCount()
            
            # Get device count
            device_count = cp.cuda.runtime.getDeviceCount()
            print(f"🔍 Detected {device_count} CUDA device(s)")
            
            for device_id in range(device_count):
                try:
                    with cp.cuda.Device(device_id):
                        # Get device properties
                        props = cp.cuda.runtime.getDeviceProperties(device_id)
                        
                        # Get memory info
                        meminfo = cp.cuda.runtime.memGetInfo()
                        memory_free, memory_total = meminfo
                        
                        gpu = GPUDevice(
                            device_id=device_id,
                            name=props['name'].decode('utf-8'),
                            memory_total=memory_total,
                            memory_free=memory_free,
                            compute_capability=(props['major'], props['minor']),
                            multiprocessor_count=props['multiProcessorCount'],
                            max_threads_per_block=props['maxThreadsPerBlock'],
                            warp_size=props['warpSize']
                        )
                        
                        gpus.append(gpu)
                        print(f"✅ GPU {device_id}: {gpu.name} ({memory_total//1024//1024}MB)")
                        
                except Exception as e:
                    print(f"⚠️ Failed to initialize GPU {device_id}: {e}")
                    continue
                    
        except Exception as e:
            print(f"⚠️ CUDA initialization failed: {e}")
            return []
        
        return gpus
    
    def _determine_optimal_mode(self, caps: SystemCapabilities) -> str:
        """Determine optimal execution mode based on hardware"""
        
        if not caps.has_cuda:
            if caps.cpu_threads >= 4:
                return "cpu_parallel"
            else:
                return "cpu_serial"
        
        # GPU is available - check capabilities
        if caps.gpus:
            primary_gpu = caps.gpus[0]
            
            # Check if GPU has sufficient memory and compute capability
            if (primary_gpu.memory_total > 1024*1024*1024 and  # 1GB minimum
                primary_gpu.compute_capability[0] >= 6):       # Compute 6.0+
                
                if caps.gpu_count > 1:
                    return "gpu_multi"
                else:
                    return "gpu_single"
            else:
                return "cpu_parallel"
        
        return "cpu_parallel"
    
    def _print_capabilities_summary(self, caps: SystemCapabilities):
        """Print hardware capabilities summary"""
        
        print(f"\n🖥️  SYSTEM CAPABILITIES SUMMARY")
        print(f"{'='*50}")
        print(f"CPU: {caps.cpu_cores} cores / {caps.cpu_threads} threads")
        print(f"RAM: {caps.ram_total//1024//1024//1024:.1f}GB total, "
              f"{caps.ram_available//1024//1024//1024:.1f}GB available")
        
        if caps.has_cuda:
            print(f"GPU: {caps.gpu_count} CUDA device(s)")
            for i, gpu in enumerate(caps.gpus):
                print(f"  GPU {i}: {gpu.name}")
                print(f"    Memory: {gpu.memory_total//1024//1024}MB")
                print(f"    Compute: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
        else:
            print("GPU: Not available")
        
        print(f"Optimal Mode: {caps.optimal_execution_mode}")
        print(f"{'='*50}")

# ===============================================================================
# GPU Memory Management
# ===============================================================================

class GPUMemoryPool:
    """Efficient GPU memory pool management"""
    
    def __init__(self, device_id: int = 0, pool_fraction: float = 0.8):
        self.device_id = device_id
        self.pool_fraction = pool_fraction
        self.pools = {}  # dtype -> List[arrays]
        self.allocated_memory = 0
        self.max_memory = 0
        self.allocation_count = 0
        self.deallocation_count = 0
        
        if GPU_AVAILABLE:
            self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize GPU memory pool"""
        try:
            with cp.cuda.Device(self.device_id):
                # Get available memory
                meminfo = cp.cuda.runtime.memGetInfo()
                memory_free, memory_total = meminfo
                
                # Set maximum pool size
                self.max_memory = int(memory_free * self.pool_fraction)
                
                print(f"🔧 GPU Memory Pool initialized:")
                print(f"   Device: {self.device_id}")
                print(f"   Available: {memory_free//1024//1024}MB")
                print(f"   Pool Size: {self.max_memory//1024//1024}MB")
                
        except Exception as e:
            warnings.warn(f"GPU memory pool initialization failed: {e}")
    
    def get_array(self, shape: Tuple[int, ...], dtype=np.float64) -> Union[np.ndarray, 'cp.ndarray']:
        """Get array from pool or allocate new"""
        
        if not GPU_AVAILABLE:
            return np.empty(shape, dtype=dtype)
        
        try:
            with cp.cuda.Device(self.device_id):
                array_size = np.prod(shape) * np.dtype(dtype).itemsize
                
                # Check pool for compatible array
                pool_key = (shape, dtype)
                if pool_key in self.pools and self.pools[pool_key]:
                    array = self.pools[pool_key].pop()
                    self.deallocation_count += 1
                    return array
                
                # Check memory limits
                if self.allocated_memory + array_size > self.max_memory:
                    self._cleanup_pool()
                
                # Allocate new array
                array = cp.empty(shape, dtype=dtype)
                self.allocated_memory += array_size
                self.allocation_count += 1
                
                return array
                
        except Exception as e:
            warnings.warn(f"GPU array allocation failed: {e}")
            # Fallback to CPU
            return np.empty(shape, dtype=dtype)
    
    def return_array(self, array: Union[np.ndarray, 'cp.ndarray']):
        """Return array to pool"""
        
        if not GPU_AVAILABLE or not hasattr(array, 'device'):
            return  # CPU array or no GPU
        
        try:
            pool_key = (array.shape, array.dtype)
            if pool_key not in self.pools:
                self.pools[pool_key] = []
            
            # Limit pool size per type
            if len(self.pools[pool_key]) < 10:
                self.pools[pool_key].append(array)
            
        except Exception as e:
            warnings.warn(f"Array return to pool failed: {e}")
    
    def _cleanup_pool(self, target_fraction: float = 0.5):
        """Clean up pool to free memory"""
        target_memory = int(self.max_memory * target_fraction)
        
        # Remove arrays from pools until target is reached
        for pool_key in list(self.pools.keys()):
            while (self.allocated_memory > target_memory and 
                   self.pools[pool_key]):
                array = self.pools[pool_key].pop()
                array_size = array.size * array.dtype.itemsize
                self.allocated_memory -= array_size
                del array
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        return {
            'device_id': self.device_id,
            'allocated_memory_mb': self.allocated_memory // 1024 // 1024,
            'max_memory_mb': self.max_memory // 1024 // 1024,
            'utilization': self.allocated_memory / self.max_memory if self.max_memory > 0 else 0,
            'allocation_count': self.allocation_count,
            'deallocation_count': self.deallocation_count,
            'pool_types': len(self.pools),
            'pooled_arrays': sum(len(pool) for pool in self.pools.values())
        }

# ===============================================================================
# Execution Context Management
# ===============================================================================

class ExecutionContext:
    """Manages execution context and device selection"""
    
    def __init__(self, capabilities: SystemCapabilities):
        self.capabilities = capabilities
        self.current_device = None
        self.memory_pools = {}
        self.execution_stats = {
            'cpu_executions': 0,
            'gpu_executions': 0,
            'gpu_fallbacks': 0,
            'total_gpu_memory_used': 0
        }
        
        # Initialize GPU memory pools
        if capabilities.has_cuda:
            for gpu in capabilities.gpus:
                self.memory_pools[gpu.device_id] = GPUMemoryPool(gpu.device_id)
    
    def select_optimal_device(self, problem_size: int, memory_requirement: int) -> Dict[str, Any]:
        """Select optimal execution device based on problem characteristics"""
        
        selection = {
            'device_type': 'cpu',
            'device_id': None,
            'memory_pool': None,
            'expected_speedup': 1.0,
            'reason': 'default'
        }
        
        # GPU selection logic
        if self.capabilities.has_cuda and problem_size >= 100:
            
            for gpu in self.capabilities.gpus:
                # Check memory availability
                if gpu.memory_free > memory_requirement * 2:  # 2x safety margin
                    
                    # Estimate speedup based on problem size
                    if problem_size >= 10000:
                        expected_speedup = min(200, problem_size / 50)
                    elif problem_size >= 1000:
                        expected_speedup = min(50, problem_size / 20)
                    else:
                        expected_speedup = min(10, problem_size / 10)
                    
                    selection.update({
                        'device_type': 'gpu',
                        'device_id': gpu.device_id,
                        'memory_pool': self.memory_pools.get(gpu.device_id),
                        'expected_speedup': expected_speedup,
                        'reason': f'gpu_optimal_size_{problem_size}'
                    })
                    break
            
            else:
                selection['reason'] = 'gpu_insufficient_memory'
        
        elif self.capabilities.cpu_threads >= 4 and problem_size >= 50:
            selection.update({
                'device_type': 'cpu_parallel',
                'expected_speedup': min(self.capabilities.cpu_threads, problem_size / 10),
                'reason': f'cpu_parallel_optimal'
            })
        
        return selection
    
    def execute_with_context(self, func, *args, **kwargs):
        """Execute function with optimal device context"""
        
        # Estimate problem size and memory requirement
        problem_size = self._estimate_problem_size(*args)
        memory_req = self._estimate_memory_requirement(*args)
        
        # Select optimal device
        device_selection = self.select_optimal_device(problem_size, memory_req)
        
        try:
            if device_selection['device_type'] == 'gpu':
                return self._execute_on_gpu(func, device_selection, *args, **kwargs)
            else:
                return self._execute_on_cpu(func, *args, **kwargs)
                
        except Exception as e:
            # GPU fallback to CPU
            if device_selection['device_type'] == 'gpu':
                warnings.warn(f"GPU execution failed, falling back to CPU: {e}")
                self.execution_stats['gpu_fallbacks'] += 1
                return self._execute_on_cpu(func, *args, **kwargs)
            else:
                raise e
    
    def _execute_on_gpu(self, func, device_selection, *args, **kwargs):
        """Execute function on GPU"""
        
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU execution requested but CuPy not available")
        
        device_id = device_selection['device_id']
        memory_pool = device_selection['memory_pool']
        
        with cp.cuda.Device(device_id):
            # Convert arguments to GPU arrays
            gpu_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    gpu_arg = memory_pool.get_array(arg.shape, arg.dtype)
                    gpu_arg[:] = cp.asarray(arg)
                    gpu_args.append(gpu_arg)
                else:
                    gpu_args.append(arg)
            
            # Execute function
            result = func(*gpu_args, **kwargs)
            
            # Convert result back to CPU if needed
            if hasattr(result, 'device'):
                cpu_result = cp.asnumpy(result)
                memory_pool.return_array(result)
                result = cpu_result
            
            # Return GPU arguments to pool
            for i, arg in enumerate(gpu_args):
                if hasattr(arg, 'device'):
                    memory_pool.return_array(arg)
            
            self.execution_stats['gpu_executions'] += 1
            return result
    
    def _execute_on_cpu(self, func, *args, **kwargs):
        """Execute function on CPU"""
        self.execution_stats['cpu_executions'] += 1
        return func(*args, **kwargs)
    
    def _estimate_problem_size(self, *args) -> int:
        """Estimate problem size from arguments"""
        total_size = 0
        for arg in args:
            if isinstance(arg, np.ndarray):
                total_size += arg.size
            elif hasattr(arg, '__len__'):
                total_size += len(arg)
        return total_size
    
    def _estimate_memory_requirement(self, *args) -> int:
        """Estimate memory requirement in bytes"""
        memory_req = 0
        for arg in args:
            if isinstance(arg, np.ndarray):
                memory_req += arg.nbytes * 3  # Input + output + workspace
        return memory_req
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total_executions = (self.execution_stats['cpu_executions'] + 
                           self.execution_stats['gpu_executions'])
        
        stats = {
            **self.execution_stats,
            'total_executions': total_executions,
            'gpu_success_rate': (self.execution_stats['gpu_executions'] / 
                               max(1, self.execution_stats['gpu_executions'] + 
                                   self.execution_stats['gpu_fallbacks'])),
            'memory_pools': {}
        }
        
        # Add memory pool stats
        for device_id, pool in self.memory_pools.items():
            stats['memory_pools'][device_id] = pool.get_memory_info()
        
        return stats

# ===============================================================================
# System Detection and Initialization
# ===============================================================================

class PerformanceSystem:
    """Complete performance system initialization and management"""
    
    def __init__(self):
        self.detector = HardwareDetector()
        self.capabilities = None
        self.execution_context = None
        self.initialized = False
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize complete performance system"""
        
        print("🚀 Initializing Performance System...")
        
        # Detect hardware capabilities
        self.capabilities = self.detector.detect_system_capabilities()
        
        # Initialize execution context
        self.execution_context = ExecutionContext(self.capabilities)
        
        self.initialized = True
        
        # Return initialization summary
        return {
            'initialized': True,
            'gpu_available': self.capabilities.has_cuda,
            'gpu_count': self.capabilities.gpu_count,
            'optimal_mode': self.capabilities.optimal_execution_mode,
            'cpu_threads': self.capabilities.cpu_threads,
            'ram_gb': self.capabilities.ram_total // 1024 // 1024 // 1024
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get complete system information"""
        
        if not self.initialized:
            return {'error': 'System not initialized'}
        
        info = {
            'hardware': {
                'cpu_cores': self.capabilities.cpu_cores,
                'cpu_threads': self.capabilities.cpu_threads,
                'ram_total_gb': self.capabilities.ram_total // 1024 // 1024 // 1024,
                'ram_available_gb': self.capabilities.ram_available // 1024 // 1024 // 1024
            },
            'gpu': {
                'available': self.capabilities.has_cuda,
                'count': self.capabilities.gpu_count,
                'devices': [
                    {
                        'id': gpu.device_id,
                        'name': gpu.name,
                        'memory_mb': gpu.memory_total // 1024 // 1024,
                        'compute_capability': f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
                    } for gpu in self.capabilities.gpus
                ]
            },
            'execution': self.execution_context.get_execution_stats() if self.execution_context else {},
            'optimal_mode': self.capabilities.optimal_execution_mode
        }
        
        return info

# ===============================================================================
# Testing and Validation
# ===============================================================================

def test_gpu_infrastructure():
    """Test GPU infrastructure and capabilities"""
    
    print("🧪 Testing GPU Infrastructure")
    print("=" * 40)
    
    # Initialize performance system
    perf_system = PerformanceSystem()
    init_result = perf_system.initialize()
    
    print(f"\n📊 Initialization Result:")
    for key, value in init_result.items():
        print(f"   {key}: {value}")
    
    # Test memory allocation if GPU available
    if init_result['gpu_available']:
        print(f"\n🔧 Testing GPU Memory Management...")
        
        device_id = 0
        memory_pool = GPUMemoryPool(device_id)
        
        # Test array allocation and deallocation
        test_shapes = [(1000,), (1000, 1000), (100, 100, 100)]
        
        for shape in test_shapes:
            try:
                array = memory_pool.get_array(shape, np.float64)
                print(f"   ✅ Allocated {shape} array: {array.shape}")
                memory_pool.return_array(array)
                print(f"   ✅ Returned {shape} array to pool")
            except Exception as e:
                print(f"   ❌ Failed to allocate {shape}: {e}")
        
        # Print memory stats
        memory_info = memory_pool.get_memory_info()
        print(f"\n📈 Memory Pool Stats:")
        for key, value in memory_info.items():
            print(f"   {key}: {value}")
    
    # Get final system info
    system_info = perf_system.get_system_info()
    print(f"\n🖥️  Final System Info:")
    print(json.dumps(system_info, indent=2))
    
    return perf_system

if __name__ == "__main__":
    test_gpu_infrastructure()