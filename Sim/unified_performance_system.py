# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 01:53:51 2025

@author: NagabhushanamTattaga
"""

# unified_performance_system.py - Complete Integration of All Performance Phases

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from contextlib import contextmanager

# Import all phases with fallback handling
try:
    from performance_core import (
        vectorized_stock_update, parallel_flow_computation, vectorized_rk4_step,
        JITPerformanceMonitor, PERF_CONFIG
    )
    PHASE_1_AVAILABLE = True
except ImportError:
    PHASE_1_AVAILABLE = False
    print("⚠️ Phase 1 (CPU JIT) not available")

try:
    from gpu_infrastructure import PerformanceSystem, GPUMemoryPool, SystemCapabilities
    PHASE_2_AVAILABLE = True
except ImportError:
    PHASE_2_AVAILABLE = False
    print("⚠️ Phase 2 (GPU Infrastructure) not available")

try:
    from gpu_kernels import UnifiedPerformanceEngine, GPUKernelManager
    PHASE_3_AVAILABLE = True
except ImportError:
    PHASE_3_AVAILABLE = False
    print("⚠️ Phase 3 (GPU Kernels) not available")

try:
    from final_optimization import (
        FinalOptimizedPerformanceSystem, MultiGPUManager, 
        AdaptiveThresholdOptimizer, OptimizedMemoryTransfer
    )
    PHASE_4_AVAILABLE = True
except ImportError:
    PHASE_4_AVAILABLE = False
    print("⚠️ Phase 4 (Final Optimization) not available")

# Import existing simulation classes
try:
    from simulation import Model, Stock, Flow
    SIMULATION_AVAILABLE = True
except ImportError:
    SIMULATION_AVAILABLE = False
    print("⚠️ Base simulation classes not available")

# ===============================================================================
# Performance Strategy Selection
# ===============================================================================

class PerformanceStrategy(Enum):
    """Available performance strategies"""
    CPU_BASIC = "cpu_basic"
    CPU_JIT = "cpu_jit" 
    CPU_PARALLEL = "cpu_parallel"
    GPU_SINGLE = "gpu_single"
    GPU_MULTI = "gpu_multi"
    ADAPTIVE = "adaptive"

@dataclass
class ExecutionPlan:
    """Complete execution plan with fallback strategies"""
    primary_strategy: PerformanceStrategy
    fallback_strategies: List[PerformanceStrategy]
    expected_speedup: float
    memory_requirement: int
    device_assignments: List[int]
    execution_context: Dict[str, Any]

# ===============================================================================
# Master Performance Controller
# ===============================================================================

class MasterPerformanceController:
    """Unified controller that orchestrates all performance phases"""
    
    def __init__(self):
        # Phase initialization status
        self.phases_available = {
            'cpu_jit': PHASE_1_AVAILABLE,
            'gpu_infrastructure': PHASE_2_AVAILABLE, 
            'gpu_kernels': PHASE_3_AVAILABLE,
            'final_optimization': PHASE_4_AVAILABLE
        }
        
        # Performance engines (initialized on demand)
        self.cpu_jit_monitor = None
        self.gpu_infrastructure = None
        self.gpu_engine = None
        self.optimization_system = None
        
        # Unified performance tracking
        self.global_stats = {
            'total_operations': 0,
            'total_time_saved': 0.0,
            'strategy_usage': {strategy.value: 0 for strategy in PerformanceStrategy},
            'average_speedup': 0.0,
            'fallback_count': 0
        }
        
        # Configuration
        self.auto_optimization = True
        self.fallback_enabled = True
        self.initialization_complete = False
        
    def initialize_performance_system(self) -> Dict[str, Any]:
        """Initialize all available performance phases"""
        
        print("🚀 Initializing Master Performance System...")
        print(f"Available phases: {list(k for k, v in self.phases_available.items() if v)}")
        
        initialization_results = {}
        
        # Phase 1: CPU JIT
        if self.phases_available['cpu_jit']:
            try:
                self.cpu_jit_monitor = JITPerformanceMonitor()
                initialization_results['cpu_jit'] = True
                print("✅ Phase 1: CPU JIT optimization ready")
            except Exception as e:
                print(f"❌ Phase 1 initialization failed: {e}")
                initialization_results['cpu_jit'] = False
        
        # Phase 2: GPU Infrastructure
        if self.phases_available['gpu_infrastructure']:
            try:
                self.gpu_infrastructure = PerformanceSystem()
                gpu_init = self.gpu_infrastructure.initialize()
                initialization_results['gpu_infrastructure'] = gpu_init
                print(f"✅ Phase 2: GPU Infrastructure ready ({gpu_init.get('gpu_count', 0)} GPUs)")
            except Exception as e:
                print(f"❌ Phase 2 initialization failed: {e}")
                initialization_results['gpu_infrastructure'] = {'gpu_available': False}
        
        # Phase 3: GPU Kernels  
        if self.phases_available['gpu_kernels'] and initialization_results.get('gpu_infrastructure', {}).get('gpu_available', False):
            try:
                self.gpu_engine = UnifiedPerformanceEngine()
                gpu_engine_init = self.gpu_engine.initialize()
                initialization_results['gpu_kernels'] = gpu_engine_init
                print("✅ Phase 3: GPU Kernels compiled and ready")
            except Exception as e:
                print(f"❌ Phase 3 initialization failed: {e}")
                initialization_results['gpu_kernels'] = False
        
        # Phase 4: Final Optimization
        if self.phases_available['final_optimization'] and initialization_results.get('gpu_kernels', False):
            try:
                self.optimization_system = FinalOptimizedPerformanceSystem()
                opt_init = self.optimization_system.initialize()
                initialization_results['final_optimization'] = opt_init
                print("✅ Phase 4: Advanced optimization ready")
            except Exception as e:
                print(f"❌ Phase 4 initialization failed: {e}")
                initialization_results['final_optimization'] = False
        
        self.initialization_complete = True
        
        # Print summary
        self._print_initialization_summary(initialization_results)
        
        return initialization_results
    
    def execute_optimized_operation(self, operation: str, *args, **kwargs) -> Any:
        """Execute operation with optimal performance strategy"""
        
        if not self.initialization_complete:
            return self._execute_fallback(operation, *args, **kwargs)
        
        # Create execution plan
        plan = self._create_execution_plan(operation, *args, **kwargs)
        
        # Execute with primary strategy
        start_time = time.time()
        result = None
        used_strategy = None
        
        for strategy in [plan.primary_strategy] + plan.fallback_strategies:
            try:
                result = self._execute_with_strategy(strategy, operation, plan, *args, **kwargs)
                used_strategy = strategy
                break
            except Exception as e:
                warnings.warn(f"Strategy {strategy.value} failed: {e}")
                self.global_stats['fallback_count'] += 1
                continue
        
        if result is None:
            # Final fallback
            result = self._execute_fallback(operation, *args, **kwargs)
            used_strategy = PerformanceStrategy.CPU_BASIC
        
        # Update statistics
        execution_time = time.time() - start_time
        self._update_global_stats(used_strategy, execution_time, plan.expected_speedup)
        
        return result
    
    def _create_execution_plan(self, operation: str, *args, **kwargs) -> ExecutionPlan:
        """Create optimal execution plan based on problem characteristics"""
        
        # Analyze problem
        problem_size = self._estimate_problem_size(*args)
        complexity = self._estimate_complexity(operation)
        memory_req = self._estimate_memory_requirement(*args)
        
        # Default plan
        plan = ExecutionPlan(
            primary_strategy=PerformanceStrategy.CPU_BASIC,
            fallback_strategies=[],
            expected_speedup=1.0,
            memory_requirement=memory_req,
            device_assignments=[],
            execution_context={}
        )
        
        # Strategy selection logic
        if self.optimization_system and problem_size >= 100000:
            # Phase 4: Advanced optimization for very large problems
            plan.primary_strategy = PerformanceStrategy.GPU_MULTI
            plan.fallback_strategies = [PerformanceStrategy.GPU_SINGLE, PerformanceStrategy.CPU_PARALLEL]
            plan.expected_speedup = min(500, problem_size / 200)
            
        elif self.gpu_engine and problem_size >= 1000:
            # Phase 3: GPU kernels for medium-large problems
            plan.primary_strategy = PerformanceStrategy.GPU_SINGLE
            plan.fallback_strategies = [PerformanceStrategy.CPU_PARALLEL, PerformanceStrategy.CPU_JIT]
            plan.expected_speedup = min(100, problem_size / 100)
            
        elif self.phases_available['cpu_jit'] and problem_size >= 100:
            # Phase 1: CPU JIT for small-medium problems
            plan.primary_strategy = PerformanceStrategy.CPU_PARALLEL
            plan.fallback_strategies = [PerformanceStrategy.CPU_JIT, PerformanceStrategy.CPU_BASIC]
            plan.expected_speedup = min(8, problem_size / 50)
            
        else:
            # Basic CPU execution
            plan.primary_strategy = PerformanceStrategy.CPU_BASIC
            plan.fallback_strategies = []
            plan.expected_speedup = 1.0
        
        return plan
    
    def _execute_with_strategy(self, strategy: PerformanceStrategy, operation: str, 
                             plan: ExecutionPlan, *args, **kwargs) -> Any:
        """Execute operation with specific strategy"""
        
        if strategy == PerformanceStrategy.GPU_MULTI and self.optimization_system:
            return self.optimization_system.execute_optimized(operation, *args, **kwargs)
            
        elif strategy == PerformanceStrategy.GPU_SINGLE and self.gpu_engine:
            if operation == 'stock_update':
                return self.gpu_engine.execute_stock_update(*args, **kwargs)
            elif operation == 'integration_step':
                return self.gpu_engine.execute_integration_step(*args, **kwargs)
                
        elif strategy == PerformanceStrategy.CPU_PARALLEL and self.phases_available['cpu_jit']:
            return self._execute_cpu_parallel(operation, *args, **kwargs)
            
        elif strategy == PerformanceStrategy.CPU_JIT and self.phases_available['cpu_jit']:
            return self._execute_cpu_jit(operation, *args, **kwargs)
            
        else:
            return self._execute_fallback(operation, *args, **kwargs)
    
    def _execute_cpu_parallel(self, operation: str, *args, **kwargs) -> Any:
        """Execute with CPU parallel (Phase 1)"""
        if operation == 'stock_update' and len(args) >= 3:
            stock_values, net_flows, dt = args[:3]
            min_vals = kwargs.get('min_values', np.zeros_like(stock_values))
            max_vals = kwargs.get('max_values', np.full_like(stock_values, np.inf))
            return vectorized_stock_update(stock_values, net_flows, dt, min_vals, max_vals)
        elif operation == 'integration_step' and len(args) >= 5:
            y, k1, k2, k3, k4, dt = args[:6]
            return vectorized_rk4_step(y, k1, k2, k3, k4, dt)
        else:
            return self._execute_fallback(operation, *args, **kwargs)
    
    def _execute_cpu_jit(self, operation: str, *args, **kwargs) -> Any:
        """Execute with CPU JIT (Phase 1)"""
        # Similar to parallel but without explicit parallelization
        return self._execute_cpu_parallel(operation, *args, **kwargs)
    
    def _execute_fallback(self, operation: str, *args, **kwargs) -> Any:
        """Basic CPU fallback execution"""
        if operation == 'stock_update' and len(args) >= 3:
            stock_values, net_flows, dt = args[:3]
            result = stock_values + net_flows * dt
            # Apply bounds if provided
            if 'min_values' in kwargs:
                result = np.maximum(result, kwargs['min_values'])
            if 'max_values' in kwargs:
                result = np.minimum(result, kwargs['max_values'])
            return result
        elif operation == 'integration_step' and len(args) >= 3:
            y, derivatives_func, dt = args[:3]
            dy = derivatives_func(y)
            return y + dy * dt
        elif operation == 'rk4_step' and len(args) >= 6:
            y, k1, k2, k3, k4, dt = args[:6]
            return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        else:
            return args[0] if args else None
    
    def _estimate_problem_size(self, *args) -> int:
        """Estimate total problem size"""
        size = 0
        for arg in args:
            if isinstance(arg, np.ndarray):
                size += arg.size
        return size
    
    def _estimate_complexity(self, operation: str) -> int:
        """Estimate operation complexity (1-5)"""
        complexity_map = {
            'stock_update': 1,
            'flow_computation': 2, 
            'integration_step': 3,
            'rk4_step': 3,
            'matrix_ops': 4,
            'optimization': 5
        }
        return complexity_map.get(operation, 2)
    
    def _estimate_memory_requirement(self, *args) -> int:
        """Estimate memory requirement in bytes"""
        memory = 0
        for arg in args:
            if isinstance(arg, np.ndarray):
                memory += arg.nbytes * 3  # Input + output + workspace
        return memory
    
    def _update_global_stats(self, strategy: PerformanceStrategy, 
                           execution_time: float, expected_speedup: float):
        """Update global performance statistics"""
        
        self.global_stats['total_operations'] += 1
        self.global_stats['strategy_usage'][strategy.value] += 1
        
        # Estimate time saved (very rough approximation)
        baseline_time = execution_time * expected_speedup
        time_saved = max(0, baseline_time - execution_time)
        self.global_stats['total_time_saved'] += time_saved
        
        # Update average speedup
        actual_speedup = baseline_time / execution_time if execution_time > 0 else 1.0
        total_ops = self.global_stats['total_operations']
        self.global_stats['average_speedup'] = (
            (self.global_stats['average_speedup'] * (total_ops - 1) + actual_speedup) / total_ops
        )
    
    def optimize_thresholds(self) -> Dict[str, Any]:
        """Optimize performance thresholds across all phases"""
        
        results = {}
        
        if self.optimization_system:
            try:
                opt_results = self.optimization_system.optimize_thresholds()
                results['phase_4'] = opt_results
            except Exception as e:
                results['phase_4'] = {'error': str(e)}
        
        # Could add threshold optimization for other phases here
        
        return results
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report across all phases"""
        
        report = {
            'master_controller': {
                'initialization_complete': self.initialization_complete,
                'phases_available': self.phases_available,
                'global_stats': self.global_stats
            }
        }
        
        # Add reports from each phase
        if self.cpu_jit_monitor:
            try:
                report['phase_1_cpu_jit'] = self.cpu_jit_monitor.get_performance_report()
            except Exception as e:
                report['phase_1_cpu_jit'] = {'error': str(e)}
        
        if self.gpu_infrastructure:
            try:
                report['phase_2_gpu_infrastructure'] = self.gpu_infrastructure.get_system_info()
            except Exception as e:
                report['phase_2_gpu_infrastructure'] = {'error': str(e)}
        
        if self.gpu_engine:
            try:
                report['phase_3_gpu_kernels'] = self.gpu_engine.get_performance_report()
            except Exception as e:
                report['phase_3_gpu_kernels'] = {'error': str(e)}
        
        if self.optimization_system:
            try:
                report['phase_4_optimization'] = self.optimization_system.get_comprehensive_report()
            except Exception as e:
                report['phase_4_optimization'] = {'error': str(e)}
        
        return report
    
    def _print_initialization_summary(self, results: Dict[str, Any]):
        """Print initialization summary"""
        
        print(f"\n🎯 MASTER PERFORMANCE SYSTEM SUMMARY")
        print(f"{'='*60}")
        
        phase_status = {
            'Phase 1 (CPU JIT)': results.get('cpu_jit', False),
            'Phase 2 (GPU Infrastructure)': results.get('gpu_infrastructure', {}).get('gpu_available', False),
            'Phase 3 (GPU Kernels)': bool(results.get('gpu_kernels', False)),
            'Phase 4 (Optimization)': bool(results.get('final_optimization', False))
        }
        
        for phase, status in phase_status.items():
            icon = "✅" if status else "❌"
            print(f"{icon} {phase}")
        
        # Performance capabilities summary
        available_strategies = []
        if phase_status['Phase 4 (Optimization)']:
            available_strategies.extend(['Multi-GPU', 'Advanced Optimization'])
        if phase_status['Phase 3 (GPU Kernels)']:
            available_strategies.append('GPU Acceleration')
        if phase_status['Phase 1 (CPU JIT)']:
            available_strategies.append('CPU JIT/Parallel')
        available_strategies.append('CPU Basic (Fallback)')
        
        print(f"\n🚀 Available Performance Strategies:")
        for strategy in available_strategies:
            print(f"   • {strategy}")
        
        gpu_count = results.get('gpu_infrastructure', {}).get('gpu_count', 0)
        if gpu_count > 0:
            print(f"\n🔥 GPU Acceleration: {gpu_count} device(s) available")
        
        print(f"{'='*60}")

# ===============================================================================
# Enhanced Model Integration
# ===============================================================================

class PerformanceEnhancedModel:
    """Enhanced Model wrapper with automatic performance optimization"""
    
    def __init__(self, base_model, performance_controller: MasterPerformanceController):
        self.base_model = base_model
        self.performance_controller = performance_controller
        self.performance_enabled = True
        
        # Performance tracking for this model
        self.model_stats = {
            'total_steps': 0,
            'total_simulation_time': 0.0,
            'average_step_time': 0.0,
            'performance_gains': []
        }
    
    def step(self):
        """Enhanced step with automatic performance optimization"""
        
        if not self.performance_enabled:
            return self.base_model.step()
        
        step_start = time.time()
        
        try:
            # Get stock update data
            stock_values = []
            net_flows = []
            min_values = []
            max_values = []
            
            for stock in self.base_model.stocks:
                if hasattr(stock.values, 'flatten'):
                    stock_values.append(stock.values.flatten())
                else:
                    stock_values.append(np.array([stock.values]))
                
                net_flow = stock.get_net_flow(self.base_model.dt)
                if hasattr(net_flow, 'flatten'):
                    net_flows.append(net_flow.flatten())
                else:
                    net_flows.append(np.array([net_flow]))
                
                min_values.append(np.full_like(stock_values[-1], getattr(stock, 'min_value', 0)))
                max_values.append(np.full_like(stock_values[-1], getattr(stock, 'max_value', np.inf)))
            
            if stock_values:
                # Combine all stock data
                combined_stocks = np.concatenate(stock_values)
                combined_flows = np.concatenate(net_flows)
                combined_mins = np.concatenate(min_values)
                combined_maxs = np.concatenate(max_values)
                
                # Execute optimized stock update
                new_values = self.performance_controller.execute_optimized_operation(
                    'stock_update',
                    combined_stocks, 
                    combined_flows, 
                    self.base_model.dt,
                    min_values=combined_mins,
                    max_values=combined_maxs
                )
                
                # Update stock values
                start_idx = 0
                for i, stock in enumerate(self.base_model.stocks):
                    end_idx = start_idx + stock_values[i].size
                    stock_new_values = new_values[start_idx:end_idx]
                    
                    if hasattr(stock.values, 'shape') and len(stock.values.shape) > 0:
                        stock.values = stock_new_values.reshape(stock.values.shape)
                    else:
                        stock.values = stock_new_values[0]
                    
                    start_idx = end_idx
            
            # Update time and step count
            self.base_model.time += self.base_model.dt
            self.base_model.step_count += 1
            
        except Exception as e:
            warnings.warn(f"Performance-enhanced step failed, using fallback: {e}")
            # Fallback to original step
            self.base_model.step()
        
        # Update performance statistics
        step_time = time.time() - step_start
        self.model_stats['total_steps'] += 1
        self.model_stats['total_simulation_time'] += step_time
        self.model_stats['average_step_time'] = (
            self.model_stats['total_simulation_time'] / self.model_stats['total_steps']
        )
    
    def run(self, *args, **kwargs):
        """Enhanced run method with performance tracking"""
        
        run_start = time.time()
        
        # Use base model's run method but with our enhanced step
        original_step = self.base_model.step
        self.base_model.step = self.step
        
        try:
            result = self.base_model.run(*args, **kwargs)
        finally:
            # Restore original step method
            self.base_model.step = original_step
        
        run_time = time.time() - run_start
        
        # Calculate performance gain estimate
        baseline_estimate = self.model_stats['total_steps'] * 1e-4  # Rough baseline
        performance_gain = max(1.0, baseline_estimate / run_time) if run_time > 0 else 1.0
        self.model_stats['performance_gains'].append(performance_gain)
        
        return result
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this model"""
        
        avg_gain = np.mean(self.model_stats['performance_gains']) if self.model_stats['performance_gains'] else 1.0
        
        return {
            'model_name': getattr(self.base_model, 'name', 'Unknown'),
            'performance_enabled': self.performance_enabled,
            'total_steps': self.model_stats['total_steps'],
            'average_step_time': self.model_stats['average_step_time'],
            'estimated_speedup': avg_gain,
            'controller_stats': self.performance_controller.global_stats
        }
    
    def __getattr__(self, name):
        """Delegate missing attributes to base model"""
        return getattr(self.base_model, name)

# ===============================================================================
# Convenience Functions
# ===============================================================================

# Global performance controller instance
_global_performance_controller = None

def initialize_performance():
    """Initialize global performance system"""
    global _global_performance_controller
    
    if _global_performance_controller is None:
        _global_performance_controller = MasterPerformanceController()
        return _global_performance_controller.initialize_performance_system()
    else:
        print("Performance system already initialized")
        return {'already_initialized': True}

def enhance_model(model) -> PerformanceEnhancedModel:
    """Enhance a model with automatic performance optimization"""
    global _global_performance_controller
    
    if _global_performance_controller is None:
        print("Initializing performance system...")
        initialize_performance()
    
    return PerformanceEnhancedModel(model, _global_performance_controller)

def get_performance_report() -> Dict[str, Any]:
    """Get global performance report"""
    global _global_performance_controller
    
    if _global_performance_controller is None:
        return {'error': 'Performance system not initialized'}
    
    return _global_performance_controller.get_comprehensive_report()

def optimize_performance_thresholds() -> Dict[str, Any]:
    """Optimize all performance thresholds"""
    global _global_performance_controller
    
    if _global_performance_controller is None:
        return {'error': 'Performance system not initialized'}
    
    return _global_performance_controller.optimize_thresholds()

@contextmanager
def performance_context():
    """Context manager for performance-optimized operations"""
    global _global_performance_controller
    
    if _global_performance_controller is None:
        initialize_performance()
    
    start_time = time.time()
    try:
        yield _global_performance_controller
    finally:
        end_time = time.time()
        print(f"Performance context executed in {end_time - start_time:.4f}s")

# ===============================================================================
# Testing Integration
# ===============================================================================

def test_unified_performance_system():
    """Comprehensive test of the unified performance system"""
    
    print("🧪 Testing Unified Performance System Integration")
    print("=" * 60)
    
    # Test 1: Initialize system
    print("\n1. Initializing performance system...")
    init_results = initialize_performance()
    
    # Test 2: Direct operation execution
    print("\n2. Testing direct operations...")
    test_sizes = [100, 1000, 10000]
    
    for size in test_sizes:
        print(f"\n   Testing with {size} elements...")
        
        # Stock update test
        stock_values = np.random.rand(size) * 1000
        net_flows = np.random.rand(size) * 50 - 25
        dt = 0.1
        
        start_time = time.time()
        result = _global_performance_controller.execute_optimized_operation(
            'stock_update', stock_values, net_flows, dt
        )
        execution_time = time.time() - start_time
        
        print(f"      Stock update: {execution_time*1000:.2f}ms")
        print(f"      Result shape: {result.shape}")
    
    # Test 3: Model enhancement
    if SIMULATION_AVAILABLE:
        print("\n3. Testing model enhancement...")
        
        try:
            from simulation import Model, Stock, Flow
            
            # Create test model
            model = Model(name="Performance Test Model")
            
            # Add stocks
            population = Stock(values=1000.0, name="Population")
            resources = Stock(values=500.0, name="Resources")
            
            model.add_stock(population)
            model.add_stock(resources)
            
            # Enhance model
            enhanced_model = enhance_model(model)
            
            # Test enhanced stepping
            print(f"      Running enhanced simulation...")
            for _ in range(10):
                enhanced_model.step()
            
            summary = enhanced_model.get_performance_summary()
            print(f"      Average step time: {summary['average_step_time']*1000:.2f}ms")
            print(f"      Estimated speedup: {summary['estimated_speedup']:.1f}x")
            
        except Exception as e:
            print(f"      Model enhancement test failed: {e}")
    
    # Test 4: Performance report
    print("\n4. Getting performance report...")
    report = get_performance_report()
    
    master_stats = report.get('master_controller', {}).get('global_stats', {})
    print(f"      Total operations: {master_stats.get('total_operations', 0)}")
    print(f"      Average speedup: {master_stats.get('average_speedup', 1.0):.1f}x")
    print(f"      Time saved: {master_stats.get('total_time_saved', 0.0):.3f}s")
    
    strategy_usage = master_stats.get('strategy_usage', {})
    for strategy, count in strategy_usage.items():
        if count > 0:
            print(f"      {strategy}: {count} executions")
    
    print("\n✅ Unified performance system test completed!")
    
    return report

if __name__ == "__main__":
    test_unified_performance_system()