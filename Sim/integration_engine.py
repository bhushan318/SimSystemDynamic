# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 22:11:40 2025

@author: NagabhushanamTattaga
"""

"""
Advanced Integration Engine for System Dynamics
Implements multiple numerical integration methods with automatic selection
"""

import numpy as np
import warnings
from abc import ABC, abstractmethod
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple, Optional, Callable, Any
from enum import Enum
import time
import networkx as nx
from dataclasses import dataclass

class IntegrationMethod(ABC):
    """Abstract base class for integration methods"""
    
    @abstractmethod
    def integrate(self, model: 'Model', t_span: tuple, y0: np.ndarray, **kwargs) -> dict:
        """Integrate the system over the given time span"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the method name"""
        pass

class EulerMethod(IntegrationMethod):
    """Simple Euler integration method"""
    
    def get_name(self) -> str:
        return "Euler"
    
    def integrate(self, model: 'Model', t_span: tuple, y0: np.ndarray, **kwargs) -> dict:
        """Simple Euler integration"""
        t0, tf = t_span
        dt = kwargs.get('dt', model.dt)
        t_eval = np.arange(t0, tf + dt, dt)
        
        # Initialize results
        y = np.zeros((len(t_eval), len(y0)))
        y[0] = y0
        
        # Integration loop
        for i in range(1, len(t_eval)):
            t = t_eval[i-1]
            try:
                dydt = model.compute_derivatives(t, y[i-1])
                y[i] = y[i-1] + dt * dydt
                
                # Apply bounds if needed
                y[i] = model.apply_bounds(y[i])
                
            # Replace existing exception handling with:
            except Exception as e:
                error_context = f"at t={t:.6f}, step {i}/{len(t_eval)}, dt={dt}"
                return {
                    't': t_eval[:i],
                    'y': y[:i].T,
                    'success': False,
                    'message': f'Euler integration failed {error_context}: {str(e)}',
                    'nfev': len(t_eval) - 1
                }
        
        return {
            't': t_eval,
            'y': y.T,
            'success': True,
            'message': 'Integration completed successfully',
            'nfev': len(t_eval) - 1  # Number of function evaluations
        }

class RK4Method(IntegrationMethod):
    """Runge-Kutta 4th order integration method"""
    
    def get_name(self) -> str:
        return "RK4"
    
    def integrate(self, model: 'Model', t_span: tuple, y0: np.ndarray, **kwargs) -> dict:
        """Runge-Kutta 4th order integration"""
        t0, tf = t_span
        dt = kwargs.get('dt', model.dt)
        t_eval = np.arange(t0, tf + dt, dt)
        
        # Initialize results
        y = np.zeros((len(t_eval), len(y0)))
        y[0] = y0
        nfev = 0
        
        # Integration loop
        for i in range(1, len(t_eval)):
            t = t_eval[i-1]
            y_curr = y[i-1]
            
            try:
                # RK4 stages
                k1 = dt * model.compute_derivatives(t, y_curr)
                k2 = dt * model.compute_derivatives(t + dt/2, y_curr + k1/2)
                k3 = dt * model.compute_derivatives(t + dt/2, y_curr + k2/2)
                k4 = dt * model.compute_derivatives(t + dt, y_curr + k3)
                
                y[i] = y_curr + (k1 + 2*k2 + 2*k3 + k4) / 6
                
                # Apply bounds if needed
                y[i] = model.apply_bounds(y[i])
                
                nfev += 4  # Four function evaluations per step
                
            except Exception as e:
                error_context = f"at t={t:.6f}, step {i}/{len(t_eval)}, dt={dt}"
                return {
                    't': t_eval[:i],
                    'y': y[:i].T,
                    'success': False,
                    'message': f'RK4 integration failed {error_context}: {str(e)}',
                    'nfev': nfev
                }
        
        return {
            't': t_eval,
            'y': y.T,
            'success': True,
            'message': 'Integration completed successfully',
            'nfev': nfev
        }

class AdaptiveMethod(IntegrationMethod):
    """Adaptive step-size integration using SciPy"""
    
    def __init__(self, method_name: str = 'RK45'):
        self.scipy_method = method_name
    
    def get_name(self) -> str:
        return f"Adaptive-{self.scipy_method}"
    
    def integrate(self, model: 'Model', t_span: tuple, y0: np.ndarray, **kwargs) -> dict:
        """Adaptive step-size integration using SciPy"""
        method = kwargs.get('method', self.scipy_method)
        rtol = kwargs.get('rtol', 1e-6)
        atol = kwargs.get('atol', 1e-9)
        max_step = kwargs.get('max_step', np.inf)
        
        def derivatives_wrapper(t, y):
            """Wrapper for model derivatives that handles bounds"""
            try:
                # Apply bounds before computing derivatives
                y_bounded = model.apply_bounds(y)
                return model.compute_derivatives(t, y_bounded)
            except Exception as e:
                # Return zero derivatives if computation fails
                warnings.warn(f"Derivative computation failed at t={t}: {e}")
                return np.zeros_like(y)
        
        try:
            result = solve_ivp(
                derivatives_wrapper,
                t_span,
                y0,
                method=method,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
                dense_output=True
            )
            
            return result
            
        except Exception as e:
            return {
                't': np.array([t_span[0]]),
                'y': y0.reshape(-1, 1),
                'success': False,
                'message': f'Adaptive integration failed: {str(e)}'
            }

@dataclass
class ModelCharacteristics:
    """Container for model analysis results"""
    is_stiff: bool = False
    is_large: bool = False
    has_discontinuities: bool = False
    max_eigenvalue_ratio: float = 1.0
    num_states: int = 0
    has_fast_dynamics: bool = False
    has_constraints: bool = False

class ModelAnalyzer:
    """Analyzes model characteristics to recommend optimal integration method"""
    
    @staticmethod
    def analyze_model_characteristics(model: 'Model') -> ModelCharacteristics:
        """Analyze model to determine optimal integration method"""
        characteristics = ModelCharacteristics()
        
        # Count total states
        total_states = sum(stock.values.size for stock in model.stocks)
        characteristics.num_states = total_states
        characteristics.is_large = total_states > 100  # Threshold for "large" models
        
        # Check for discontinuities (simplified - would need formula parsing in full implementation)
        has_discontinuities = False
        has_constraints = False
        has_fast_dynamics = False
        
        for stock in model.stocks:
            # Check for constraints (bounds)
            if stock.min_value > -np.inf or stock.max_value < np.inf:
                has_constraints = True
            
            # Check flows for potential discontinuities or fast dynamics
            for flow in stock.inflows.values():
                if hasattr(flow, 'rate_expression') and callable(flow.rate_expression):
                    # This is a heuristic - in practice would analyze the actual expressions
                    try:
                        rate = flow.get_rate()
                        if isinstance(rate, np.ndarray):
                            # Check for large rate values that might indicate fast dynamics
                            if np.any(np.abs(rate) > 100):  # Heuristic threshold
                                has_fast_dynamics = True
                    except:
                        pass
            
            for flow in stock.outflows.values():
                if hasattr(flow, 'rate_expression') and callable(flow.rate_expression):
                    try:
                        rate = flow.get_rate()
                        if isinstance(rate, np.ndarray):
                            if np.any(np.abs(rate) > 100):
                                has_fast_dynamics = True
                    except:
                        pass
        
        characteristics.has_discontinuities = has_discontinuities
        characteristics.has_constraints = has_constraints
        characteristics.has_fast_dynamics = has_fast_dynamics
        
        # Estimate stiffness (simplified heuristic)
        characteristics.is_stiff = ModelAnalyzer._estimate_stiffness(model)
        
        return characteristics
    
    @staticmethod
    def _estimate_stiffness(model: 'Model') -> bool:
        """Improved stiffness estimation using Jacobian analysis"""
        try:
            # Get current state
            y0 = model.get_flattened_state()
            if len(y0) < 2:
                return False  # Need at least 2 states for stiffness
            
            # Compute Jacobian numerically
            epsilon = 1e-8
            t = model.time
            f0 = model.compute_derivatives(t, y0)
            eigenvalue_estimates = []
            
            # Sample a few partial derivatives to estimate eigenvalues
            for i in range(min(len(y0), 5)):  # Sample max 5 states for efficiency
                y_pert = y0.copy()
                y_pert[i] += epsilon
                f_pert = model.compute_derivatives(t, y_pert)
                eigenvalue_estimates.append(abs((f_pert[i] - f0[i]) / epsilon))
            
            # Check for large eigenvalue spread (stiffness indicator)
            if len(eigenvalue_estimates) > 1:
                max_eigen = max(eigenvalue_estimates)
                min_eigen = min([e for e in eigenvalue_estimates if e > 1e-12])
                if min_eigen > 0:
                    stiffness_ratio = max_eigen / min_eigen
                    return stiffness_ratio > 1000  # Stiff if ratio > 1000
            
            return False
        except Exception:
            return False  # Conservative: assume not stiff if analysis fails   
        
    @staticmethod
    def recommend_integration_method(model: 'Model') -> Tuple[str, dict]:
        """Recommend optimal integration method and parameters"""
        chars = ModelAnalyzer.analyze_model_characteristics(model)
        
        if chars.is_stiff:
            return 'lsoda', {
                'rtol': 1e-6, 
                'atol': 1e-9,
                'max_step': model.dt * 10
            }
        elif chars.is_large:
            return 'bdf', {
                'rtol': 1e-5, 
                'atol': 1e-8,
                'max_step': model.dt * 5
            }
        elif chars.has_discontinuities or chars.has_constraints:
            return 'rk4', {
                'dt': model.dt / 2  # Smaller timestep for discontinuities
            }
        elif chars.has_fast_dynamics:
            return 'rk45', {
                'rtol': 1e-7,
                'atol': 1e-10,
                'max_step': model.dt
            }
        else:
            return 'rk4', {
                'dt': model.dt
            }

class IntegrationEngine:
    """Main integration engine that manages all methods"""
    
    def __init__(self):
        self.methods = {
            'euler': EulerMethod(),
            'rk4': RK4Method(),
            'rk45': AdaptiveMethod('RK45'),
            'lsoda': AdaptiveMethod('LSODA'),
            'bdf': AdaptiveMethod('BDF'),
            'dopri5': AdaptiveMethod('DOP853'),
            'radau': AdaptiveMethod('Radau')
        }
        self.last_analysis = None
        self.integration_stats = {}
    
    def get_available_methods(self) -> List[str]:
        """Get list of available integration methods"""
        return list(self.methods.keys())
    
    def integrate(self, model: 'Model', method: str = 'auto', **kwargs) -> dict:
        """Run simulation with specified integration method"""
        
        # Auto-select method if requested
        if method == 'auto':
            method, auto_params = ModelAnalyzer.recommend_integration_method(model)
            print(f"Auto-selected integration method: {method}")
            # Merge auto parameters with user parameters (user parameters take precedence)
            merged_kwargs = {**auto_params, **kwargs}
            kwargs = merged_kwargs
        
        if method not in self.methods:
            available = ', '.join(self.get_available_methods())
            raise ValueError(f"Unknown integration method '{method}'. Available: {available}")
        
        # Prepare for integration
        y0 = model.get_flattened_state()
        t_span = (model.time, kwargs.get('end_time', model.time + 10.0))
        
        print(f"Starting integration with {method.upper()} method")
        print(f"  Initial state size: {len(y0)}")
        print(f"  Time span: {t_span[0]:.3f} to {t_span[1]:.3f}")
        
        # Run integration
        start_time = time.time()
        result = self.methods[method].integrate(model, t_span, y0, **kwargs)
        integration_time = time.time() - start_time
        
        # Store statistics
        self.integration_stats[method] = {
            'last_runtime': integration_time,
            'last_success': result['success'],
            'last_nfev': result.get('nfev', 0),
            'last_steps': len(result.get('t', [])),
            'efficiency': result.get('nfev', len(result.get('t', []))) / max(1, integration_time)
        }
        
        # Update results with metadata
        if result['success']:
            model.update_from_flattened_results(result)

            # Add this right after model.update_from_flattened_results(result)
            validation = self._validate_integration_result(result, model)
            result['validation'] = validation
            
            if not validation['valid']:
                print(f"⚠️  Integration validation failed:")
                for error in validation['errors']:
                    print(f"     Error: {error}")
                
            if validation['warnings']:
                for warning in validation['warnings']:
                    print(f"     Warning: {warning}")
            
            result['integration_method'] = method
            result['integration_time'] = integration_time
            result['method_name'] = self.methods[method].get_name()
            
            print(f"Integration completed successfully:")
            print(f"  Runtime: {integration_time:.3f}s")
            print(f"  Steps: {len(result['t'])}")
            print(f"  Function evaluations: {result.get('nfev', 'N/A')}")
        else:
            print(f"Integration failed: {result.get('message', 'Unknown error')}")
        
        return result
    
    def compare_methods(self, model: 'Model', methods: List[str] = None, 
                       duration: float = 10.0, **kwargs) -> Dict[str, dict]:
        """Compare performance of different integration methods"""
        
        if methods is None:
            methods = ['euler', 'rk4', 'rk45']
        
        print(f"Comparing integration methods: {', '.join(methods)}")
        results = {}
        
        for method in methods:
            if method not in self.methods:
                print(f"Skipping unknown method: {method}")
                continue
                
            print(f"\nTesting {method.upper()}...")
            
            # Reset model to initial state
            model.reset()
            
            # Run integration
            try:
                result = self.integrate(
                    model, 
                    method=method, 
                    end_time=model.time + duration,
                    **kwargs
                )
                
                # Calculate accuracy metrics (if reference solution available)
                accuracy_metrics = self._calculate_accuracy_metrics(result)
                
                results[method] = {
                    'success': result['success'],
                    'runtime': result.get('integration_time', 0),
                    'steps': len(result.get('t', [])),
                    'nfev': result.get('nfev', 0),
                    'final_time': result.get('t', [0])[-1],
                    'accuracy': accuracy_metrics,
                    'efficiency': result.get('nfev', len(result.get('t', []))) / max(0.001, result.get('integration_time', 0.001))
                }
                
            except Exception as e:
                results[method] = {
                    'success': False,
                    'error': str(e),
                    'runtime': 0,
                    'steps': 0,
                    'nfev': 0
                }
        
        # Print comparison summary
        self._print_comparison_summary(results)
        
        return results
    
    def _calculate_accuracy_metrics(self, result: dict) -> dict:
        """Calculate accuracy metrics (simplified version)"""
        # In a full implementation, this would compare against analytical solutions
        # or high-precision reference solutions
        
        if not result['success'] or 't' not in result:
            return {'error': 'No solution available'}
        
        # Simple metrics based on solution properties
        y = result['y']
        
        return {
            'mass_conservation_error': self._check_mass_conservation(y),
            'smoothness': self._check_solution_smoothness(y),
            'final_values': y[:, -1] if y.size > 0 else []
        }
    
    def _check_mass_conservation(self, y: np.ndarray) -> float:
        """Check mass conservation (simplified)"""
        if y.size == 0:
            return float('inf')
        
        # Calculate total mass at each time step
        total_mass = np.sum(y, axis=0)
        
        # Check conservation
        mass_variation = np.std(total_mass) / (np.mean(total_mass) + 1e-12)
        
        return mass_variation
    
    def _check_solution_smoothness(self, y: np.ndarray) -> float:
        """Check solution smoothness"""
        if y.size == 0 or y.shape[1] < 3:
            return float('inf')
        
        # Calculate second derivatives (curvature)
        second_deriv = np.diff(y, n=2, axis=1)
        smoothness = np.mean(np.abs(second_deriv))
        
        return smoothness
    
    def _print_comparison_summary(self, results: Dict[str, dict]):
        """Print a formatted comparison summary"""
        print(f"\n{'='*80}")
        print("INTEGRATION METHOD COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        print(f"{'Method':<15} {'Success':<8} {'Runtime(s)':<12} {'Steps':<8} {'NFEv':<8} {'Efficiency':<12}")
        print("-" * 80)
        
        for method, result in results.items():
            success = "✓" if result['success'] else "✗"
            runtime = f"{result['runtime']:.4f}" if result['runtime'] > 0 else "N/A"
            steps = str(result['steps'])
            nfev = str(result['nfev'])
            efficiency = f"{result.get('efficiency', 0):.1f}" if result['success'] else "N/A"
            
            print(f"{method:<15} {success:<8} {runtime:<12} {steps:<8} {nfev:<8} {efficiency:<12}")
        
        # Recommend best method
        successful_methods = {k: v for k, v in results.items() if v['success']}
        if successful_methods:
            best_method = min(successful_methods.keys(), 
                            key=lambda k: successful_methods[k]['runtime'])
            print(f"\nRecommended method based on runtime: {best_method.upper()}")
            
            
    def _validate_integration_result(self, result: dict, model: 'Model') -> dict:
        """Validate integration results for common problems"""
        validation = {'valid': True, 'warnings': [], 'errors': []}
        
        if not result.get('success', False):
            validation['valid'] = False
            validation['errors'].append(f"Integration failed: {result.get('message', 'Unknown error')}")
            return validation
        
        try:
            t = result.get('t', [])
            y = result.get('y', np.array([]))
            
            # Check 1: Non-empty results
            if len(t) < 2:
                validation['errors'].append("Integration produced insufficient time points")
                validation['valid'] = False
                return validation
            
            # Check 2: Finite values
            if not np.all(np.isfinite(y)):
                validation['errors'].append("Integration produced non-finite values (NaN/Inf)")
                validation['valid'] = False
                return validation
            
            # Check 3: Reasonable value ranges
            max_value = np.max(np.abs(y))
            if max_value > 1e12:
                validation['warnings'].append(f"Very large values detected: {max_value:.2e}")
            
            # Check 4: Monotonic time
            if not np.all(np.diff(t) > 0):
                validation['errors'].append("Time values are not monotonically increasing")
                validation['valid'] = False
            
            # Check 5: Negative values in stocks (if bounds exist)
            min_value = np.min(y)
            if min_value < -1e-10:  # Allow small numerical errors
                validation['warnings'].append(f"Negative values detected: {min_value:.2e}")
            
        except Exception as e:
            validation['errors'].append(f"Result validation failed: {str(e)}")
            validation['valid'] = False
        
        return validation            

# Extensions to the existing Model class to support advanced integration
class ModelIntegrationExtensions:
    """Extensions for the Model class to support advanced integration"""
    
    @staticmethod
    def add_integration_support(model_instance):
        """Add integration support to existing Model instance"""
        
        def compute_derivatives(self, t: float, state_vector: np.ndarray) -> np.ndarray:
            """Compute derivatives for integration engine"""
            # Update model time
            old_time = self.time
            self.time = t
            
            # Set model state from flattened vector
            self.set_flattened_state(state_vector)
            
            # Compute derivatives
            derivatives = np.zeros_like(state_vector)
            state_index = 0
            
            for stock in self.stocks:
                stock_size = stock.values.size if hasattr(stock.values, 'size') else 1
                stock_derivatives = np.zeros_like(stock.values).flatten()
                
                # Calculate net flow for this stock
                try:
                    net_flow = stock.get_net_flow(self.dt)
                    stock_derivatives = net_flow.flatten() if hasattr(net_flow, 'flatten') else [net_flow]
                except Exception as e:
                    warnings.warn(f"Error computing derivatives for stock {stock.name}: {e}")
                    stock_derivatives = np.zeros(stock_size)
                
                # Add to derivatives vector
                derivatives[state_index:state_index + stock_size] = stock_derivatives
                state_index += stock_size
            
            # Restore original time
            self.time = old_time
            
            return derivatives
        
        def get_flattened_state(self) -> np.ndarray:
            """Get current state as flattened array for integration"""
            state_parts = []
            for stock in self.stocks:
                if hasattr(stock.values, 'flatten'):
                    state_parts.append(stock.values.flatten())
                else:
                    state_parts.append(np.array([stock.values]))
            
            return np.concatenate(state_parts) if state_parts else np.array([])
        
        def set_flattened_state(self, state_vector: np.ndarray):
            """Set model state from flattened array"""
            index = 0
            for stock in self.stocks:
                if hasattr(stock.values, 'size'):
                    stock_size = stock.values.size
                    stock_shape = stock.values.shape
                else:
                    stock_size = 1
                    stock_shape = ()
                
                # Extract stock values from vector
                stock_values = state_vector[index:index + stock_size]
                
                # Reshape and assign
                if stock_shape:
                    stock.values = stock_values.reshape(stock_shape)
                else:
                    stock.values = stock_values[0] if len(stock_values) > 0 else 0.0
                
                index += stock_size
        
        def apply_bounds(self, state_vector: np.ndarray) -> np.ndarray:
            """Apply stock bounds to state vector"""
            bounded_state = state_vector.copy()
            index = 0
            
            for stock in self.stocks:
                stock_size = stock.values.size if hasattr(stock.values, 'size') else 1
                
                # Apply bounds
                stock_values = bounded_state[index:index + stock_size]
                if hasattr(stock, 'min_value') and hasattr(stock, 'max_value'):
                    stock_values = np.clip(stock_values, stock.min_value, stock.max_value)
                    bounded_state[index:index + stock_size] = stock_values
                
                index += stock_size
            
            return bounded_state
        
        def update_from_flattened_results(self, results: dict):
            """Update model with simulation results"""
            if 'y' in results and results['success']:
                # Take the final state
                final_state = results['y'][:, -1]
                self.set_flattened_state(final_state)
                
                # Update time to final time
                if 't' in results:
                    self.time = results['t'][-1]
        
        # Add methods to model instance
        import types
        model_instance.compute_derivatives = types.MethodType(compute_derivatives, model_instance)
        model_instance.get_flattened_state = types.MethodType(get_flattened_state, model_instance)
        model_instance.set_flattened_state = types.MethodType(set_flattened_state, model_instance)
        model_instance.apply_bounds = types.MethodType(apply_bounds, model_instance)
        model_instance.update_from_flattened_results = types.MethodType(update_from_flattened_results, model_instance)
        
        # Add integration engine
        model_instance.integration_engine = IntegrationEngine()
        
        return model_instance

# Convenience function to create enhanced model with integration support
def create_enhanced_model(*args, **kwargs):
    """Create a model with advanced integration support"""
    from simulation import Model  # Import your existing Model class
    
    model = Model(*args, **kwargs)
    return ModelIntegrationExtensions.add_integration_support(model)

# Example usage and testing functions
def test_integration_methods():
    """Test function demonstrating the integration methods"""
    
    # This would use your existing simulation classes
    print("Integration Engine Test")
    print("======================")
    
    # Create a simple test model (you would replace this with your actual model)
    # For demonstration, showing how it would work:
    
    engine = IntegrationEngine()
    print(f"Available methods: {engine.get_available_methods()}")
    
    # Show analysis capabilities
    analyzer = ModelAnalyzer()
    chars = ModelCharacteristics()
    chars.num_states = 5
    chars.is_large = False
    chars.has_discontinuities = False
    chars.is_stiff = False
    
    print(f"Example model characteristics: {chars}")
    
    return engine

if __name__ == "__main__":
    test_integration_methods()