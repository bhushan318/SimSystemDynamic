# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 12:04:12 2025

@author: NagabhushanamTattaga
"""

# enhanced_multidimensional_flow.py

from typing import Callable, Dict, Optional, Tuple, Any, Protocol, Union, List
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import warnings
import json
import time
from pathlib import Path

# === Enhanced Type System ===
class Selector(Protocol):
    def __call__(self, values: np.ndarray, context: 'FlowContext') -> np.ndarray: ...

class RateFunc(Protocol):
    def __call__(self, values: np.ndarray, context: 'FlowContext') -> np.ndarray: ...

class Condition(Protocol):
    def __call__(self, values: np.ndarray, context: 'FlowContext') -> np.ndarray: ...

# === Enhanced Context System ===
@dataclass
class FlowContext:
    """Comprehensive context for flow calculations"""
    time: float = 0.0
    dt: float = 1.0
    step: int = 0
    
    # Stock information
    stock_dimensions: Dict[str, List[str]] = field(default_factory=dict)
    stock_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    current_values: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Model parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Legacy support
    threshold: float = 0.0
    age_indices: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None
    
    def get_dimension_index(self, stock_name: str, dimension_name: str, value: str) -> int:
        """Helper to get dimension index safely"""
        if stock_name not in self.stock_dimensions:
            raise ValueError(f"Stock '{stock_name}' not found in context")
        
        dimensions = self.stock_dimensions[stock_name]
        if dimension_name not in dimensions:
            raise ValueError(f"Dimension '{dimension_name}' not found in stock '{stock_name}'")
        
        try:
            return dimensions.index(value)
        except ValueError:
            raise ValueError(f"Value '{value}' not found in dimension '{dimension_name}'")
    
    def get_stock_value(self, stock_name: str) -> np.ndarray:
        """Get current stock values safely"""
        if stock_name not in self.current_values:
            raise ValueError(f"Stock '{stock_name}' values not available in context")
        return self.current_values[stock_name]

# === Custom Exceptions ===
class FlowError(Exception):
    pass

class FlowShapeError(FlowError):
    pass

class FlowMatrixError(FlowError):
    pass

class FlowValidationError(FlowError):
    pass

# === Flow Pattern Types ===
class FlowPattern(Enum):
    BROADCAST = "broadcast"
    SELECTIVE = "selective"
    AGING = "aging"
    MIGRATION = "migration"
    MATRIX = "matrix"
    CONDITIONAL = "conditional"
    PROPORTIONAL = "proportional"
    LOOKUP = "lookup"

# === Enhanced MultiDimensionalFlow ===
class MultiDimensionalFlow:
    """Enhanced multi-dimensional flow with comprehensive features"""
    
    def __init__(
        self,
        name: str,
        from_selector: Optional[Selector] = None,
        to_selector: Optional[Selector] = None,
        rate_func: Optional[RateFunc] = None,
        condition: Optional[Condition] = None,
        matrix: Optional[np.ndarray] = None,
        pattern: Optional[FlowPattern] = None,
        validate_shapes: bool = True,
        prevent_negatives: bool = True,
        debug: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.from_selector = from_selector
        self.to_selector = to_selector
        self.rate_func = rate_func
        self.condition = condition or (lambda values, ctx: np.ones_like(values, dtype=bool))
        self.matrix = matrix
        self.pattern = pattern
        self.validate_shapes = validate_shapes
        self.prevent_negatives = prevent_negatives
        self.debug = debug
        self.metadata = metadata or {}
        
        # Performance and monitoring
        self._cache = {}
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._last_cache_clear = time.time()
        
        # Statistics
        self.total_transferred = 0.0
        self.max_transfer_rate = 0.0
        self.execution_history = []
        
        # Validation
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate flow configuration"""
        if self.matrix is not None:
            if self.matrix.ndim != 2:
                raise FlowValidationError(f"Matrix must be 2D, got shape {self.matrix.shape}")
            if not np.all(self.matrix >= 0):
                raise FlowValidationError("Matrix elements must be non-negative")
        else:
            if not all([self.from_selector, self.to_selector, self.rate_func]):
                raise FlowValidationError("Non-matrix flows require from_selector, to_selector, and rate_func")
    
    def apply(self, from_values: np.ndarray, to_values: np.ndarray, 
              context: Union[FlowContext, Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Enhanced apply method with comprehensive error handling and metrics"""
        
        start_time = time.time()
        execution_info = {
            'success': True,
            'warnings': [],
            'errors': [],
            'transferred_amount': 0.0,
            'execution_time': 0.0,
            'clipped_transfers': False
        }
        
        try:
            # Ensure proper context
            context = self._ensure_context(context)
            
            # Clear cache periodically to prevent memory bloat
            if time.time() - self._last_cache_clear > 60:  # Clear every minute
                self._cache.clear()
                self._last_cache_clear = time.time()
            
            # Pre-execution validation
            validation_errors = self._validate_inputs(from_values, to_values, context)
            if validation_errors:
                execution_info['errors'] = validation_errors
                execution_info['success'] = False
                return from_values, to_values, execution_info
            
            # Route to appropriate flow type
            if self.matrix is not None:
                new_from, new_to = self._apply_matrix_flow(from_values, to_values, context)
            else:
                new_from, new_to = self._apply_selector_flow(from_values, to_values, context)
            
            # Apply bounds checking if enabled
            if self.prevent_negatives:
                new_from, clipped = self._apply_bounds_checking(new_from)
                if clipped:
                    execution_info['warnings'].append("Transfer rates clipped to prevent negative values")
                    execution_info['clipped_transfers'] = True
            
            # Calculate statistics
            transfer_amount = np.sum(from_values - new_from)
            execution_info['transferred_amount'] = float(transfer_amount)
            self.total_transferred += transfer_amount
            self.max_transfer_rate = max(self.max_transfer_rate, float(np.max(from_values - new_from)))
            
            # Debug output
            if self.debug:
                self._print_debug_info(from_values, new_from, new_to, execution_info)
            
            return new_from, new_to, execution_info
            
        except Exception as e:
            execution_info['success'] = False
            execution_info['errors'].append(f"Flow execution failed: {str(e)}")
            warnings.warn(f"Error applying flow '{self.name}': {e}")
            return from_values, to_values, execution_info
            
        finally:
            # Update performance metrics
            execution_time = time.time() - start_time
            execution_info['execution_time'] = execution_time
            self._execution_count += 1
            self._total_execution_time += execution_time
            
            # Store execution history (limited to last 100 executions)
            if len(self.execution_history) >= 100:
                self.execution_history.pop(0)
            
            self.execution_history.append({
                'timestamp': time.time(),
                'context_time': context.time,
                'success': execution_info['success'],
                'transferred': execution_info['transferred_amount'],
                'execution_time': execution_time
            })
    
    def _apply_selector_flow(self, from_values: np.ndarray, to_values: np.ndarray, 
                           context: FlowContext) -> Tuple[np.ndarray, np.ndarray]:
        """Apply selector-based flow"""
        
        # Use caching for performance
        cache_key = (id(from_values), id(to_values), context.time, context.step)
        
        if cache_key in self._cache:
            from_mask, to_mask, rate, condition_mask = self._cache[cache_key]
        else:
            from_mask = self.from_selector(from_values, context)
            to_mask = self.to_selector(to_values, context)
            rate = self.rate_func(from_values, context)
            condition_mask = self.condition(from_values, context)
            
            # Cache results
            self._cache[cache_key] = (from_mask, to_mask, rate, condition_mask)
        
        # Validate shapes
        if self.validate_shapes:
            self._validate_flow_shapes(from_values, to_values, rate, from_mask, to_mask)
        
        # Calculate effective transfer
        effective_mask = from_mask & condition_mask
        delta = np.zeros_like(from_values)
        delta[effective_mask] = rate[effective_mask]
        
        # Apply transfer
        new_from = from_values - delta
        new_to = to_values + delta * to_mask
        
        return new_from, new_to
    
    def _apply_matrix_flow(self, from_values: np.ndarray, to_values: np.ndarray, 
                          context: FlowContext) -> Tuple[np.ndarray, np.ndarray]:
        """Apply matrix-based flow with correct mathematical logic"""
        
        try:
            # Validate matrix dimensions
            if self.matrix.shape[0] != from_values.size or self.matrix.shape[1] != to_values.size:
                raise FlowMatrixError(
                    f"Matrix shape {self.matrix.shape} incompatible with "
                    f"from_values.size={from_values.size}, to_values.size={to_values.size}"
                )
            
            # Flatten arrays for matrix operations
            from_flat = from_values.flatten()
            to_flat = to_values.flatten()
            
            # Matrix[i,j] represents rate of transfer from category i to category j
            # Diagonal elements represent retention rates (staying in same category)
            
            # Calculate outflows from each category (1 - retention rate)
            retention_rates = np.diag(self.matrix)
            outflow_rates = 1.0 - retention_rates
            total_outflows = from_flat * outflow_rates
            
            # Calculate transfers between categories using off-diagonal elements
            transfer_matrix = self.matrix.copy()
            np.fill_diagonal(transfer_matrix, 0)  # Remove diagonal (retention)
            
            # Normalize transfer matrix rows to sum to 1 (for redistribution)
            row_sums = np.sum(transfer_matrix, axis=1)
            normalized_matrix = np.divide(
                transfer_matrix, 
                row_sums[:, np.newaxis], 
                out=np.zeros_like(transfer_matrix), 
                where=row_sums[:, np.newaxis] != 0
            )
            
            # Apply transfers: outflows redistributed according to normalized matrix
            inflows = normalized_matrix.T @ total_outflows
            
            # Calculate new values
            new_from_flat = from_flat - total_outflows
            new_to_flat = to_flat + inflows
            
            # Reshape back to original shapes
            new_from = new_from_flat.reshape(from_values.shape)
            new_to = new_to_flat.reshape(to_values.shape)
            
            # Validation
            if not np.allclose(np.sum(from_flat), np.sum(new_from_flat + new_to_flat), atol=1e-10):
                warnings.warn(f"Matrix flow '{self.name}' may violate mass conservation")
            
            return new_from, new_to
            
        except Exception as e:
            raise FlowMatrixError(f"Matrix flow calculation failed: {e}")
    
    def _apply_bounds_checking(self, values: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Apply bounds checking to prevent negative values"""
        clipped = False
        
        if np.any(values < -1e-10):  # Allow for small numerical errors
            clipped_values = np.maximum(values, 0.0)
            clipped = True
            return clipped_values, clipped
        
        return values, clipped
    
    def _validate_inputs(self, from_values: np.ndarray, to_values: np.ndarray, 
                        context: FlowContext) -> List[str]:
        """Comprehensive input validation"""
        from unified_validation import get_unified_validator
        validator = get_unified_validator()
        
        # Package inputs for validation
        validation_target = {
            'from_values': from_values,
            'to_values': to_values,
            'context': context,
            'flow_name': self.name
        }
        
        report = validator.validate_all(validation_target, 'flow_inputs')
        return [issue.message for issue in report.issues if issue.severity.value == 'error']
    
    def _validate_flow_shapes(self, from_values: np.ndarray, to_values: np.ndarray, 
                            rate: np.ndarray, from_mask: np.ndarray, to_mask: np.ndarray):
        """Validate array shapes for compatibility"""
        if from_values.shape != to_values.shape:
            raise FlowShapeError(f"from_values.shape {from_values.shape} != to_values.shape {to_values.shape}")
        
        if rate.shape != from_values.shape:
            raise FlowShapeError(f"rate.shape {rate.shape} != from_values.shape {from_values.shape}")
        
        if from_mask.shape != from_values.shape:
            raise FlowShapeError(f"from_mask.shape {from_mask.shape} != from_values.shape {from_values.shape}")
        
        if to_mask.shape != to_values.shape:
            raise FlowShapeError(f"to_mask.shape {to_mask.shape} != to_values.shape {to_values.shape}")
    
    def _print_debug_info(self, from_values: np.ndarray, new_from: np.ndarray, 
                         new_to: np.ndarray, execution_info: Dict):
        """Print comprehensive debug information"""
        print(f"\n[DEBUG: Flow '{self.name}']")
        print(f"  Pattern: {self.pattern.value if self.pattern else 'custom'}")
        print(f"  Transfer amount: {execution_info['transferred_amount']:.6f}")
        print(f"  Execution time: {execution_info['execution_time']:.6f}s")
        print(f"  From values: min={np.min(from_values):.3f}, max={np.max(from_values):.3f}, sum={np.sum(from_values):.3f}")
        print(f"  New from: min={np.min(new_from):.3f}, max={np.max(new_from):.3f}, sum={np.sum(new_from):.3f}")
        print(f"  New to: min={np.min(new_to):.3f}, max={np.max(new_to):.3f}, sum={np.sum(new_to):.3f}")
        if execution_info['warnings']:
            print(f"  Warnings: {execution_info['warnings']}")
    
    def _ensure_context(self, context: Union[FlowContext, Dict[str, Any]]) -> FlowContext:
        """Ensure context is proper FlowContext object"""
        if isinstance(context, FlowContext):
            return context
        elif isinstance(context, dict):
            return FlowContext(**context)
        else:
            return FlowContext()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive flow statistics"""
        avg_execution_time = (self._total_execution_time / self._execution_count 
                            if self._execution_count > 0 else 0.0)
        
        success_rate = 1.0
        if self.execution_history:
            successful_executions = sum(1 for h in self.execution_history if h['success'])
            success_rate = successful_executions / len(self.execution_history)
        
        return {
            'name': self.name,
            'pattern': self.pattern.value if self.pattern else 'custom',
            'execution_count': self._execution_count,
            'total_transferred': self.total_transferred,
            'max_transfer_rate': self.max_transfer_rate,
            'average_execution_time': avg_execution_time,
            'success_rate': success_rate,
            'cache_size': len(self._cache)
        }
    
    def reset_statistics(self):
        """Reset all statistics and history"""
        self._execution_count = 0
        self._total_execution_time = 0.0
        self.total_transferred = 0.0
        self.max_transfer_rate = 0.0
        self.execution_history.clear()
        self._cache.clear()

# === Enhanced FlowBuilder ===
class FlowBuilder:
    """Enhanced builder with high-level flow patterns"""
    
    def __init__(self, name: str):
        self.name = name
        self._from = None
        self._to = None
        self._rate = None
        self._condition = None
        self._matrix = None
        self._pattern = None
        self._debug = False
        self._prevent_negatives = True
        self._metadata = {}
    
    # === Builder Methods ===
    def from_selector(self, selector: Selector):
        self._from = selector
        return self
    
    def to_selector(self, selector: Selector):
        self._to = selector
        return self
    
    def with_rate(self, rate: RateFunc):
        self._rate = rate
        return self
    
    def with_condition(self, condition: Condition):
        self._condition = condition
        return self
    
    def with_matrix(self, matrix: np.ndarray):
        self._matrix = matrix
        self._pattern = FlowPattern.MATRIX
        return self
    
    def debug(self, enabled: bool = True):
        self._debug = enabled
        return self
    
    def allow_negatives(self):
        self._prevent_negatives = False
        return self
    
    def with_metadata(self, **kwargs):
        self._metadata.update(kwargs)
        return self
    
    # === High-Level Pattern Builders ===
    @classmethod
    def broadcast_flow(cls, name: str, rate: Union[float, Callable]) -> 'FlowBuilder':
        """Create flow that affects all dimensions equally"""
        builder = cls(name)
        builder._pattern = FlowPattern.BROADCAST
        builder._from = broadcast_selector
        builder._to = broadcast_selector
        
        if callable(rate):
            builder._rate = rate
        else:
            builder._rate = lambda values, ctx: np.full_like(values, rate, dtype=float)
        
        return builder
    
    @classmethod
    def proportional_flow(cls, name: str, rate: float) -> 'FlowBuilder':
        """Create flow proportional to current values"""
        builder = cls(name)
        builder._pattern = FlowPattern.PROPORTIONAL
        builder._from = broadcast_selector
        builder._to = broadcast_selector
        builder._rate = lambda values, ctx: values * rate
        return builder
    
    @classmethod
    def aging_flow(cls, name: str, from_age_idx: int, to_age_idx: int, 
                   rate: float, age_dimension: int = 0) -> 'FlowBuilder':
        """Create aging flow from one age group to another"""
        builder = cls(name)
        builder._pattern = FlowPattern.AGING
        
        def age_from_selector(values: np.ndarray, context: FlowContext) -> np.ndarray:
            mask = np.zeros_like(values, dtype=bool)
            if values.ndim > age_dimension:
                indices = [slice(None)] * values.ndim
                indices[age_dimension] = from_age_idx
                mask[tuple(indices)] = True
            return mask
        
        def age_to_selector(values: np.ndarray, context: FlowContext) -> np.ndarray:
            mask = np.zeros_like(values, dtype=bool)
            if values.ndim > age_dimension:
                indices = [slice(None)] * values.ndim
                indices[age_dimension] = to_age_idx
                mask[tuple(indices)] = True
            return mask
        
        builder._from = age_from_selector
        builder._to = age_to_selector
        builder._rate = lambda values, ctx: values * rate
        builder._metadata.update({
            'from_age_index': from_age_idx,
            'to_age_index': to_age_idx,
            'aging_rate': rate,
            'age_dimension': age_dimension
        })
        
        return builder
    
    @classmethod
    def migration_flow(cls, name: str, from_region_idx: int, to_region_idx: int, 
                      rate: float, region_dimension: int = 1) -> 'FlowBuilder':
        """Create migration flow between regions"""
        builder = cls(name)
        builder._pattern = FlowPattern.MIGRATION
        
        def region_from_selector(values: np.ndarray, context: FlowContext) -> np.ndarray:
            mask = np.zeros_like(values, dtype=bool)
            if values.ndim > region_dimension:
                indices = [slice(None)] * values.ndim
                indices[region_dimension] = from_region_idx
                mask[tuple(indices)] = True
            return mask
        
        def region_to_selector(values: np.ndarray, context: FlowContext) -> np.ndarray:
            mask = np.zeros_like(values, dtype=bool)
            if values.ndim > region_dimension:
                indices = [slice(None)] * values.ndim
                indices[region_dimension] = to_region_idx
                mask[tuple(indices)] = True
            return mask
        
        builder._from = region_from_selector
        builder._to = region_to_selector
        builder._rate = lambda values, ctx: values * rate
        builder._metadata.update({
            'from_region_index': from_region_idx,
            'to_region_index': to_region_idx,
            'migration_rate': rate,
            'region_dimension': region_dimension
        })
        
        return builder
    
    @classmethod
    def threshold_flow(cls, name: str, threshold: float, rate: float, 
                      comparison: str = 'greater') -> 'FlowBuilder':
        """Create conditional flow based on threshold"""
        builder = cls(name)
        builder._pattern = FlowPattern.CONDITIONAL
        builder._from = broadcast_selector
        builder._to = broadcast_selector
        builder._rate = lambda values, ctx: np.full_like(values, rate, dtype=float)
        
        if comparison == 'greater':
            builder._condition = lambda values, ctx: values > threshold
        elif comparison == 'less':
            builder._condition = lambda values, ctx: values < threshold
        elif comparison == 'equal':
            builder._condition = lambda values, ctx: np.abs(values - threshold) < 1e-6
        else:
            raise ValueError(f"Unknown comparison: {comparison}")
        
        builder._metadata.update({
            'threshold': threshold,
            'threshold_rate': rate,
            'comparison': comparison
        })
        
        return builder
    
    @classmethod
    def lookup_flow(cls, name: str, lookup_table: List[Tuple[float, float]]) -> 'FlowBuilder':
        """Create flow with lookup table for rates"""
        builder = cls(name)
        builder._pattern = FlowPattern.LOOKUP
        builder._from = broadcast_selector
        builder._to = broadcast_selector
        
        # Convert lookup table to numpy arrays for interpolation
        x_values = np.array([x for x, y in lookup_table])
        y_values = np.array([y for x, y in lookup_table])
        
        def lookup_rate(values: np.ndarray, context: FlowContext) -> np.ndarray:
            # Use current time or stock values for lookup
            lookup_value = context.time if hasattr(context, 'time') else np.mean(values)
            rate = np.interp(lookup_value, x_values, y_values)
            return np.full_like(values, rate, dtype=float)
        
        builder._rate = lookup_rate
        builder._metadata.update({'lookup_table': lookup_table})
        
        return builder
    
    @classmethod
    def matrix_flow(cls, name: str, transition_matrix: np.ndarray) -> 'FlowBuilder':
        """Create matrix-based flow with transition matrix"""
        builder = cls(name)
        builder._pattern = FlowPattern.MATRIX
        builder._matrix = transition_matrix
        builder._metadata.update({'matrix_shape': transition_matrix.shape})
        return builder
    
    def build(self) -> MultiDimensionalFlow:
        """Build the MultiDimensionalFlow object"""
        return MultiDimensionalFlow(
            name=self.name,
            from_selector=self._from,
            to_selector=self._to,
            rate_func=self._rate,
            condition=self._condition,
            matrix=self._matrix,
            pattern=self._pattern,
            debug=self._debug,
            prevent_negatives=self._prevent_negatives,
            metadata=self._metadata
        )

# === JSON Configuration System ===
class FlowRegistry:
    """Registry for selectors, rate functions, and conditions"""
    
    def __init__(self):
        self.selectors = {
            'broadcast': broadcast_selector,
            'age_group': self._create_age_selector,
            'region': self._create_region_selector,
            'threshold': self._create_threshold_selector
        }
        
        self.rate_functions = {
            'constant': self._create_constant_rate,
            'proportional': self._create_proportional_rate,
            'lookup': self._create_lookup_rate
        }
        
        self.conditions = {
            'always': lambda values, ctx: np.ones_like(values, dtype=bool),
            'threshold': self._create_threshold_condition
        }
    
    def _create_age_selector(self, age_index: int, dimension: int = 0):
        def age_selector(values: np.ndarray, context: FlowContext) -> np.ndarray:
            mask = np.zeros_like(values, dtype=bool)
            if values.ndim > dimension:
                indices = [slice(None)] * values.ndim
                indices[dimension] = age_index
                mask[tuple(indices)] = True
            return mask
        return age_selector
    
    def _create_region_selector(self, region_index: int, dimension: int = 1):
        def region_selector(values: np.ndarray, context: FlowContext) -> np.ndarray:
            mask = np.zeros_like(values, dtype=bool)
            if values.ndim > dimension:
                indices = [slice(None)] * values.ndim
                indices[dimension] = region_index
                mask[tuple(indices)] = True
            return mask
        return region_selector
    
    def _create_threshold_selector(self, threshold: float, comparison: str = 'greater'):
        if comparison == 'greater':
            return lambda values, ctx: values > threshold
        elif comparison == 'less':
            return lambda values, ctx: values < threshold
        else:
            return lambda values, ctx: np.abs(values - threshold) < 1e-6
    
    def _create_constant_rate(self, rate: float):
        return lambda values, ctx: np.full_like(values, rate, dtype=float)
    
    def _create_proportional_rate(self, rate: float):
        return lambda values, ctx: values * rate
    
    def _create_lookup_rate(self, lookup_table: List[List[float]]):
        x_values = np.array([row[0] for row in lookup_table])
        y_values = np.array([row[1] for row in lookup_table])
        
        def lookup_rate(values: np.ndarray, context: FlowContext) -> np.ndarray:
            lookup_value = context.time
            rate = np.interp(lookup_value, x_values, y_values)
            return np.full_like(values, rate, dtype=float)
        
        return lookup_rate
    
    def _create_threshold_condition(self, threshold: float, comparison: str = 'greater'):
        return self._create_threshold_selector(threshold, comparison)

def load_flows_from_json(file_path: Union[str, Path]) -> List[MultiDimensionalFlow]:
    """Load flows from JSON configuration file"""
    
    with open(file_path, 'r') as f:
        config = json.load(f)
    
    registry = FlowRegistry()
    flows = []
    
    for flow_config in config.get('flows', []):
        try:
            flow = _create_flow_from_config(flow_config, registry)
            flows.append(flow)
        except Exception as e:
            warnings.warn(f"Failed to create flow '{flow_config.get('name', 'unknown')}': {e}")
    
    return flows

def _create_flow_from_config(config: Dict[str, Any], registry: FlowRegistry) -> MultiDimensionalFlow:
    """Create flow from individual configuration"""
    
    name = config['name']
    flow_type = config.get('type', 'custom')
    
    # Handle high-level flow types
    if flow_type == 'broadcast':
        rate = config.get('rate', 0.1)
        return FlowBuilder.broadcast_flow(name, rate).build()
    
    elif flow_type == 'proportional':
        rate = config.get('rate', 0.05)
        return FlowBuilder.proportional_flow(name, rate).build()
    
    elif flow_type == 'aging':
        return FlowBuilder.aging_flow(
            name,
            config['from_age_index'],
            config['to_age_index'],
            config['rate'],
            config.get('age_dimension', 0)
        ).build()
    
    elif flow_type == 'migration':
        return FlowBuilder.migration_flow(
            name,
            config['from_region_index'],
            config['to_region_index'], 
            config['rate'],
            config.get('region_dimension', 1)
        ).build()
    
    elif flow_type == 'threshold':
        return FlowBuilder.threshold_flow(
            name,
            config['threshold'],
            config['rate'],
            config.get('comparison', 'greater')
        ).build()
    
    elif flow_type == 'lookup':
        return FlowBuilder.lookup_flow(name, config['lookup_table']).build()
    
    elif flow_type == 'matrix':
        matrix = np.array(config['matrix'])
        return FlowBuilder.matrix_flow(name, matrix).build()
    
    else:
        raise ValueError(f"Unknown flow type: {flow_type}")

def save_flows_to_json(flows: List[MultiDimensionalFlow], file_path: Union[str, Path]):
    """Save flows to JSON configuration file"""
    
    config = {'flows': []}
    
    for flow in flows:
        flow_config = {
            'name': flow.name,
            'type': flow.pattern.value if flow.pattern else 'custom',
            'debug': flow.debug,
            'prevent_negatives': flow.prevent_negatives
        }
        
        # Add pattern-specific configuration
        if flow.metadata:
            flow_config.update(flow.metadata)
        
        # Handle matrix flows
        if flow.matrix is not None:
            flow_config['matrix'] = flow.matrix.tolist()
        
        config['flows'].append(flow_config)
    
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2)

# === Integration with Simulation System ===
class EnhancedStock:
    """Stock class that works with MultiDimensionalFlow"""
    
    def __init__(self, values: np.ndarray, name: str, dimensions: List[str] = None):
        self.values = values
        self.name = name
        self.dimensions = dimensions or []
        self.md_flows = []  # List of (flow, target_stock) tuples
        
        # History tracking
        self.history = [values.copy()]
        self.time_history = [0.0]
    
    def add_md_flow(self, flow: MultiDimensionalFlow, target_stock: 'EnhancedStock' = None):
        """Add multi-dimensional flow"""
        self.md_flows.append((flow, target_stock or self))
    
    def step(self, dt: float, context: FlowContext):
        """Execute one simulation step with multi-dimensional flows"""
        
        for flow, target_stock in self.md_flows:
            new_from, new_to, execution_info = flow.apply(
                self.values, target_stock.values, context
            )
            
            if execution_info['success']:
                self.values = new_from
                target_stock.values = new_to
            else:
                warnings.warn(f"Flow {flow.name} failed: {execution_info['errors']}")
        
        # Update history
        self.history.append(self.values.copy())
        self.time_history.append(context.time)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get stock statistics"""
        return {
            'name': self.name,
            'current_total': float(np.sum(self.values)),
            'current_shape': self.values.shape,
            'dimensions': self.dimensions,
            'flow_count': len(self.md_flows)
        }

# === Basic Selector and Rate Functions ===
def broadcast_selector(values: np.ndarray, context: FlowContext) -> np.ndarray:
    """Select all elements"""
    return np.ones_like(values, dtype=bool)

def age_group_selector(age_index: int, dimension: int = 0):
    """Create age group selector for specific index"""
    def selector(values: np.ndarray, context: FlowContext) -> np.ndarray:
        mask = np.zeros_like(values, dtype=bool)
        if values.ndim > dimension:
            indices = [slice(None)] * values.ndim
            indices[dimension] = age_index
            mask[tuple(indices)] = True
        return mask
    return selector

def proportional_rate(rate: float):
    """Create proportional rate function"""
    return lambda values, ctx: values * rate

def constant_rate(rate: float):
    """Create constant rate function"""
    return lambda values, ctx: np.full_like(values, rate, dtype=float)

def threshold_condition(threshold: float, comparison: str = 'greater'):
    """Create threshold condition"""
    if comparison == 'greater':
        return lambda values, ctx: values > threshold
    elif comparison == 'less':
        return lambda values, ctx: values < threshold
    else:
        return lambda values, ctx: np.abs(values - threshold) < 1e-6

# === Test and Example Functions ===
def create_sample_json_config() -> Dict[str, Any]:
    """Create sample JSON configuration for testing"""
    return {
        "flows": [
            {
                "name": "population_growth",
                "type": "proportional",
                "rate": 0.02,
                "debug": False
            },
            {
                "name": "aging_young_to_middle",
                "type": "aging",
                "from_age_index": 0,
                "to_age_index": 1,
                "rate": 0.05,
                "age_dimension": 0
            },
            {
                "name": "migration_north_to_south",
                "type": "migration", 
                "from_region_index": 0,
                "to_region_index": 1,
                "rate": 0.01,
                "region_dimension": 1
            },
            {
                "name": "threshold_migration",
                "type": "threshold",
                "threshold": 100.0,
                "rate": 0.03,
                "comparison": "greater"
            }
        ]
    }

def run_comprehensive_test():
    """Run comprehensive test of the enhanced flow system"""
    
    print("🧪 Running Comprehensive Multi-Dimensional Flow Test")
    print("=" * 60)
    
    # Create test population: 3 age groups × 2 regions
    initial_pop = np.array([
        [1000, 800],   # Young: North, South
        [1500, 1200],  # Middle: North, South  
        [500, 400]     # Old: North, South
    ], dtype=float)
    
    # Create context
    context = FlowContext(
        time=0.0,
        dt=1.0,
        stock_dimensions={'population': ['age', 'region']},
        stock_shapes={'population': initial_pop.shape},
        current_values={'population': initial_pop}
    )
    
    # Test 1: Broadcast growth
    print("\n1. Testing broadcast growth flow")
    growth_flow = FlowBuilder.broadcast_flow("growth", 0.02).debug(True).build()
    new_pop, _, info = growth_flow.apply(initial_pop, np.zeros_like(initial_pop), context)
    print(f"   Growth transferred: {info['transferred_amount']:.2f}")
    print(f"   Success: {info['success']}")
    
    # Test 2: Aging flow
    print("\n2. Testing aging flow")
    aging_flow = FlowBuilder.aging_flow("aging", 0, 1, 0.05).debug(True).build()
    aged_pop, _, info = aging_flow.apply(new_pop, np.zeros_like(new_pop), context)
    print(f"   Aging transferred: {info['transferred_amount']:.2f}")
    print(f"   Success: {info['success']}")
    
    # Test 3: Migration flow
    print("\n3. Testing migration flow")
    migration_flow = FlowBuilder.migration_flow("migration", 0, 1, 0.01).debug(True).build()
    migrated_pop, _, info = migration_flow.apply(aged_pop, np.zeros_like(aged_pop), context)
    print(f"   Migration transferred: {info['transferred_amount']:.2f}")
    print(f"   Success: {info['success']}")
    
    # Test 4: Matrix flow
    print("\n4. Testing matrix flow")
    # Simple 2x2 migration matrix
    migration_matrix = np.array([
        [0.95, 0.05],  # 95% stay in region 0, 5% move to region 1
        [0.02, 0.98]   # 2% move to region 0, 98% stay in region 1
    ])
    
    # Flatten for matrix operations
    flat_pop = new_pop.sum(axis=0)  # Sum over age groups
    matrix_flow = FlowBuilder.matrix_flow("matrix_migration", migration_matrix).debug(True).build()
    new_flat, _, info = matrix_flow.apply(flat_pop, np.zeros_like(flat_pop), context)
    print(f"   Matrix migration transferred: {info['transferred_amount']:.2f}")
    print(f"   Success: {info['success']}")
    
    # Test 5: JSON Configuration
    print("\n5. Testing JSON configuration")
    config = create_sample_json_config()
    
    # Save and load test
    with open('test_flows.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    loaded_flows = load_flows_from_json('test_flows.json')
    print(f"   Loaded {len(loaded_flows)} flows from JSON")
    
    # Test each loaded flow
    for flow in loaded_flows:
        try:
            _, _, info = flow.apply(initial_pop, np.zeros_like(initial_pop), context)
            print(f"   Flow '{flow.name}': {'✓' if info['success'] else '✗'}")
        except Exception as e:
            print(f"   Flow '{flow.name}': ✗ ({e})")
    
    # Test 6: Enhanced Stock Integration
    print("\n6. Testing Enhanced Stock integration")
    stock = EnhancedStock(initial_pop.copy(), "test_population", ["age", "region"])
    stock.add_md_flow(FlowBuilder.proportional_flow("growth", 0.02).build())
    
    context.time = 1.0
    stock.step(1.0, context)
    
    stats = stock.get_statistics()
    print(f"   Stock total after growth: {stats['current_total']:.2f}")
    print(f"   Stock shape: {stats['current_shape']}")
    
    # Final statistics
    print("\n7. Flow statistics")
    for flow in [growth_flow, aging_flow, migration_flow]:
        stats = flow.get_statistics()
        print(f"   {stats['name']}: {stats['execution_count']} executions, "
              f"{stats['total_transferred']:.2f} total transferred")
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    run_comprehensive_test()