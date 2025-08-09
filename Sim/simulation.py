"""
Enhanced System Dynamics Simulation Module
Automatically integrates advanced integration methods when available
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from enum import Enum
import json
import warnings
from typing import Union, List, Callable, Any, Optional, Tuple, Dict

# Multi-dimensional flow support
try:
    from enhanced_flow import MultiDimensionalFlow, FlowContext
    ENHANCED_FLOWS_AVAILABLE = True
except ImportError:
    ENHANCED_FLOWS_AVAILABLE = False
    
    # Mock classes for when enhanced flows not available
    class FlowContext:
        pass
    class MultiDimensionalFlow:
        pass



# Import your existing classes (I'll include the essential ones here)
class TimeUnit(Enum):
    """Time units for simulation"""
    SECOND = ("second", "seconds", 1)
    MINUTE = ("minute", "minutes", 60)
    HOUR = ("hour", "hours", 3600)
    DAY = ("day", "days", 86400)
    WEEK = ("week", "weeks", 604800)
    MONTH = ("month", "months", 2629746)
    YEAR = ("year", "years", 31556952)
    
    def __init__(self, singular, plural, seconds_per_unit):
        self.singular = singular
        self.plural = plural
        self.seconds_per_unit = seconds_per_unit
    
    @classmethod
    def from_string(cls, time_str: str):
        """Create TimeUnit from string"""
        time_str = time_str.lower().strip()
        for unit in cls:
            if time_str in [unit.singular, unit.plural, unit.singular + 's']:
                return unit
        raise ValueError(f"Unknown time unit: {time_str}")

class Stock:
    """Enhanced Stock class with better tracking and validation"""
    
        
    def __init__(self, dim: Union[int, tuple] = None, 
                 values: Union[float, List, np.ndarray] = 0.0, 
                 name: str = "", 
                 min_value: float = 0.0, 
                 max_value: float = np.inf, 
                 units: str = "", 
                 dimensions: Optional[List[str]] = None,
                 dimension_labels: Optional[Dict[str, List[str]]] = None):
            
        
        self.name = name or f"Stock_{id(self)}"
        self.units = units
        
        # Multi-dimensional support
        self.dimensions = dimensions or []  # Dimension names like ['age', 'region'] 
        self.dimension_labels = dimension_labels or {}  # {'age': ['young', 'old'], 'region': ['north', 'south']}
        self.md_flows = []  # List of (flow, target_stock) tuples for multi-dimensional flows
        
        # Enhanced statistics for multi-dimensional data
        self.dimension_statistics = {}        
        
        self.min_value = min_value
        self.max_value = max_value
        
        # Handle dimensions and initial values
        if dim is None:
            self.values = np.array(values, dtype=np.float64)
            self.dim = ()
        elif isinstance(dim, int):
            if isinstance(values, (int, float)):
                self.values = np.full(dim, values, dtype=float)
            else:
                self.values = np.array(values, dtype=np.float64)
                if self.values.shape[0] != dim:
                    raise ValueError(f"Values length {len(values)} doesn't match dimension {dim}")
            self.dim = (dim,)
        elif isinstance(dim, tuple):
            if isinstance(values, (int, float)):
                self.values = np.full(dim, values, dtype=float)
            else:
                self.values = np.array(values, dtype=np.float64)
                if self.values.shape != dim:
                    raise ValueError(f"Values shape {self.values.shape} doesn't match dimension {dim}")
            self.dim = dim
        
        # Validate initial values
        self.values = np.clip(self.values, self.min_value, self.max_value)
        self.initial_values = np.copy(self.values)
        
        # Flow connections
        self._inflows = {}
        self._outflows = {}
        
        # History tracking
        self.history = [np.copy(self.values)]
        self.time_history = [0.0]
        self.min_recorded = np.copy(self.values)
        self.max_recorded = np.copy(self.values)
        print(f"Created Stock '{self.name}': dim={self.dim}, shape={self.values.shape}, units={self.units}")        
    
    @property
    def inflows(self):
        """Get dictionary of inflows"""
        return self._inflows
    
    @property
    def outflows(self):
        '''Get dictionary of outflows'''
        return self._outflows
    
    def add_inflow(self, flow, connection_name: str = None):
        """Add an inflow to this stock"""
        flow_name = connection_name or f"inflow_{len(self._inflows)}"
        self._inflows[flow_name] = flow
        flow._to_stock = self
        flow._to_connection_name = flow_name
    
    def add_outflow(self, flow, connection_name: str = None):
        """Add an outflow from this stock"""
        flow_name = connection_name or f"outflow_{len(self._outflows)}"
        self._outflows[flow_name] = flow
        flow._from_stock = self
        flow._from_connection_name = flow_name
        
    
    def remove_flow(self, flow_name: str):
        """Remove a flow connection"""
        if flow_name in self._inflows:
            del self._inflows[flow_name]
        if flow_name in self._outflows:
            del self._outflows[flow_name]
        
    


    
    def get_net_flow(self, dt: float):
        """Calculate net flow into this stock"""        
        net_flow = np.zeros_like(self.values, dtype=np.float64)
        
        # Add inflows
        for flow in self._inflows.values():
            try:
                flow_rate = flow.get_rate()
                # Handle None flow rates
                if flow_rate is None:
                    flow_rate = 0.0
                
                # Handle scalar flows for multi-dimensional stocks
                if isinstance(flow_rate, (int, float)) and self.values.ndim > 0:
                    flow_rate = np.full_like(self.values, flow_rate, dtype=np.float64)
                elif isinstance(flow_rate, np.ndarray):
                    flow_rate = flow_rate.astype(np.float64)
                    if flow_rate.shape != self.values.shape:
                        try:
                            flow_rate = np.broadcast_to(flow_rate, self.values.shape)
                        except ValueError:
                            warnings.warn(f"Inflow rate shape {flow_rate.shape} incompatible with stock shape {self.values.shape}")
                            flow_rate = np.zeros_like(self.values, dtype=np.float64)
                else:
                    # Convert scalar to array
                    flow_rate = np.full_like(self.values, float(flow_rate), dtype=np.float64)
                    
                net_flow += flow_rate
            except Exception as e:
                warnings.warn(f"Error in inflow calculation for {self.name}: {e}")
        
        # Subtract outflows  
        for flow in self._outflows.values():
            try:
                flow_rate = flow.get_rate()
                # Handle None flow rates
                if flow_rate is None:
                    flow_rate = 0.0
                    
                # Handle scalar flows for multi-dimensional stocks
                if isinstance(flow_rate, (int, float)) and self.values.ndim > 0:
                    flow_rate = np.full_like(self.values, flow_rate, dtype=np.float64)
                elif isinstance(flow_rate, np.ndarray):
                    flow_rate = flow_rate.astype(np.float64)
                    if flow_rate.shape != self.values.shape:
                        try:
                            flow_rate = np.broadcast_to(flow_rate, self.values.shape)
                        except ValueError:
                            warnings.warn(f"Outflow rate shape {flow_rate.shape} incompatible with stock shape {self.values.shape}")
                            flow_rate = np.zeros_like(self.values, dtype=np.float64)
                else:
                    # Convert scalar to array
                    flow_rate = np.full_like(self.values, float(flow_rate), dtype=np.float64)
                    
                net_flow -= flow_rate
            except Exception as e:
                warnings.warn(f"Error in outflow calculation for {self.name}: {e}")
    
        # Handle multi-dimensional flows (if available)
        if hasattr(self, 'md_flows') and self.md_flows:
            try:
                from enhanced_flow import FlowContext
                for flow_name, md_flow, target_stock in self.md_flows:
                    try:
                        context = FlowContext(
                            time=getattr(self, '_current_time', 0.0),
                            dt=dt,
                            stock_dimensions={self.name: getattr(self, 'dimensions', [])},
                            stock_shapes={self.name: self.values.shape},
                            current_values={self.name: self.values}
                        )
                        
                        new_from, new_to, info = md_flow.apply(self.values, target_stock.values, context)
                        if info['success']:
                            if target_stock == self:
                                md_net_flow = new_to - self.values
                            else:
                                md_net_flow = new_from - self.values
                            net_flow += md_net_flow
                            
                    except Exception as e:
                        warnings.warn(f"Error calculating MD flow {md_flow.name}: {e}")
            except ImportError:
                pass  # Enhanced flows not available
    
        return net_flow  # CRITICAL: Make sure this return statement exists!                
                
                
        
    def add_md_flow(self, flow: MultiDimensionalFlow, target_stock: 'Stock' = None, connection_name: str = None):
        """Add multi-dimensional flow to this stock"""
        flow_name = connection_name or f"md_flow_{len(self.md_flows)}"
        self.md_flows.append((flow_name, flow, target_stock or self))
        print(f"Added MD flow '{flow.name}' to stock '{self.name}'")
    
    def remove_md_flow(self, flow_name: str):
        """Remove multi-dimensional flow by name"""
        self.md_flows = [(name, flow, target) for name, flow, target in self.md_flows 
                        if name != flow_name]
    
    def get_md_flows(self) -> List[Tuple[str, MultiDimensionalFlow, 'Stock']]:
        """Get all multi-dimensional flows"""
        return self.md_flows.copy()    
        
    
    def update(self, dt: float, sim_time: float):
        """Update stock value based on flows"""
        net_flow = self.get_net_flow(dt)
        potential_new_value = self.values + net_flow * dt
        # Apply constraints
        self.values = np.clip(potential_new_value, self.min_value, self.max_value)

        # Store current time for MD flows
        self._current_time = sim_time
        
        # Update dimension-specific statistics
        if self.dimensions and self.values.ndim > 0:
            self._update_dimension_statistics()        
        
        
        
        # Update history and statistics
        self.history.append(np.copy(self.values))
        self.time_history.append(sim_time)
        self.min_recorded = np.minimum(self.min_recorded, self.values)
        self.max_recorded = np.maximum(self.max_recorded, self.values)
    
    def reset(self):
        """Reset stock to initial values"""
        self.values = np.copy(self.initial_values)
        self.history = [np.copy(self.values)]
        self.time_history = [0.0]
        self.min_recorded = np.copy(self.values)
        self.max_recorded = np.copy(self.values)
    
    def _update_dimension_statistics(self):
        """Update statistics for each dimension"""
        try:
            if len(self.dimensions) == 1:
                # 1D case: statistics per dimension element
                self.dimension_statistics = {
                    f"{self.dimensions[0]}_{i}": float(self.values[i]) 
                    for i in range(len(self.values))
                }
            elif len(self.dimensions) == 2:
                # 2D case: statistics for each combination
                for i in range(self.values.shape[0]):
                    for j in range(self.values.shape[1]):
                        key = f"{self.dimensions[0]}_{i}_{self.dimensions[1]}_{j}"
                        self.dimension_statistics[key] = float(self.values[i, j])
            # Add more dimensions as needed
        except Exception:
            pass  # Silently skip if dimension analysis fails    
        
    
    
    def total(self):
        """Get total value"""        
        return np.sum(self.values)
    
    
    def average(self):
        """Get average value"""
        return np.mean(self.values)


    def get_statistics(self):
        """Get comprehensive statistics"""
        stats = {
            'current': self.total(),
            'initial': np.sum(self.initial_values),
            'min': np.sum(self.min_recorded),
            'max': np.sum(self.max_recorded),
            'final_time': self.time_history[-1] if self.time_history else 0,
            'shape': self.values.shape,
            'units': self.units,
            
            # NEW: Multi-dimensional fields
            'dimensions': self.dimensions,
            'dimension_labels': self.dimension_labels,
            'dimension_statistics': self.dimension_statistics,
            'md_flow_count': len(self.md_flows),
            'is_multidimensional': len(self.dimensions) > 0
        }
        return stats


    def __mul__(self, other):
        """Enable stock * rate syntax for creating flows"""
        if isinstance(other, (int, float, np.ndarray)):
            return Flow(rate_expression=lambda: self.values * other, name=f"{self.name}*{other}")
        return NotImplemented
    
    def __rmul__(self, other):
        """Enable rate * stock syntax"""
        return self.__mul__(other)
    
    def __str__(self):
        if self.values.ndim == 0:
            units_str = f" {self.units}" if self.units else ""
            return f"{self.name}: {self.values.item():.2f}{units_str}"
        else:
            return f"{self.name}: {self.values} {self.units}"


    def get_dimension_slice(self, dimension_name: str, dimension_value: Union[str, int]) -> np.ndarray:
        """Get slice of stock for specific dimension value"""
        if dimension_name not in self.dimensions:
            raise ValueError(f"Dimension '{dimension_name}' not found in stock '{self.name}'")
        
        dim_index = self.dimensions.index(dimension_name)
        
        if isinstance(dimension_value, str) and dimension_name in self.dimension_labels:
            value_index = self.dimension_labels[dimension_name].index(dimension_value)
        else:
            value_index = dimension_value
        
        # Create slice
        indices = [slice(None)] * self.values.ndim
        indices[dim_index] = value_index
        
        return self.values[tuple(indices)]
    
    def set_dimension_slice(self, dimension_name: str, dimension_value: Union[str, int], 
                           values: np.ndarray):
        """Set values for specific dimension slice"""
        if dimension_name not in self.dimensions:
            raise ValueError(f"Dimension '{dimension_name}' not found in stock '{self.name}'")
        
        dim_index = self.dimensions.index(dimension_name)
        
        if isinstance(dimension_value, str) and dimension_name in self.dimension_labels:
            value_index = self.dimension_labels[dimension_name].index(dimension_value)
        else:
            value_index = dimension_value
        
        # Set slice
        indices = [slice(None)] * self.values.ndim
        indices[dim_index] = value_index
        
        self.values[tuple(indices)] = values
    
    def create_flow_context(self, dt: float, sim_time: float, 
                           all_stocks: Dict[str, 'Stock'] = None) -> FlowContext:
        """Create FlowContext for multi-dimensional flows"""
        stock_dimensions = {self.name: self.dimensions}
        stock_shapes = {self.name: self.values.shape}
        current_values = {self.name: self.values}
        
        # Add other stocks if provided
        if all_stocks:
            for stock_name, stock in all_stocks.items():
                stock_dimensions[stock_name] = stock.dimensions
                stock_shapes[stock_name] = stock.values.shape
                current_values[stock_name] = stock.values
        
        return FlowContext(
            time=sim_time,
            dt=dt,
            stock_dimensions=stock_dimensions,
            stock_shapes=stock_shapes,
            current_values=current_values
        )


    def get_dimension_slice(self, dimension_name: str, dimension_value: Union[str, int]) -> np.ndarray:
        """Get slice of stock for specific dimension value"""
        if dimension_name not in self.dimensions:
            raise ValueError(f"Dimension '{dimension_name}' not found in stock '{self.name}'")
        
        dim_index = self.dimensions.index(dimension_name)
        
        if isinstance(dimension_value, str) and dimension_name in self.dimension_labels:
            value_index = self.dimension_labels[dimension_name].index(dimension_value)
        else:
            value_index = dimension_value
        
        # Create slice
        indices = [slice(None)] * self.values.ndim
        indices[dim_index] = value_index
        
        return self.values[tuple(indices)]
    
    def set_dimension_slice(self, dimension_name: str, dimension_value: Union[str, int], 
                           values: np.ndarray):
        """Set values for specific dimension slice"""
        if dimension_name not in self.dimensions:
            raise ValueError(f"Dimension '{dimension_name}' not found in stock '{self.name}'")
        
        dim_index = self.dimensions.index(dimension_name)
        
        if isinstance(dimension_value, str) and dimension_name in self.dimension_labels:
            value_index = self.dimension_labels[dimension_name].index(dimension_value)
        else:
            value_index = dimension_value
        
        # Set slice
        indices = [slice(None)] * self.values.ndim
        indices[dim_index] = value_index
        
        self.values[tuple(indices)] = values
    
    def create_flow_context(self, dt: float, sim_time: float, 
                           all_stocks: Dict[str, 'Stock'] = None) -> FlowContext:
        """Create FlowContext for multi-dimensional flows"""
        stock_dimensions = {self.name: self.dimensions}
        stock_shapes = {self.name: self.values.shape}
        current_values = {self.name: self.values}
        
        # Add other stocks if provided
        if all_stocks:
            for stock_name, stock in all_stocks.items():
                stock_dimensions[stock_name] = stock.dimensions
                stock_shapes[stock_name] = stock.values.shape
                current_values[stock_name] = stock.values
        
        return FlowContext(
            time=sim_time,
            dt=dt,
            stock_dimensions=stock_dimensions,
            stock_shapes=stock_shapes,
            current_values=current_values
        )





class Flow:
    """Enhanced Flow class with better error handling and tracking"""
    
    def __init__(self, rate_expression: Union[float, Callable, np.ndarray] = 0.0, 
                 name: str = "", units: str = "", min_rate: float = -np.inf, 
                 max_rate: float = np.inf):
        self.name = name or f"Flow_{id(self)}"
        self.units = units
        self.min_rate = min_rate
        self.max_rate = max_rate
        
        if callable(rate_expression):
            self.rate_expression = rate_expression
        else:
            self.constant_rate = np.array(rate_expression, dtype=float)
            self.rate_expression = lambda: self.constant_rate
        
        # Stock connections
        self._from_stock = None
        self._to_stock = None
        self._from_connection_name = None
        self._to_connection_name = None
        
        
        # history
        self.history = []
        self.time_history = []
        self.cumulative_flow = 0.0
        print(f"Created Flow '{self.name}': units={self.units}")
    
    def get_rate(self):
        """Get current flow rate with error handling"""
        try:
            rate = self.rate_expression()
            rate = np.array(rate, dtype=float)
            # Apply constraints
            rate = np.clip(rate, self.min_rate, self.max_rate)
            return rate
        except Exception as e:
            warnings.warn(f"Error calculating flow rate for {self.name}: {e}")
            return np.array(0.0)
    
    def update_history(self, rate: np.ndarray, sim_time: float, dt: float):
        """Update flow history with cumulative tracking"""
        self.history.append(np.copy(rate))
        self.time_history.append(sim_time)
        self.cumulative_flow += np.sum(rate) * dt
        
    def get_statistics(self):
            """Get flow statistics"""
            if not self.history:
                return {'total_flow': 0, 'avg_rate': 0, 'max_rate': 0, 'units': self.units}
            
            rates = np.array(self.history)
            return {
                'total_flow': self.cumulative_flow,
                'avg_rate': np.mean(rates),
                'max_rate': np.max(rates),
                'min_rate': np.min(rates),
                'units': self.units
            }
        
        
    
    def connect(self, from_stock: Stock = None, to_stock: Stock = None):
        """Connect flow between stocks"""
        if from_stock:
            from_stock.add_outflow(self)
        if to_stock:
            to_stock.add_inflow(self)
        return self

class Model:
    """Enhanced Model class with automatic integration engine support"""
    
    def __init__(self, time_unit: Union[str, TimeUnit] = TimeUnit.DAY, 
                 start_time: Union[datetime, str, None] = None,
                 end_time: Union[datetime, str, int, float, None] = None,
                 dt: Optional[float] = None,
                 name: str = "Simulation Model",
                 integration_method: str = "auto"):
        
        # Time management
        self.time_unit = TimeUnit.from_string(time_unit) if isinstance(time_unit, str) else time_unit
        self.name = name
        
        
        if dt is None:
            if self.time_unit == TimeUnit.SECOND:
                self.dt = 0.001  # 1 millisecond
            elif self.time_unit == TimeUnit.MINUTE:
                self.dt = 0.06   # 3.6 seconds
            elif self.time_unit == TimeUnit.HOUR:
                self.dt = 3.6    # 3.6 minutes
            elif self.time_unit == TimeUnit.DAY:
                self.dt = 0.024  # ~35 minutes
            else:
                self.dt = 1.0 / 5000.0  # Generic 1/1000
        else:
            self.dt = dt        
        
        
        # Time management (simplified for brevity)
        if start_time is None:
            self.start_datetime = datetime.now()
        elif isinstance(start_time, str):
            self.start_datetime = datetime.fromisoformat(start_time)
        else:
            self.start_datetime = start_time
        
        # End time handling
        self.end_datetime = None
        self.duration = None
        
        if end_time is not None:
            if isinstance(end_time, (int, float)):
                self.duration = float(end_time)
                self.end_datetime = self._add_time_units(self.start_datetime, end_time)
            elif isinstance(end_time, str):
                self.end_datetime = datetime.fromisoformat(end_time)
                self.duration = self._calculate_duration()
            elif isinstance(end_time, datetime):
                self.end_datetime = end_time
                self.duration = self._calculate_duration()
                
        
        # Simulation state
        self.time = 0.0
        self.current_datetime = self.start_datetime
        self.stocks = []
        self.flows = []
        self.time_history = [0.0]
        self.datetime_history = [self.start_datetime]
        
        # Simulation control
        self.is_running = False
        self.step_count = 0
        self.max_steps = None
        
        # Results and statistics
        self.results = {}
        
        # Integration settings
        self.integration_method = integration_method
        self.use_advanced_integration = self._setup_advanced_integration()
        
        print(f"Created Model '{self.name}' with {self.time_unit.singular} time unit")
        if self.use_advanced_integration:
            print(f"✓ Advanced integration engine loaded (default method: {integration_method})")
        else:
            print("! Using basic Euler integration (install scipy for advanced methods)")
    
    def _setup_advanced_integration(self):
        """Automatically setup advanced integration if available"""
        try:
            # Try to import and setup the integration engine
            from integration_engine import IntegrationEngine, ModelIntegrationExtensions
            
            # Add integration support
            ModelIntegrationExtensions.add_integration_support(self)
            return True
            
        except ImportError:
            # Integration engine not available, use basic methods
            warnings.warn(
                "Advanced integration engine not available. "
                "Using basic Euler integration. "
                "For better performance and accuracy, ensure integration_engine.py is available."
            )
            return False
    
    def _add_time_units(self, base_datetime: datetime, units: float) -> datetime:
        if self.time_unit == TimeUnit.SECOND:
            return base_datetime + timedelta(seconds=units)
        elif self.time_unit == TimeUnit.MINUTE:
            return base_datetime + timedelta(minutes=units)
        elif self.time_unit == TimeUnit.HOUR:
            return base_datetime + timedelta(hours=units)
        elif self.time_unit == TimeUnit.DAY:
            return base_datetime + timedelta(days=units)
        elif self.time_unit == TimeUnit.WEEK:
            return base_datetime + timedelta(weeks=units)
        else:
            # For months/years, use average durations
            seconds = units * self.time_unit.seconds_per_unit
            return base_datetime + timedelta(seconds=seconds)
    
    def _calculate_duration(self) -> float:
        """Calculate duration between start and end times"""
        if self.end_datetime is None:
            return None
        
        time_diff = self.end_datetime - self.start_datetime
        total_seconds = time_diff.total_seconds()
        
        return total_seconds / self.time_unit.seconds_per_unit
    
    def add_stock(self, stock: Stock):
        """Add stock to model"""
        if stock not in self.stocks:
            self.stocks.append(stock)
            
            # NEW: Validate multi-dimensional setup
            if stock.dimensions:
                print(f"Added multi-dimensional stock '{stock.name}' with dimensions: {stock.dimensions}")
                if stock.dimension_labels:
                    for dim, labels in stock.dimension_labels.items():
                        print(f"  {dim}: {labels}")
        
        return stock
    
    
    
    
    def add_flow(self, flow: Flow):
        """Add flow to model"""
        if flow not in self.flows:
            self.flows.append(flow)
        return flow
    
    def get_current_datetime(self):
        """Get current simulation datetime"""
        return self._add_time_units(self.start_datetime, self.time)
    
    def validate_mass_conservation(self, tolerance: float = 1e-6) -> dict:
        """Check if total mass is conserved in the system"""
        from unified_validation import get_unified_validator
        validator = get_unified_validator()
        report = validator.validate_mass_conservation(self)
        
        # Backward compatibility wrapper
        return {
            'is_conserved': report.is_valid,
            'relative_error': report.get_metric('relative_error', 0.0),
            'total_inflow': report.get_metric('total_inflow', 0.0),
            'total_outflow': report.get_metric('total_outflow', 0.0),
            'net_flow': report.get_metric('net_flow', 0.0),
            'type': 'unified_validation'
        }

    
    def step(self):
        """Perform one simulation step (basic Euler)"""
        
        # Update current datetime
        self.current_datetime = self._add_time_units(self.start_datetime, self.time)
    
        # Store current time in stocks for MD flow context
        for stock in self.stocks:
            stock._current_time = self.time
    
        # Update all stocks        
        for stock in self.stocks:
            stock.update(self.dt, self.time)
        
        # Record flow history
        for flow in self.flows:
            rate = flow.get_rate()
            flow.update_history(rate, self.time, self.dt)
            
        # Update time        
        self.time += self.dt
        self.step_count += 1
    
        # Update history        
        self.time_history.append(self.time)
        self.datetime_history.append(self.current_datetime)


        # Update time        
        self.time += self.dt
        self.step_count += 1

        # # Update history        
        # self.time_history.append(self.time)
        # self.datetime_history.append(self.current_datetime)
    
    def run(self, duration: Optional[float] = None, 
            until: Optional[datetime] = None, 
            max_steps: Optional[int] = None, 
            progress_callback: Optional[Callable] = None,
            method: Optional[str] = None):
        """
        Enhanced run method that automatically uses advanced integration when available
        
        Parameters:
        -----------
        duration : float, optional
            Duration to run simulation
        until : datetime, optional  
            Run until specific datetime
        max_steps : int, optional
            Maximum number of steps
        progress_callback : callable, optional
            Progress callback function
        method : str, optional
            Integration method ('auto', 'euler', 'rk4', 'rk45', etc.)
            Overrides model default if specified
        """
        
        # Determine integration method to use
        use_method = method or self.integration_method
        
        # Use advanced integration if available and not explicitly using basic euler
        if self.use_advanced_integration and use_method != 'euler_basic':
            return self._run_with_advanced_integration(
                duration, until, max_steps, progress_callback, use_method
            )
        else:
            return self._run_with_basic_integration(
                duration, until, max_steps, progress_callback
            )
    
    def add_md_flow_between_stocks(self, md_flow: 'MultiDimensionalFlow', 
                                  from_stock: Stock, to_stock: Stock = None):
        """Add multi-dimensional flow between stocks"""
        if not ENHANCED_FLOWS_AVAILABLE:
            warnings.warn("Enhanced flows not available. Install enhanced_flow module.")
            return
        
        target_stock = to_stock or from_stock
        from_stock.add_md_flow(md_flow, target_stock)
        
        print(f"Added MD flow '{md_flow.name}' from '{from_stock.name}' to '{target_stock.name}'")
        return md_flow    
        
    
    def _run_with_advanced_integration(self, duration, until, max_steps, progress_callback, method):
        """Run simulation using advanced integration engine"""
        
        print(f"Running simulation with advanced integration (method: {method})")
        
        # Determine end time
        if duration is not None:
            end_time = self.time + duration
        elif until is not None:
            duration_to_until = (until - self.current_datetime).total_seconds() / self.time_unit.seconds_per_unit
            end_time = self.time + duration_to_until
        elif self.duration is not None:
            end_time = self.duration
        else:
            end_time = self.time + 10.0  # Default duration
        
        # Prepare integration parameters
        integration_params = {}
        if max_steps:
            # Convert max_steps to max_step size for adaptive methods
            total_duration = end_time - self.time
            integration_params['max_step'] = total_duration / max_steps
        
        # Run integration
        try:
            result = self.integration_engine.integrate(
                self,
                method=method,
                end_time=end_time,
                **integration_params
            )
            
            if result['success']:
                print(f"✓ Advanced integration completed successfully")
                print(f"  Method: {result.get('integration_method', method)}")
                print(f"  Runtime: {result.get('integration_time', 0):.3f}s")
                print(f"  Steps: {len(result.get('t', []))}")
                
                conservation_check = self.validate_mass_conservation()
                print(conservation_check)
                if (not conservation_check['is_conserved'] and 
                    conservation_check['relative_error'] > 1e-6 and
                    conservation_check.get('type') not in ['pure_source_system', 'pure_sink_system']):
                    warnings.warn(f"Mass conservation violated: {conservation_check['relative_error']:.2e} relative error")              
                            
                # Update model state from integration results
                self._update_from_integration_results(result)
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(self)
                
                # UPDATE RESULTS FIRST
                self._update_results()
                
                self._last_integration_method = result.get('integration_method', method)  # ADD THIS LINE

                # THEN ADD VALIDATION TO FINAL RESULTS (MOVE THIS BLOCK HERE)
                if hasattr(self, 'integration_engine'):
                    validation = self.integration_engine._validate_integration_result(result, self)
                    self.results['validation'] = validation  # ADD TO self.results instead of result
                    print(f"🐛 DEBUG: Validation added to final results: {validation['valid']}")
                    
                    if not validation['valid']:
                        print(f"⚠️  Integration validation failed:")
                        for error in validation['errors']:
                            print(f"     Error: {error}")
                    
                    if validation['warnings']:
                        for warning in validation['warnings']:
                            print(f"     Warning: {warning}")
                
                return self.results
                
            else:
                print(f"✗ Advanced integration failed: {result.get('message', 'Unknown error')}")
                print("  Falling back to basic integration...")
                return self._run_with_basic_integration(duration, until, max_steps, progress_callback)
            
            return self.results
            
        except Exception as e:
            print(f"✗ Advanced integration error: {e}")
            print("  Falling back to basic integration...")
            return self._run_with_basic_integration(duration, until, max_steps, progress_callback)
    
    def _run_with_basic_integration(self, duration, until, max_steps, progress_callback):
        """Run simulation using basic Euler integration"""
        
        print("Running simulation with basic Euler integration")
        
        self.is_running = True
        start_step = self.step_count
        
        # Determine stopping condition
        target_time = None
        if duration is not None:
            target_time = self.time + duration
        elif until is not None:
            duration_to_until = (until - self.current_datetime).total_seconds() / self.time_unit.seconds_per_unit
            target_time = self.time + duration_to_until
        elif self.duration is not None:
            target_time = self.duration
        
        if max_steps is not None:
            self.max_steps = start_step + max_steps
        
        # Run simulation loop
        while self.is_running:
            self.step()
            
            if target_time is not None and self.time >= target_time:
                break
            
            if self.max_steps is not None and self.step_count >= self.max_steps:
                break
            
            if progress_callback and self.step_count % 100 == 0:
                progress_callback(self)
        
        self.is_running = False
        
        steps_taken = self.step_count - start_step
        print(f"✓ Basic integration completed: {steps_taken} steps, final time: {self.time:.4f}")
        
        
        self._update_results()
                    
        return self.results
    
    def _update_from_integration_results(self, result):
        """Update model state from advanced integration results"""
        if 'y' in result and result['success']:
            # Update time arrays
            self.time_history = result['t'].tolist()
            self.time = result['t'][-1]
            
            # Update datetime history
            self.datetime_history = [
                self._add_time_units(self.start_datetime, t) 
                for t in result['t']
            ]
            self.current_datetime = self.datetime_history[-1]
            
            # Update stock histories
            state_idx = 0
            for stock in self.stocks:
                stock_size = stock.values.size if hasattr(stock.values, 'size') else 1
                stock_results = result['y'][state_idx:state_idx + stock_size]
                
                # Update stock history
                if hasattr(stock.values, 'shape') and stock.values.shape:
                    # Multi-dimensional stock
                    stock.history = [
                        stock_results[:, i].reshape(stock.values.shape) 
                        for i in range(len(result['t']))
                    ]
                else:
                    # Scalar stock
                    stock.history = stock_results[0].tolist()
                
                stock.time_history = result['t'].tolist()
                
                # Update current values to final state
                if hasattr(stock.values, 'shape') and stock.values.shape:
                    stock.values = stock_results[:, -1].reshape(stock.values.shape)
                else:
                    stock.values = stock_results[0, -1]
                
                # Update min/max recorded
                if len(stock.history) > 1:
                    all_values = np.array(stock.history)
                    stock.min_recorded = np.min(all_values, axis=0)
                    stock.max_recorded = np.max(all_values, axis=0)
                
                state_idx += stock_size
            
            # Update step count
            self.step_count = len(result['t']) - 1
    
    
    def _update_results(self):
        """Update simulation results"""
        self.results = {
        'simulation_time': self.time,
        'steps': self.step_count,
        'start_datetime': self.start_datetime,
        'end_datetime': self.current_datetime,
        'time_unit': self.time_unit.singular,
        'time_history': self.time_history,  # ADD THIS
        'integration_method': getattr(self, '_last_integration_method', 'unknown'),  # ADD THIS
        'stocks': {stock.name: stock.get_statistics() for stock in self.stocks},
        'flows': {flow.name: flow.get_statistics() for flow in self.flows if hasattr(flow, 'get_statistics')}
    }
    
    
    
    
    
    def reset(self):
        """Reset simulation to initial state"""
        self.time = 0.0
        self.step_count = 0
        self.current_datetime = self.start_datetime
        self.time_history = [0.0]
        self.datetime_history = [self.start_datetime]
        self.is_running = False
        
        for stock in self.stocks:
            stock.reset()
        for flow in self.flows:
            flow.history = []
            flow.time_history = []
            flow.cumulative_flow = 0.0
    
    def analyze_model(self):
        """Analyze model characteristics (if advanced integration available)"""
        if self.use_advanced_integration:
            from integration_engine import ModelAnalyzer
            analyzer = ModelAnalyzer()
            characteristics = analyzer.analyze_model_characteristics(self)
            
            print(f"Model Analysis for '{self.name}':")
            print(f"  States: {characteristics.num_states}")
            print(f"  Large model: {characteristics.is_large}")
            print(f"  Stiff system: {characteristics.is_stiff}")
            print(f"  Has constraints: {characteristics.has_constraints}")
            print(f"  Fast dynamics: {characteristics.has_fast_dynamics}")
            
            # Get recommendation
            method, params = analyzer.recommend_integration_method(self)
            print(f"  Recommended method: {method}")
            print(f"  Suggested parameters: {params}")
            
            return characteristics
        else:
            print("Model analysis requires advanced integration engine")
            return None
    
    def compare_integration_methods(self, methods=None, duration=10.0):
        """Compare different integration methods (if available)"""
        if self.use_advanced_integration:
            if methods is None:
                methods = ['euler', 'rk4', 'rk45']
            
            return self.integration_engine.compare_methods(
                self, methods=methods, duration=duration
            )
        else:
            print("Method comparison requires advanced integration engine")
            return None

    
    def get_all_stocks_context(self, dt: float, sim_time: float) -> Dict[str, Any]:
        """Get context information for all stocks"""
        return {
            stock.name: stock for stock in self.stocks
        }
    
    def validate_multidimensional_model(self) -> Dict[str, Any]:
        """Validate multi-dimensional model setup"""
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'multidimensional_stocks': 0,
            'md_flows': 0
        }
        
        for stock in self.stocks:
            if stock.dimensions:
                validation['multidimensional_stocks'] += 1
                validation['md_flows'] += len(stock.md_flows)
                
                # Validate dimension setup
                if stock.values.ndim != len(stock.dimensions):
                    validation['errors'].append(
                        f"Stock '{stock.name}' has {len(stock.dimensions)} dimensions "
                        f"but array has {stock.values.ndim} dimensions"
                    )
                    validation['valid'] = False
        
        return validation


    
    def plot(self, stocks: List[Stock] = None, flows: List[Flow] = None, 
             use_datetime: bool = False, save_path: str = None):
        """Enhanced plotting (same as your existing implementation)"""
        if stocks is None:
            stocks = self.stocks
        if flows is None:
            flows = self.flows
        
        if use_datetime and len(self.datetime_history) > 0:
            time_axis = self.datetime_history
            time_label = 'Date/Time'
        else:
            time_axis = self.time_history
            time_label = f'Time ({self.time_unit.plural})'
        
        n_plots = (1 if stocks else 0) + (1 if flows else 0)
        if n_plots == 0:
            return
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 6*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        if stocks:
            ax = axes[plot_idx]
            for stock in stocks:
                if stock.values.ndim == 0:
                    values = [h.item() if hasattr(h, 'item') else h for h in stock.history]
                    ax.plot(time_axis, values, label=f'{stock.name}', linewidth=2, marker='o', markersize=2)
                else:
                    totals = [np.sum(h) for h in stock.history]
                    ax.plot(time_axis, totals, label=f"{stock.name} (total)", linewidth=2, marker='o', markersize=2)
            
            ax.set_xlabel(time_label)
            ax.set_ylabel('Stock Value')
            ax.set_title('Stock Values Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        if flows:
            ax = axes[plot_idx]
            for flow in flows:
                if flow.history:
                    flow_time_axis = time_axis[1:len(flow.history)+1] if len(flow.history) < len(time_axis) else time_axis[1:]
                    if isinstance(flow.history[0], np.ndarray) and flow.history[0].ndim > 0:
                        totals = [np.sum(h) for h in flow.history]
                        ax.plot(flow_time_axis, totals, label=f"{flow.name} (total)", linewidth=2, marker='s', markersize=2)
                    else:
                        values = [h.item() if hasattr(h, 'item') else h for h in flow.history]
                        ax.plot(flow_time_axis, values, label=flow.name, linewidth=2, marker='s', markersize=2)
            
            ax.set_xlabel(time_label)
            ax.set_ylabel('Flow Rate')
            ax.set_title('Flow Rates Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    
    
    def save_results(self, filepath: str):
        """Save simulation results to JSON"""
        # Convert datetime objects to ISO format for JSON serialization
        results_copy = self.results.copy()
        if 'start_datetime' in results_copy:
            results_copy['start_datetime'] = results_copy['start_datetime'].isoformat()
        if 'end_datetime' in results_copy:
            results_copy['end_datetime'] = results_copy['end_datetime'].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(results_copy, f, indent=2)
        print(f"Results saved to: {filepath}")


    def validate_multidimensional_setup(self) -> Dict[str, Any]:
        """Validate multi-dimensional model setup"""
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'multidimensional_stocks': 0,
            'md_flows': 0,
            'dimension_consistency': {}
        }
        
        for stock in self.stocks:
            if stock.dimensions:
                validation['multidimensional_stocks'] += 1
                validation['md_flows'] += len(stock.md_flows)
                
                # Validate dimension setup
                if stock.values.ndim != len(stock.dimensions):
                    validation['errors'].append(
                        f"Stock '{stock.name}' has {len(stock.dimensions)} dimensions "
                        f"but array has {stock.values.ndim} dimensions"
                    )
                    validation['valid'] = False
                
                # Check dimension labels consistency
                for i, dim_name in enumerate(stock.dimensions):
                    if dim_name in stock.dimension_labels:
                        expected_size = len(stock.dimension_labels[dim_name])
                        actual_size = stock.values.shape[i]
                        if expected_size != actual_size:
                            validation['warnings'].append(
                                f"Stock '{stock.name}' dimension '{dim_name}' "
                                f"has {expected_size} labels but {actual_size} values"
                            )
        
        return validation


    def print_summary(self):
        """Print comprehensive simulation summary"""
        print(f"\n{'='*60}")
        print(f"SIMULATION SUMMARY: {self.name}")
        print(f"{'='*60}")
        print(f"Integration: {'Advanced' if self.use_advanced_integration else 'Basic Euler'}")
        print(f"Time Unit: {self.time_unit.singular}")
        print(f"Duration: {self.time:.4f} {self.time_unit.plural}")
        print(f"Steps: {self.step_count}")
        
        print(f"\nStocks ({len(self.stocks)}):")
        for stock in self.stocks:
            stats = stock.get_statistics()
            print(f"  {stock.name}: {stats['current']:.2f} {stock.units}")
        
        print(f"\nFlows ({len(self.flows)}):")
        for flow in self.flows:
            if hasattr(flow, 'get_statistics'):
                stats = flow.get_statistics()
                print(f"  {flow.name}: Total={stats.get('total_flow', 0):.2f}")

# Convenience functions that work with both basic and advanced modes
def stock(dim=None, values=0.0, name="", units="", min_value=0.0, max_value=np.inf):
    """Create a stock with specified parameters"""
    return Stock(dim=dim, values=values, name=name, units=units, min_value=min_value, max_value=max_value)

def flow(rate=0.0, name="", units="", min_rate=-np.inf, max_rate=np.inf):
    """Create a flow with specified parameters"""
    return Flow(rate_expression=rate, name=name, units=units, min_rate=min_rate, max_rate=max_rate)

def model(time_unit="day", start_time=None, end_time=None, dt=None, name="Simulation", integration_method="auto"):
    """Create a simulation model with automatic advanced integration if available"""
    return Model(
        time_unit=time_unit, 
        start_time=start_time, 
        end_time=end_time, 
        dt=dt, 
        name=name,
        integration_method=integration_method
    )
