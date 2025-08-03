# # -*- coding: utf-8 -*-
# """
# Created on Sat Aug  2 22:10:07 2025

# @author: NagabhushanamTattaga
# """

# """
# sim_SD.py - System Dynamics Module
# AnyLogic-style Stock and Flow implementation
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Union, List, Callable, Any

# class Stock:
#     """Stock class similar to AnyLogic stocks"""
    
#     def __init__(self, dim: Union[int, tuple] = None, values: Union[float, List, np.ndarray] = 0.0, name: str = ""):
#         self.name = name or f"Stock_{id(self)}"
        
#         # Handle dimensions and initial values
#         if dim is None:
#             # Scalar stock
#             self.values = np.array(values, dtype=float)
#             self.dim = ()
#         elif isinstance(dim, int):
#             # 1D stock
#             if isinstance(values, (int, float)):
#                 self.values = np.full(dim, values, dtype=float)
#             else:
#                 self.values = np.array(values, dtype=float)
#                 if self.values.shape[0] != dim:
#                     raise ValueError(f"Values length {len(values)} doesn't match dimension {dim}")
#             self.dim = (dim,)
#         elif isinstance(dim, tuple):
#             # Multi-dimensional stock
#             if isinstance(values, (int, float)):
#                 self.values = np.full(dim, values, dtype=float)
#             else:
#                 self.values = np.array(values, dtype=float)
#                 if self.values.shape != dim:
#                     raise ValueError(f"Values shape {self.values.shape} doesn't match dimension {dim}")
#             self.dim = dim
        
#         # Initial value for reset
#         self.initial_values = np.copy(self.values)
        
#         # Flow connections
#         self._inflows = []
#         self._outflows = []
        
#         # History tracking
#         self.history = [np.copy(self.values)]
        
#         print(f"Created Stock '{self.name}': dim={self.dim}, shape={self.values.shape}")
    
#     @property
#     def in_flows(self):
#         """Get list of inflows"""
#         return self._inflows
    
#     @property
#     def out_flows(self):
#         """Get list of outflows"""
#         return self._outflows
    
#     def add_inflow(self, flow):
#         """Add an inflow to this stock"""
#         if flow not in self._inflows:
#             self._inflows.append(flow)
#             flow._to_stock = self
    
#     def add_outflow(self, flow):
#         """Add an outflow from this stock"""
#         if flow not in self._outflows:
#             self._outflows.append(flow)
#             flow._from_stock = self
    
#     def get_net_flow(self, dt: float):
#         """Calculate net flow into this stock"""
#         net_flow = np.zeros_like(self.values)
        
#         # Add inflows
#         for flow in self._inflows:
#             flow_rate = flow.get_rate()
#             if isinstance(flow_rate, np.ndarray):
#                 if flow_rate.shape != self.values.shape:
#                     raise ValueError(f"Inflow rate shape {flow_rate.shape} doesn't match stock shape {self.values.shape}")
#             net_flow += flow_rate
        
#         # Subtract outflows
#         for flow in self._outflows:
#             flow_rate = flow.get_rate()
#             if isinstance(flow_rate, np.ndarray):
#                 if flow_rate.shape != self.values.shape:
#                     raise ValueError(f"Outflow rate shape {flow_rate.shape} doesn't match stock shape {self.values.shape}")
#             net_flow -= flow_rate
        
#         return net_flow
    
#     def update(self, dt: float):
#         """Update stock value based on flows"""
#         net_flow = self.get_net_flow(dt)
        
#         # Ensure we don't go negative
#         potential_new_value = self.values + net_flow * dt
        
#         # For outflows, limit to available stock
#         if np.any(potential_new_value < 0):
#             # Reduce outflows proportionally to prevent negative values
#             negative_mask = potential_new_value < 0
#             for flow in self._outflows:
#                 if np.any(negative_mask):
#                     # Reduce this flow rate to prevent negative stock
#                     available = self.values[negative_mask]
#                     required = flow.get_rate() * dt
#                     if isinstance(required, np.ndarray):
#                         reduction_factor = np.minimum(1.0, available / (required[negative_mask] + 1e-10))
#                         # Apply reduction (this is simplified - more complex logic may be needed)
                    
#         self.values = np.maximum(0, potential_new_value)  # Ensure non-negative
#         self.history.append(np.copy(self.values))
    
#     def reset(self):
#         """Reset stock to initial values"""
#         self.values = np.copy(self.initial_values)
#         self.history = [np.copy(self.values)]
    
#     def total(self):
#         """Get total value (sum of all elements)"""
#         return np.sum(self.values)
    
#     def __mul__(self, other):
#         """Enable stock * rate syntax for creating flows"""
#         if isinstance(other, (int, float, np.ndarray)):
#             return Flow(rate_expression=lambda: self.values * other, name=f"{self.name}*{other}")
#         return NotImplemented
    
#     def __rmul__(self, other):
#         """Enable rate * stock syntax"""
#         return self.__mul__(other)
    
#     def __str__(self):
#         if self.values.ndim == 0:
#             return f"{self.name}: {self.values.item():.2f}"
#         else:
#             return f"{self.name}: {self.values}"

# class Flow:
#     """Flow class similar to AnyLogic flows"""
    
#     def __init__(self, rate_expression: Union[float, Callable, np.ndarray] = 0.0, name: str = ""):
#         self.name = name or f"Flow_{id(self)}"
        
#         if callable(rate_expression):
#             self.rate_expression = rate_expression
#         else:
#             # Constant rate
#             self.constant_rate = np.array(rate_expression, dtype=float)
#             self.rate_expression = lambda: self.constant_rate
        
#         # Stock connections
#         self._from_stock = None
#         self._to_stock = None
        
#         # History
#         self.history = []
        
#         print(f"Created Flow '{self.name}'")
    
#     def get_rate(self):
#         """Get current flow rate"""
#         try:
#             rate = self.rate_expression()
#             return np.array(rate, dtype=float)
#         except Exception as e:
#             print(f"Error calculating flow rate for {self.name}: {e}")
#             return np.array(0.0)
    
#     def connect(self, from_stock: Stock = None, to_stock: Stock = None):
#         """Connect flow between stocks"""
#         if from_stock:
#             from_stock.add_outflow(self)
#         if to_stock:
#             to_stock.add_inflow(self)
#         return self
    
#     def __rshift__(self, to_stock: Stock):
#         """Enable flow >> stock syntax"""
#         to_stock.add_inflow(self)
#         return self
    
#     def __lshift__(self, from_stock: Stock):
#         """Enable stock << flow syntax"""
#         from_stock.add_outflow(self)
#         return self

# class Model:
#     """Main simulation model"""
    
#     def __init__(self, dt: float = 1.0):
#         self.dt = dt
#         self.time = 0.0
#         self.stocks = []
#         self.flows = []
#         self.time_history = [0.0]
    
#     def add_stock(self, stock: Stock):
#         """Add stock to model"""
#         if stock not in self.stocks:
#             self.stocks.append(stock)
#         return stock
    
#     def add_flow(self, flow: Flow):
#         """Add flow to model"""
#         if flow not in self.flows:
#             self.flows.append(flow)
#         return flow
    
#     def step(self):
#         """Perform one simulation step"""
#         # Update all stocks
#         for stock in self.stocks:
#             stock.update(self.dt)
        
#         # Record flow history
#         for flow in self.flows:
#             flow.history.append(flow.get_rate())
        
#         # Update time
#         self.time += self.dt
#         self.time_history.append(self.time)
    
#     def run(self, duration: float):
#         """Run simulation for specified duration"""
#         steps = int(duration / self.dt)
#         for _ in range(steps):
#             self.step()
        
#         print(f"\nAfter {duration} time units:")
#         for stock in self.stocks:
#             print(f"  {stock}")
    
#     def reset(self):
#         """Reset all stocks and flows"""
#         self.time = 0.0
#         self.time_history = [0.0]
#         for stock in self.stocks:
#             stock.reset()
#         for flow in self.flows:
#             flow.history = []
    
#     def plot(self, stocks: List[Stock] = None):
#         """Plot stock values over time"""
#         if stocks is None:
#             stocks = self.stocks
        
#         plt.figure(figsize=(12, 8))
        
#         # Plot stocks
#         plt.subplot(2, 1, 1)
#         for stock in stocks:
#             if stock.values.ndim == 0:
#                 # Scalar stock
#                 values = [h.item() if h.ndim == 0 else h for h in stock.history]
#                 plt.plot(self.time_history, values, label=stock.name, linewidth=2)
#             else:
#                 # Multi-dimensional - plot total
#                 totals = [np.sum(h) for h in stock.history]
#                 plt.plot(self.time_history, totals, label=f"{stock.name} (total)", linewidth=2)
        
#         plt.xlabel('Time')
#         plt.ylabel('Stock Value')
#         plt.title('Stock Values Over Time')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
        
#         # Plot flows
#         plt.subplot(2, 1, 2)
#         for flow in self.flows:
#             if flow.history:
#                 if isinstance(flow.history[0], np.ndarray) and flow.history[0].ndim > 0:
#                     # Multi-dimensional flow - plot total
#                     totals = [np.sum(h) for h in flow.history]
#                     plt.plot(self.time_history[1:], totals, label=f"{flow.name} (total)", linewidth=2)
#                 else:
#                     # Scalar flow
#                     values = [h.item() if hasattr(h, 'item') else h for h in flow.history]
#                     plt.plot(self.time_history[1:], values, label=flow.name, linewidth=2)
        
#         plt.xlabel('Time')
#         plt.ylabel('Flow Rate')
#         plt.title('Flow Rates Over Time')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.show()

# # Convenience functions for easy import
# def stock(dim=None, values=0.0, name=""):
#     """Create a stock with specified dimensions and values"""
#     return Stock(dim=dim, values=values, name=name)

# def flow(rate=0.0, name=""):
#     """Create a flow with specified rate"""
#     return Flow(rate_expression=rate, name=name)

# def model(dt=1.0):
#     """Create a simulation model"""
#     return Model(dt=dt)



"""
sim_SD.py - Enhanced System Dynamics Module
Advanced AnyLogic-style Stock and Flow implementation with time management
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from enum import Enum
from typing import Union, List, Callable, Any, Optional
import json
import warnings

class TimeUnit(Enum):
    """Time units for simulation"""
    SECOND = ("second", "seconds", 1)
    MINUTE = ("minute", "minutes", 60)
    HOUR = ("hour", "hours", 3600)
    DAY = ("day", "days", 86400)
    WEEK = ("week", "weeks", 604800)
    MONTH = ("month", "months", 2629746)  # Average month
    YEAR = ("year", "years", 31556952)   # Average year
    
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
    
    def __init__(self, dim: Union[int, tuple] = None, values: Union[float, List, np.ndarray] = 0.0, 
                 name: str = "", min_value: float = 0.0, max_value: float = np.inf, 
                 units: str = ""):
        self.name = name or f"Stock_{id(self)}"
        self.units = units
        self.min_value = min_value
        self.max_value = max_value
        
        # Handle dimensions and initial values
        if dim is None:
            self.values = np.array(values, dtype=float)
            self.dim = ()
        elif isinstance(dim, int):
            if isinstance(values, (int, float)):
                self.values = np.full(dim, values, dtype=float)
            else:
                self.values = np.array(values, dtype=float)
                if self.values.shape[0] != dim:
                    raise ValueError(f"Values length {len(values)} doesn't match dimension {dim}")
            self.dim = (dim,)
        elif isinstance(dim, tuple):
            if isinstance(values, (int, float)):
                self.values = np.full(dim, values, dtype=float)
            else:
                self.values = np.array(values, dtype=float)
                if self.values.shape != dim:
                    raise ValueError(f"Values shape {self.values.shape} doesn't match dimension {dim}")
            self.dim = dim
        
        # Validate initial values
        self.values = np.clip(self.values, self.min_value, self.max_value)
        self.initial_values = np.copy(self.values)
        
        # Flow connections with names
        self._inflows = {}   # {flow_name: flow_object}
        self._outflows = {}  # {flow_name: flow_object}
        
        # Enhanced history tracking
        self.history = [np.copy(self.values)]
        self.time_history = [0.0]
        
        # Statistics
        self.min_recorded = np.copy(self.values)
        self.max_recorded = np.copy(self.values)
        
        print(f"Created Stock '{self.name}': dim={self.dim}, shape={self.values.shape}, units={self.units}")
    
    @property
    def inflows(self):
        """Get dictionary of inflows"""
        return self._inflows
    
    @property
    def outflows(self):
        """Get dictionary of outflows"""
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
        net_flow = np.zeros_like(self.values)
        
        # Add inflows
        for flow in self._inflows.values():
            try:
                flow_rate = flow.get_rate()
                if isinstance(flow_rate, np.ndarray):
                    if flow_rate.shape != self.values.shape:
                        raise ValueError(f"Inflow rate shape {flow_rate.shape} doesn't match stock shape {self.values.shape}")
                net_flow += flow_rate
            except Exception as e:
                warnings.warn(f"Error in inflow calculation for {self.name}: {e}")
        
        # Subtract outflows
        for flow in self._outflows.values():
            try:
                flow_rate = flow.get_rate()
                if isinstance(flow_rate, np.ndarray):
                    if flow_rate.shape != self.values.shape:
                        raise ValueError(f"Outflow rate shape {flow_rate.shape} doesn't match stock shape {self.values.shape}")
                net_flow -= flow_rate
            except Exception as e:
                warnings.warn(f"Error in outflow calculation for {self.name}: {e}")
        
        return net_flow
    
    def update(self, dt: float, sim_time: float):
        """Update stock value based on flows"""
        net_flow = self.get_net_flow(dt)
        
        # Calculate potential new value
        potential_new_value = self.values + net_flow * dt
        
        # Apply constraints
        self.values = np.clip(potential_new_value, self.min_value, self.max_value)
        
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
    
    def total(self):
        """Get total value"""
        return np.sum(self.values)
    
    def average(self):
        """Get average value"""
        return np.mean(self.values)
    
    def get_statistics(self):
        """Get comprehensive statistics"""
        return {
            'current': self.total(),
            'initial': np.sum(self.initial_values),
            'min': np.sum(self.min_recorded),
            'max': np.sum(self.max_recorded),
            'final_time': self.time_history[-1] if self.time_history else 0,
            'shape': self.values.shape,
            'units': self.units
        }
    
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
        
        # Enhanced history
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
    """Enhanced simulation model with comprehensive time management"""
    
    def __init__(self, time_unit: Union[str, TimeUnit] = TimeUnit.DAY, 
                 start_time: Union[datetime, str, None] = None,
                 end_time: Union[datetime, str, int, float, None] = None,
                 dt: Optional[float] = None,
                 name: str = "Simulation Model"):
        
        # Time management
        self.time_unit = TimeUnit.from_string(time_unit) if isinstance(time_unit, str) else time_unit
        self.name = name
        
        # Set default dt (1/1000 of time unit, but ensure reasonable values)
        if dt is None:
            # if self.time_unit == TimeUnit.SECOND:
            #     self.dt = 0.001  # 1 millisecond
            # elif self.time_unit == TimeUnit.MINUTE:
            #     self.dt = 0.06   # 3.6 seconds
            # elif self.time_unit == TimeUnit.HOUR:
            #     self.dt = 3.6    # 3.6 minutes
            # elif self.time_unit == TimeUnit.DAY:
            #     self.dt = 0.024  # ~35 minutes
            # else:
            self.dt = 1.0 / 5000.0  # Generic 1/1000
        else:
            self.dt = dt
        
        # Start time handling
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
                # Duration in time units
                self.duration = float(end_time)
                self.end_datetime = self._add_time_units(self.start_datetime, end_time)
            elif isinstance(end_time, str):
                self.end_datetime = datetime.fromisoformat(end_time)
                self.duration = self._calculate_duration()
            elif isinstance(end_time, datetime):
                self.end_datetime = end_time
                self.duration = self._calculate_duration()
        
        # Simulation state
        self.time = 0.0  # Simulation time in time units
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
        
        print(f"Created Model '{self.name}':")
        print(f"  Time Unit: {self.time_unit.singular}")
        print(f"  Start: {self.start_datetime}")
        print(f"  End: {self.end_datetime}")
        print(f"  Duration: {self.duration} {self.time_unit.plural if self.duration != 1 else self.time_unit.singular}")
        print(f"  Time Step (dt): {self.dt} {self.time_unit.plural}")
    
    def _add_time_units(self, base_datetime: datetime, units: float) -> datetime:
        """Add time units to datetime"""
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
        return stock
    
    def add_flow(self, flow: Flow):
        """Add flow to model"""
        if flow not in self.flows:
            self.flows.append(flow)
        return flow
    
    def get_current_datetime(self):
        """Get current simulation datetime"""
        return self._add_time_units(self.start_datetime, self.time)
    
    def step(self):
        """Perform one simulation step"""
        # Update current datetime
        self.current_datetime = self.get_current_datetime()
        
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
    
    def run(self, duration: Optional[float] = None, until: Optional[datetime] = None, 
            max_steps: Optional[int] = None, progress_callback: Optional[Callable] = None):
        """Run simulation with flexible stopping conditions"""
        
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
            
            # Check stopping conditions
            if target_time is not None and self.time >= target_time:
                break
            
            if self.max_steps is not None and self.step_count >= self.max_steps:
                break
            
            # Progress callback
            if progress_callback and self.step_count % 100 == 0:
                progress_callback(self)
        
        self.is_running = False
        
        # Print results
        steps_taken = self.step_count - start_step
        print(f"\nSimulation completed:")
        print(f"  Steps taken: {steps_taken}")
        print(f"  Final time: {self.time:.4f} {self.time_unit.plural}")
        print(f"  Final datetime: {self.current_datetime}")
        
        # Update results
        self._update_results()
        
        return self.results
    
    def _update_results(self):
        """Update simulation results"""
        self.results = {
            'simulation_time': self.time,
            'steps': self.step_count,
            'start_datetime': self.start_datetime,
            'end_datetime': self.current_datetime,
            'time_unit': self.time_unit.singular,
            'stocks': {stock.name: stock.get_statistics() for stock in self.stocks},
            'flows': {flow.name: flow.get_statistics() for flow in self.flows}
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
    
    def plot(self, stocks: List[Stock] = None, flows: List[Flow] = None, 
             use_datetime: bool = False, save_path: str = None):
        """Enhanced plotting with datetime support"""
        if stocks is None:
            stocks = self.stocks
        if flows is None:
            flows = self.flows
        
        # Choose time axis
        if use_datetime and len(self.datetime_history) > 0:
            time_axis = self.datetime_history
            time_label = 'Date/Time'
        else:
            time_axis = self.time_history
            time_label = f'Time ({self.time_unit.plural})'
        
        # Create subplots
        n_plots = (1 if stocks else 0) + (1 if flows else 0)
        if n_plots == 0:
            return
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 6*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot stocks
        if stocks:
            ax = axes[plot_idx]
            for stock in stocks:
                if stock.values.ndim == 0:
                    values = [h.item() if h.ndim == 0 else h for h in stock.history]
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
        
        # Plot flows
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
    
    def print_summary(self):
        """Print comprehensive simulation summary"""
        print(f"\n{'='*60}")
        print(f"SIMULATION SUMMARY: {self.name}")
        print(f"{'='*60}")
        print(f"Time Settings:")
        print(f"  Unit: {self.time_unit.singular}")
        print(f"  Start: {self.start_datetime}")
        print(f"  End: {self.current_datetime}")
        print(f"  Duration: {self.time:.4f} {self.time_unit.plural}")
        print(f"  Time Step: {self.dt}")
        print(f"  Total Steps: {self.step_count}")
        
        print(f"\nStocks ({len(self.stocks)}):")
        for stock in self.stocks:
            stats = stock.get_statistics()
            print(f"  {stock.name}: {stats['current']:.2f} {stock.units} (range: {stats['min']:.2f} - {stats['max']:.2f})")
        
        print(f"\nFlows ({len(self.flows)}):")
        for flow in self.flows:
            stats = flow.get_statistics()
            print(f"  {flow.name}: Total={stats['total_flow']:.2f}, Avg Rate={stats['avg_rate']:.4f} {flow.units}")

# Convenience functions
def stock(dim=None, values=0.0, name="", units="", min_value=0.0, max_value=np.inf):
    """Create a stock with specified parameters"""
    return Stock(dim=dim, values=values, name=name, units=units, min_value=min_value, max_value=max_value)

def flow(rate=0.0, name="", units="", min_rate=-np.inf, max_rate=np.inf):
    """Create a flow with specified parameters"""
    return Flow(rate_expression=rate, name=name, units=units, min_rate=min_rate, max_rate=max_rate)

def model(time_unit="day", start_time=None, end_time=None, dt=None, name="Simulation"):
    """Create a simulation model with time management"""
    return Model(time_unit=time_unit, start_time=start_time, end_time=end_time, dt=dt, name=name)

# Progress callback example
def print_progress(model_instance):
    """Example progress callback function"""
    progress = (model_instance.time / model_instance.duration * 100) if model_instance.duration else 0
    print(f"Progress: {progress:.1f}% - Time: {model_instance.time:.2f} {model_instance.time_unit.plural}")