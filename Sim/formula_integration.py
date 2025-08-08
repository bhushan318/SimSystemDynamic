# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 23:35:09 2025

@author: NagabhushanamTattaga
"""

# -*- coding: utf-8 -*-
"""
Integration of Formula Engine with System Dynamics Simulation
Shows how to connect the advanced formula engine with existing simulation classes
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass

# Import the formula engine
from formula_engine import (
    FormulaEngine, ModelContext, FormulaType, 
    enhance_stock_with_formulas, enhance_flow_with_formulas
)

# Import existing simulation classes (adjust import path as needed)
try:
    from simulation import Stock, Flow, Model
    SIMULATION_AVAILABLE = True
except ImportError:
    print("⚠️  Simulation module not found. Using mock classes for demonstration.")
    SIMULATION_AVAILABLE = False
    
    # Mock classes for demonstration
    class Stock:
        def __init__(self, values=0.0, name="", **kwargs):
            self.values = np.array([values]) if np.isscalar(values) else np.array(values)
            self.name = name
    
    class Flow:
        def __init__(self, rate_expression=None, name="", **kwargs):
            self.rate_expression = rate_expression or (lambda: 0.0)
            self.name = name
    
    class Model:
        def __init__(self, **kwargs):
            self.stocks = []
            self.flows = []
            self.time = 0.0

# ===============================================================================
# Enhanced Model with Formula Support
# ===============================================================================

class EnhancedSystemDynamicsModel(Model):
    """System Dynamics Model with advanced formula evaluation"""
    
    def __init__(self, name: str = "Enhanced Model", **kwargs):
        super().__init__(name=name, **kwargs)
        
        # Initialize formula engine
        self.formula_engine = FormulaEngine(max_cache_size=1000)
        
        # Model parameters
        self.parameters = {}
        self.auxiliaries = {}
        self.constants = {}
        
        # Formula management
        self.formula_dependencies = {}
        self.evaluation_order = []
        
        # Enhance existing classes
        if SIMULATION_AVAILABLE:
            enhance_stock_with_formulas(Stock)
            enhance_flow_with_formulas(Flow)
        
        print(f"✅ Enhanced System Dynamics Model '{name}' initialized with formula engine")
    
    def add_parameter(self, name: str, value: float, description: str = ""):
        """Add a model parameter"""
        self.parameters[name] = value
        print(f"Added parameter: {name} = {value}")
    
    def add_constant(self, name: str, value: float, description: str = ""):
        """Add a model constant"""
        self.constants[name] = value
        print(f"Added constant: {name} = {value}")
    
    def add_auxiliary(self, name: str, formula: str, description: str = ""):
        """Add auxiliary calculation with formula"""
        self.auxiliaries[name] = {
            'formula': formula,
            'value': 0.0,
            'description': description
        }
        
        # Validate formula
        context = self._create_current_context()
        validation = self.formula_engine.validate_formula(formula, context)
        
        if not validation['valid']:
            warnings.warn(f"Invalid auxiliary formula for '{name}': {validation['errors']}")
        
        print(f"Added auxiliary: {name} = {formula}")
    
    def add_stock_with_formula(self, name: str, initial_formula: str, 
                              dimensions=None, units: str = "", **kwargs):
        """Add stock with formula-based initial value"""
        
        # Create stock with temporary initial value
        stock = Stock(values=0.0, name=name, units=units, **kwargs)
        
        # Set initial formula
        if hasattr(stock, 'set_initial_formula'):
            stock.set_initial_formula(initial_formula, self.formula_engine)
        
        # Evaluate initial value
        context = self._create_current_context()
        try:
            initial_value = self.formula_engine.evaluate(
                initial_formula, context, FormulaType.INITIAL
            )
            
            if dimensions:
                stock.values = np.full(dimensions, initial_value)
            else:
                stock.values = np.array([initial_value])
                
        except Exception as e:
            warnings.warn(f"Failed to evaluate initial formula for stock '{name}': {e}")
            stock.values = np.array([0.0])
        
        self.add_stock(stock)
        print(f"Added stock: {name} with initial formula: {initial_formula}")
        return stock
    
    def add_flow_with_formula(self, name: str, rate_formula: str, 
                             from_stock=None, to_stock=None, units: str = "", **kwargs):
        """Add flow with formula-based rate calculation"""
        
        # Create flow
        flow = Flow(name=name, units=units, **kwargs)
        
        # Set rate formula
        if hasattr(flow, 'set_rate_formula'):
            flow.set_rate_formula(rate_formula, self.formula_engine)
        else:
            # Fallback: create rate expression manually
            def formula_rate_expression():
                context = self._create_current_context()
                return self.formula_engine.evaluate(rate_formula, context, FormulaType.RATE)
            flow.rate_expression = formula_rate_expression
        
        # Connect to stocks if specified
        if hasattr(flow, 'connect'):
            flow.connect(from_stock, to_stock)
        elif from_stock or to_stock:
            # Manual connection for basic Flow class
            if from_stock and hasattr(from_stock, 'add_outflow'):
                from_stock.add_outflow(flow, f"outflow_{name}")
            if to_stock and hasattr(to_stock, 'add_inflow'):
                to_stock.add_inflow(flow, f"inflow_{name}")
        
        self.add_flow(flow)
        print(f"Added flow: {name} with rate formula: {rate_formula}")
        return flow
    
    def _create_current_context(self) -> ModelContext:
        """Create context for formula evaluation"""
        
        # Get current stock values
        stock_values = {}
        for stock in self.stocks:
            if hasattr(stock.values, 'item') and stock.values.size == 1:
                stock_values[stock.name] = stock.values.item()
            else:
                stock_values[stock.name] = stock.values
        
        # Get auxiliary values  
        aux_values = {}
        for name, aux in self.auxiliaries.items():
            aux_values[name] = aux['value']
        
        # Combine parameters and constants
        all_parameters = {**self.parameters, **self.constants}
        
        return ModelContext(
            current_time=self.time,
            dt=getattr(self, 'dt', 1.0),
            start_time=getattr(self, 'start_time', 0.0),
            end_time=getattr(self, 'end_time', 100.0),
            stocks=stock_values,
            parameters=all_parameters,
            auxiliaries=aux_values
        )
    
    def update_auxiliaries(self):
        """Update all auxiliary calculations"""
        context = self._create_current_context()
        
        # Calculate auxiliaries in dependency order
        for name, aux in self.auxiliaries.items():
            try:
                new_value = self.formula_engine.evaluate(
                    aux['formula'], context, FormulaType.AUXILIARY
                )
                aux['value'] = new_value
                
                # Update context for dependent auxiliaries
                context.auxiliaries[name] = new_value
                
            except Exception as e:
                warnings.warn(f"Failed to evaluate auxiliary '{name}': {e}")
                aux['value'] = 0.0
    
    def step(self):
        """Enhanced step function with formula evaluation"""
        
        # Update auxiliaries first
        self.update_auxiliaries()
        
        # Update flow contexts
        context = self._create_current_context()
        for flow in self.flows:
            if hasattr(flow, 'set_context'):
                flow.set_context(context)
        
        # Call parent step method
        if hasattr(super(), 'step'):
            super().step()
        else:
            # Basic step implementation
            self.time += getattr(self, 'dt', 1.0)
    
    def validate_all_formulas(self) -> Dict[str, Any]:
        """Validate all formulas in the model"""
        
        validation_results = {
            'valid': True,
            'auxiliaries': {},
            'flows': {},
            'dependencies': {},
            'errors': [],
            'warnings': []
        }
        
        context = self._create_current_context()
        
        # Validate auxiliary formulas
        for name, aux in self.auxiliaries.items():
            validation = self.formula_engine.validate_formula(aux['formula'], context)
            validation_results['auxiliaries'][name] = validation
            
            if not validation['valid']:
                validation_results['valid'] = False
                validation_results['errors'].extend([f"Auxiliary {name}: {err}" for err in validation['errors']])
        
        # Validate flow formulas
        for flow in self.flows:
            if hasattr(flow, 'validate_formula'):
                validation = flow.validate_formula(context)
                validation_results['flows'][flow.name] = validation
                
                if not validation['valid']:
                    validation_results['valid'] = False
                    validation_results['errors'].extend([f"Flow {flow.name}: {err}" for err in validation['errors']])
        
        return validation_results
    
    def analyze_dependencies(self) -> Dict[str, Set[str]]:
        """Analyze formula dependencies across the model"""
        
        dependencies = {}
        
        # Auxiliary dependencies
        for name, aux in self.auxiliaries.items():
            deps = self.formula_engine.get_formula_dependencies(aux['formula'])
            dependencies[f"aux_{name}"] = deps
        
        # Flow dependencies
        for flow in self.flows:
            if hasattr(flow, 'get_formula_dependencies'):
                deps = flow.get_formula_dependencies()
                dependencies[f"flow_{flow.name}"] = deps
        
        return dependencies
    
    def get_formula_statistics(self) -> Dict[str, Any]:
        """Get comprehensive formula evaluation statistics"""
        return self.formula_engine.get_statistics()
    
    def export_model_structure(self, filepath: str):
        """Export model structure including formulas to JSON"""
        
        model_data = {
            'name': getattr(self, 'name', 'Unnamed Model'),
            'parameters': self.parameters,
            'constants': self.constants,
            'auxiliaries': {
                name: {
                    'formula': aux['formula'],
                    'description': aux.get('description', ''),
                    'current_value': aux['value']
                } for name, aux in self.auxiliaries.items()
            },
            'stocks': [
                {
                    'name': stock.name,
                    'current_value': stock.values.tolist() if hasattr(stock.values, 'tolist') else float(stock.values),
                    'units': getattr(stock, 'units', '')
                } for stock in self.stocks
            ],
            'flows': [
                {
                    'name': flow.name,
                    'formula': getattr(flow, '_rate_formula', 'N/A'),
                    'units': getattr(flow, 'units', '')
                } for flow in self.flows
            ],
            'dependencies': self.analyze_dependencies(),
            'validation': self.validate_all_formulas()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2, default=str)
        
        print(f"Model structure exported to: {filepath}")

# ===============================================================================
# Example Usage and Testing
# ===============================================================================

def create_population_growth_model():
    """Create example population growth model with formulas"""
    
    print("\n🏗️  Creating Population Growth Model with Formulas")
    print("=" * 60)
    
    # Create enhanced model
    model = EnhancedSystemDynamicsModel("Population Growth with Carrying Capacity")
    
    # Add parameters
    model.add_parameter("initial_population", 100, "Starting population")
    model.add_parameter("growth_rate", 0.03, "Annual growth rate")
    model.add_parameter("carrying_capacity", 1000, "Environmental carrying capacity")
    model.add_parameter("death_rate", 0.01, "Annual death rate")
    
    # Add constants
    model.add_constant("minimum_population", 1, "Minimum viable population")
    
    # Add auxiliaries with formulas
    model.add_auxiliary("population_pressure", 
                       "Population / carrying_capacity",
                       "Ratio of current to maximum population")
    
    model.add_auxiliary("effective_growth_rate",
                       "growth_rate * (1 - population_pressure) - death_rate",
                       "Growth rate adjusted for carrying capacity and deaths")
    
    model.add_auxiliary("net_birth_rate",
                       "MAX(0, effective_growth_rate)",
                       "Net positive growth rate")
    
    # Add stocks with formula-based initial values
    population = model.add_stock_with_formula(
        "Population", 
        "initial_population",
        units="individuals"
    )
    
    # Add flows with formula-based rates
    births = model.add_flow_with_formula(
        "Births",
        "Population * net_birth_rate",
        to_stock=population,
        units="individuals/year"
    )
    
    deaths = model.add_flow_with_formula(
        "Deaths", 
        "Population * death_rate",
        from_stock=population,
        units="individuals/year"
    )
    
    # Add a migration flow with conditional logic
    migration = model.add_flow_with_formula(
        "Migration",
        "IF_THEN_ELSE(TIME > 10, STEP(50, 15), 0)",
        to_stock=population,
        units="individuals/year"
    )
    
    return model

def run_enhanced_model_example():
    """Run comprehensive example of enhanced model"""
    
    # Create model
    model = create_population_growth_model()
    
    # Validate all formulas
    print("\n🔍 Validating Model Formulas:")
    validation = model.validate_all_formulas()
    
    if validation['valid']:
        print("   ✅ All formulas are valid")
    else:
        print("   ❌ Formula validation errors:")
        for error in validation['errors']:
            print(f"      - {error}")
    
    # Analyze dependencies
    print("\n🔗 Formula Dependencies:")
    dependencies = model.analyze_dependencies()
    for formula_name, deps in dependencies.items():
        if deps:
            print(f"   {formula_name}: {', '.join(deps)}")
    
    # Run simulation steps
    print("\n🚀 Running Simulation:")
    print(f"{'Time':<6} {'Population':<12} {'Births':<8} {'Deaths':<8} {'Growth Rate':<12}")
    print("-" * 55)
    
    for step in range(11):
        # Update model state
        model.update_auxiliaries()
        
        # Get current values
        population = model.stocks[0].values.item()
        growth_rate = model.auxiliaries['effective_growth_rate']['value']
        
        # Get flow rates (simplified)
        context = model._create_current_context()
        birth_rate = model.formula_engine.evaluate("Population * net_birth_rate", context)
        death_rate_val = model.formula_engine.evaluate("Population * death_rate", context)
        
        print(f"{model.time:<6.1f} {population:<12.1f} {birth_rate:<8.1f} {death_rate_val:<8.1f} {growth_rate:<12.3f}")
        
        # Step forward
        model.step()
        
        # Update stock (simplified for demonstration)
        net_change = birth_rate - death_rate_val
        if step >= 10:  # Add migration after time 10
            migration_flow = model.formula_engine.evaluate("IF_THEN_ELSE(TIME > 10, STEP(50, 15), 0)", context)
            net_change += migration_flow
        
        model.stocks[0].values = np.array([max(1, population + net_change * model.dt)])
    
    # Show statistics
    print("\n📊 Formula Engine Statistics:")
    stats = model.get_formula_statistics()
    print(f"   Total evaluations: {stats['evaluation']['total_evaluations']}")
    print(f"   Success rate: {stats['evaluation']['success_rate']:.1%}")
    print(f"   Cache hit rate: {stats['compilation']['cache_hit_rate']:.1%}")
    print(f"   Average evaluation time: {stats['evaluation']['avg_evaluation_time']:.6f}s")
    
    # Export model structure
    model.export_model_structure("population_model_structure.json")
    print("\n💾 Model structure exported to: population_model_structure.json")
    
    return model

def test_formula_integration():
    """Test formula engine integration"""
    
    print("🧪 Testing Formula Engine Integration")
    print("=" * 60)
    
    # Test basic integration
    model = EnhancedSystemDynamicsModel("Test Model")
    
    # Test parameter and auxiliary system
    model.add_parameter("test_param", 42)
    model.add_auxiliary("computed_value", "test_param * 2 + 10")
    
    model.update_auxiliaries()
    computed = model.auxiliaries['computed_value']['value']
    expected = 42 * 2 + 10
    
    status = "✅" if abs(computed - expected) < 1e-10 else "❌"
    print(f"{status} Auxiliary calculation: {computed} (expected: {expected})")
    
    # Test system dynamics functions
    model.time = 5.0
    context = model._create_current_context()
    
    test_cases = [
        ("STEP(100, 3)", 100),
        ("PULSE(50, 2, 4)", 50),
        ("TIME", 5.0),
        ("IF_THEN_ELSE(test_param > 40, 1, 0)", 1)
    ]
    
    print("\nSystem Dynamics Functions:")
    for formula, expected in test_cases:
        try:
            result = model.formula_engine.evaluate(formula, context)
            status = "✅" if abs(result - expected) < 1e-10 else "❌"
            print(f"   {status} {formula} = {result}")
        except Exception as e:
            print(f"   ❌ {formula} = ERROR: {e}")
    
    print("\n✅ Integration test completed!")

# ===============================================================================
# Main Execution
# ===============================================================================

if __name__ == "__main__":
    # Run integration test
    test_formula_integration()
    
    # Run full example
    print("\n" + "=" * 80)
    enhanced_model = run_enhanced_model_example()
    
    print("\n✅ Formula Engine Integration demonstration completed!")
    print("\nKey Features Demonstrated:")
    print("   • Secure formula evaluation with AST validation")
    print("   • System dynamics functions (STEP, PULSE, IF_THEN_ELSE, etc.)")
    print("   • Formula caching and performance optimization")
    print("   • Dependency analysis and validation")
    print("   • Integration with existing Stock and Flow classes")
    print("   • Real-time auxiliary calculations")
    print("   • Comprehensive error handling and statistics")