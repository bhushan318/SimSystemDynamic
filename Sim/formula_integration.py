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


# Add this import near the top
import warnings
from typing import Dict, Any, List, Optional, Callable

# Update the existing try/except block for validation imports
try:
    from unified_validation import get_unified_validator, ValidationLevel, ValidationReport
    UNIFIED_VALIDATION_AVAILABLE = True
except ImportError:
    UNIFIED_VALIDATION_AVAILABLE = False
    warnings.warn("Unified validation framework not available - using fallback validation")

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
                # Try to get context from flow first, then create new one
                if hasattr(flow, '_current_context') and flow._current_context:
                    context = flow._current_context
                else:
                    context = self._create_current_context()
                return self.formula_engine.evaluate(rate_formula, context, FormulaType.RATE)
            
            flow.rate_expression = formula_rate_expression
            flow._current_context = None  # Initialize context storage
        
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
        """Enhanced auxiliary update with validation"""
        
        # Pre-validation check
        try:
            from unified_validation import get_unified_validator
            validator = get_unified_validator()
            
            # Quick validation of auxiliary formulas before evaluation
            for name, aux in self.auxiliaries.items():
                formula_report = validator.validate_formula(aux['formula'])
                if not formula_report.is_valid:
                    critical_issues = [i for i in formula_report.issues if i.severity.value == 'critical']
                    if critical_issues:
                        warnings.warn(f"Critical validation issues in auxiliary '{name}': {[i.message for i in critical_issues]}")
                        aux['value'] = 0.0  # Safe fallback
                        continue
        except ImportError:
            pass  # Continue with original validation if unified framework not available
        
        # Create evaluation context
        context = self._create_current_context()
        
        # Sort auxiliaries by dependency order (if available)
        try:
            sorted_auxiliaries = self._sort_auxiliaries_by_dependencies()
        except Exception:
            sorted_auxiliaries = list(self.auxiliaries.items())
        
        # Update auxiliaries in dependency order
        for name, aux in sorted_auxiliaries:
            try:
                new_value = self.formula_engine.evaluate(
                    aux['formula'], context, FormulaType.AUXILIARY
                )
                aux['value'] = new_value
                
                # Update context for dependent auxiliaries
                context.auxiliaries[name] = new_value
                
            except Exception as e:
                warnings.warn(f"Failed to evaluate auxiliary '{name}': {e}")
                aux['value'] = 0.0  # Safe fallback
    
    def _sort_auxiliaries_by_dependencies(self) -> list:
        """Sort auxiliaries by their dependencies"""
        
        try:
            from unified_validation import get_unified_validator
            validator = get_unified_validator()
            
            # Build dependency graph
            dependencies = {}
            for name, aux in self.auxiliaries.items():
                report = validator.validate_formula(aux['formula'])
                deps = []
                for issue in report.issues:
                    if issue.details and 'dependencies' in issue.details:
                        deps.extend(issue.details['dependencies'])
                dependencies[name] = [d for d in deps if d in self.auxiliaries]
            
            # Topological sort
            sorted_names = self._topological_sort(dependencies)
            return [(name, self.auxiliaries[name]) for name in sorted_names if name in self.auxiliaries]
        
        except Exception:
            # Fallback to original order
            return list(self.auxiliaries.items())
    
    def _topological_sort(self, dependencies: Dict[str, list]) -> list:
        """Perform topological sort on dependency graph"""
        
        # Simple topological sort implementation
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(node):
            if node in temp_visited:
                return  # Circular dependency - skip
            if node in visited:
                return
            
            temp_visited.add(node)
            for dep in dependencies.get(node, []):
                visit(dep)
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)
        
        for node in dependencies:
            if node not in visited:
                visit(node)
        
        return result
    
    def step(self):
        """Enhanced step function with formula evaluation"""
        
        # Update auxiliaries first
        self.update_auxiliaries()
        
        # Update flow contexts
        context = self._create_current_context()
        for flow in self.flows:
            if hasattr(flow, 'set_context'):
                flow.set_context(context)
                # Also set context directly for flows with formulas
            if hasattr(flow, '_current_context'):
                flow._current_context = context
        
        # Call parent step method
        if hasattr(super(), 'step'):
            super().step()
        else:
            # Basic step implementation
            self.time += getattr(self, 'dt', 1.0)
    
    
    def validate_comprehensive(self) -> Dict[str, Any]:
        """Run comprehensive validation using unified framework"""
        try:
            from unified_validation import get_unified_validator, ValidationLevel
            validator = get_unified_validator()
            report = validator.validate_all(self, 'model', ValidationLevel.COMPREHENSIVE)
            
            return {
                'valid': report.is_valid,
                'framework': 'unified_comprehensive',
                'total_issues': len(report.issues),
                'issues_by_severity': {
                    'critical': [i for i in report.issues if i.severity.value == 'critical'],
                    'error': [i for i in report.issues if i.severity.value == 'error'],
                    'warning': [i for i in report.issues if i.severity.value == 'warning'],
                    'info': [i for i in report.issues if i.severity.value == 'info']
                },
                'issues_by_category': self._group_issues_by_category(report.issues),
                'validation_statistics': validator.get_validation_statistics()
            }
        except Exception as e:
            return {
                'valid': False,
                'framework': 'error',
                'error': str(e),
                'fallback_used': False
            }
    
    def validate_security(self) -> Dict[str, Any]:
        """Run security-focused validation"""
        try:
            from unified_validation import get_unified_validator, ValidationLevel
            validator = get_unified_validator()
            
            # Focus on formula security validation
            report = validator.validate_formula(
                '\n'.join([aux['formula'] for aux in self.auxiliaries.values()]),
                self._create_current_context()
            )
            
            return {
                'secure': report.is_valid,
                'security_issues': [i for i in report.issues if i.category == 'formula_security'],
                'critical_vulnerabilities': [i for i in report.issues if i.severity.value == 'critical'],
                'framework': 'unified_security'
            }
        except Exception as e:
            return {
                'secure': False,
                'framework': 'error',
                'error': str(e)
            }
    
    def validate_performance(self) -> Dict[str, Any]:
        """Validate model for performance issues"""
        try:
            from unified_validation import get_unified_validator, ValidationLevel
            validator = get_unified_validator()
            report = validator.validate_all(self, 'model', ValidationLevel.STANDARD)
            
            performance_issues = [i for i in report.issues if 'performance' in i.category.lower()]
            complexity_issues = [i for i in report.issues if 'complexity' in i.message.lower()]
            
            return {
                'performance_optimal': len(performance_issues) == 0,
                'performance_issues': performance_issues,
                'complexity_issues': complexity_issues,
                'optimization_suggestions': [i.suggestion for i in performance_issues + complexity_issues],
                'framework': 'unified_performance'
            }
        except Exception as e:
            return {
                'performance_optimal': False,
                'framework': 'error',
                'error': str(e)
            }
    
    def _group_issues_by_category(self, issues) -> Dict[str, list]:
        """Group validation issues by category"""
        categories = {}
        for issue in issues:
            if issue.category not in categories:
                categories[issue.category] = []
            categories[issue.category].append({
                'severity': issue.severity.value,
                'message': issue.message,
                'suggestion': issue.suggestion,
                'element': issue.element_name
            })
        return categories
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of all validation aspects"""
        
        summary = {
            'timestamp': None,
            'model_name': getattr(self, 'name', 'unnamed_model'),
            'validation_aspects': {}
        }
        
        # Run different validation aspects
        validation_aspects = [
            ('basic_formulas', lambda: self.validate_all_formulas()),
            ('comprehensive', lambda: self.validate_comprehensive()),
            ('security', lambda: self.validate_security()),
            ('performance', lambda: self.validate_performance())
        ]
        
        for aspect_name, aspect_func in validation_aspects:
            try:
                summary['validation_aspects'][aspect_name] = aspect_func()
            except Exception as e:
                summary['validation_aspects'][aspect_name] = {
                    'valid': False,
                    'error': str(e),
                    'framework': 'error'
                }
        
        # Overall validity
        summary['overall_valid'] = all(
            aspect.get('valid', aspect.get('secure', aspect.get('performance_optimal', False)))
            for aspect in summary['validation_aspects'].values()
            if isinstance(aspect, dict) and 'error' not in aspect
        )
        
        return summary


    def validate_all_formulas(self) -> Dict[str, Any]:
        """Validate all formulas using unified validation framework"""
        
        try:
            from unified_validation import get_unified_validator
            
            validator = get_unified_validator()
            
            # DON'T validate the whole model - just validate individual formulas
            # This breaks the recursion
            all_issues = []
            
            # Validate each auxiliary formula individually
            for name, aux in self.auxiliaries.items():
                formula_report = validator.validate_formula(aux['formula'])
                for issue in formula_report.issues:
                    issue.element_name = f"auxiliary_{name}"
                    all_issues.append(issue)
            
            # Validate flow formulas individually
            for flow in getattr(self, 'flows', []):
                if hasattr(flow, 'rate_formula'):
                    flow_report = validator.validate_formula(flow.rate_formula)
                    for issue in flow_report.issues:
                        issue.element_name = getattr(flow, 'name', 'unnamed_flow')
                        all_issues.append(issue)
            
            # Convert to legacy format
            validation_results = {
                'valid': len([i for i in all_issues if i.severity.value in ['error', 'critical']]) == 0,
                'framework': 'unified',
                'auxiliaries': {},
                'flows': {},
                'dependencies': {},
                'errors': [i.message for i in all_issues if i.severity.value in ['error', 'critical']],
                'warnings': [i.message for i in all_issues if i.severity.value == 'warning'],
                'info': [i.message for i in all_issues if i.severity.value == 'info']
            }
            
            return validation_results
            
        except Exception as e:
            return self._fallback_formula_validation()




    def validate_model_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive model validation (the full approach we wanted)"""
        
        try:
            from unified_validation import get_unified_validator, ValidationLevel
            
            validator = get_unified_validator()
            
            # Use a DIFFERENT validation approach to avoid recursion
            # Validate model structure but SKIP formula validation to prevent loops
            report = validator.validators['mass_conservation'].validate(self, ValidationLevel.STANDARD)
            
            # Add flow compatibility validation
            flow_report = validator.validators['flow_compatibility'].validate(self, ValidationLevel.STANDARD) 
            report.merge(flow_report)
            
            # Add our OWN formula validation (not through the unified validator)
            formula_results = self.validate_all_formulas()  # This calls the safe version above
            
            # Combine everything
            return {
                'valid': report.is_valid and formula_results['valid'],
                'framework': 'unified_comprehensive',
                'formula_validation': formula_results,
                'structural_validation': {
                    'valid': report.is_valid,
                    'issues': [{'severity': i.severity.value, 'message': i.message, 'category': i.category} for i in report.issues]
                },
                'total_issues': len(report.issues) + len(formula_results.get('errors', [])) + len(formula_results.get('warnings', []))
            }
            
        except Exception as e:
            return {
                'valid': False,
                'framework': 'comprehensive_error',
                'error': str(e),
                'fallback_used': True
            }
    
    def validate_quick(self) -> bool:
        """Ultra-fast validation - just returns True/False"""
        try:
            result = self.validate_all_formulas()
            return result['valid']
        except:
            return False


    
    def _fallback_formula_validation(self) -> Dict[str, Any]:
        """Fallback validation using original formula engine"""
        
        validation_results = {
            'valid': True,
            'framework': 'fallback',
            'auxiliaries': {},
            'flows': {},
            'dependencies': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            context = self._create_current_context()
            
            # Validate auxiliary formulas using formula engine
            for name, aux in self.auxiliaries.items():
                try:
                    if hasattr(self, 'formula_engine') and self.formula_engine:
                        validation = self.formula_engine.validate_formula(aux['formula'], context)
                        validation_results['auxiliaries'][name] = validation
                        
                        if not validation.get('valid', True):
                            validation_results['valid'] = False
                            validation_results['errors'].extend([f"Auxiliary {name}: {err}" for err in validation.get('errors', [])])
                        
                        validation_results['warnings'].extend([f"Auxiliary {name}: {warn}" for warn in validation.get('warnings', [])])
                        
                    else:
                        # Basic syntax check if no formula engine
                        try:
                            compile(aux['formula'], '<string>', 'eval')
                            validation_results['auxiliaries'][name] = {'valid': True, 'errors': [], 'warnings': []}
                        except SyntaxError as se:
                            validation_results['valid'] = False
                            validation_results['auxiliaries'][name] = {'valid': False, 'errors': [str(se)], 'warnings': []}
                            validation_results['errors'].append(f"Auxiliary {name}: {str(se)}")
                
                except Exception as aux_error:
                    validation_results['warnings'].append(f"Could not validate auxiliary {name}: {str(aux_error)}")
            
            # Validate flow formulas
            for flow in getattr(self, 'flows', []):
                flow_name = getattr(flow, 'name', 'unnamed_flow')
                try:
                    if hasattr(flow, 'validate_formula') and callable(flow.validate_formula):
                        flow_validation = flow.validate_formula(context)
                        validation_results['flows'][flow_name] = flow_validation
                        
                        if not flow_validation.get('valid', True):
                            validation_results['valid'] = False
                            validation_results['errors'].extend([f"Flow {flow_name}: {err}" for err in flow_validation.get('errors', [])])
                    
                    elif hasattr(flow, 'rate_formula'):
                        # Basic validation for flow rate formula
                        try:
                            if hasattr(self, 'formula_engine') and self.formula_engine:
                                flow_validation = self.formula_engine.validate_formula(flow.rate_formula, context)
                            else:
                                compile(flow.rate_formula, '<string>', 'eval')
                                flow_validation = {'valid': True, 'errors': [], 'warnings': []}
                            
                            validation_results['flows'][flow_name] = flow_validation
                            
                            if not flow_validation.get('valid', True):
                                validation_results['valid'] = False
                                validation_results['errors'].extend([f"Flow {flow_name}: {err}" for err in flow_validation.get('errors', [])])
                        
                        except Exception as flow_error:
                            validation_results['valid'] = False
                            validation_results['flows'][flow_name] = {'valid': False, 'errors': [str(flow_error)], 'warnings': []}
                            validation_results['errors'].append(f"Flow {flow_name}: {str(flow_error)}")
                
                except Exception as flow_error:
                    validation_results['warnings'].append(f"Could not validate flow {flow_name}: {str(flow_error)}")
        
        except Exception as e:
            validation_results['errors'].append(f"Fallback validation failed: {str(e)}")
            validation_results['valid'] = False
        
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


def test_formula_integration_validation():
    """Test the unified validation integration in formula_integration"""
    print("🧪 Testing Formula Integration with Unified Validation")
    print("=" * 60)
    
    try:
        # Create test enhanced model
        model = EnhancedSystemDynamicsModel("Test Model")
        
        # Add test auxiliaries
        model.add_auxiliary("growth_rate", "0.02")
        model.add_auxiliary("capacity", "10000")
        model.add_auxiliary("complex_calc", "growth_rate * capacity * (1 - Population / capacity)")
        
        # Test 1: Basic formula validation
        print("\n📊 Test 1: Basic formula validation")
        validation_result = model.validate_all_formulas()
        print(f"   Framework used: {validation_result.get('framework', 'unknown')}")
        print(f"   Valid: {validation_result['valid']}")
        print(f"   Total issues: {len(validation_result.get('errors', []) + validation_result.get('warnings', []))}")
        
        # Test 2: Comprehensive validation
        print("\n📊 Test 2: Comprehensive validation")
        comprehensive_result = model.validate_comprehensive()
        print(f"   Framework used: {comprehensive_result.get('framework', 'unknown')}")
        print(f"   Valid: {comprehensive_result.get('valid', False)}")
        print(f"   Total issues: {comprehensive_result.get('total_issues', 0)}")
        
        # Test 3: Security validation
        print("\n📊 Test 3: Security validation")
        security_result = model.validate_security()
        print(f"   Framework used: {security_result.get('framework', 'unknown')}")
        print(f"   Secure: {security_result.get('secure', False)}")
        print(f"   Security issues: {len(security_result.get('security_issues', []))}")
        
        # Test 4: Performance validation
        print("\n📊 Test 4: Performance validation")
        performance_result = model.validate_performance()
        print(f"   Framework used: {performance_result.get('framework', 'unknown')}")
        print(f"   Performance optimal: {performance_result.get('performance_optimal', False)}")
        
        # Test 5: Validation summary
        print("\n📊 Test 5: Validation summary")
        summary = model.get_validation_summary()
        print(f"   Model: {summary['model_name']}")
        print(f"   Overall valid: {summary['overall_valid']}")
        print(f"   Validation aspects: {len(summary['validation_aspects'])}")
        
        print("\n✅ Formula integration validation test completed!")
        
        
        # For regular use (fast, safe)
        result = model.validate_all_formulas()
        
        # When you want everything checked (comprehensive)
        result = model.validate_model_comprehensive()
        
        # When you just need to know if it's valid (fastest)
        is_valid = model.validate_quick()
        
    except Exception as e:
        print(f"\n❌ Formula integration validation test failed: {str(e)}")


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
    test_formula_integration_validation()


