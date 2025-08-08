# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 00:47:05 2025

@author: NagabhushanamTattaga
"""

# -*- coding: utf-8 -*-
"""
Comprehensive Model Validation Framework for System Dynamics
Implements structural validation and behavioral analysis
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
import json
from scipy.optimize import fsolve, minimize
from scipy.linalg import eigvals
import time

# ===============================================================================
# Validation Result Data Structures
# ===============================================================================

class ValidationLevel(Enum):
    """Validation thoroughness levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    DEEP = "deep"

class IssueSeverity(Enum):
    """Issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationIssue:
    """Individual validation issue"""
    severity: IssueSeverity
    category: str  # e.g., "mass_balance", "circular_dependency"
    message: str
    suggestion: str
    element_name: Optional[str] = None
    element_type: Optional[str] = None  # 'stock', 'flow', 'parameter'
    details: Optional[Dict[str, Any]] = None





@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    model_name: str
    validation_level: ValidationLevel
    timestamp: float = 0.0  # Add default value
    
    # Results
    is_valid: bool = True
    issues: List[ValidationIssue] = None
    
    # Statistics
    total_issues: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    
    # Analysis results
    structural_analysis: Optional[Dict[str, Any]] = None
    behavioral_analysis: Optional[Dict[str, Any]] = None
    performance_analysis: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        self.timestamp = time.time()
    
    def add_issue(self, issue: ValidationIssue):
        """Add validation issue and update statistics"""
        self.issues.append(issue)
        self.total_issues += 1
        
        if issue.severity == IssueSeverity.ERROR or issue.severity == IssueSeverity.CRITICAL:
            self.error_count += 1
            self.is_valid = False
        elif issue.severity == IssueSeverity.WARNING:
            self.warning_count += 1
        else:
            self.info_count += 1
    
    def get_issues_by_severity(self, severity: IssueSeverity) -> List[ValidationIssue]:
        """Get issues of specific severity"""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_category(self, category: str) -> List[ValidationIssue]:
        """Get issues of specific category"""
        return [issue for issue in self.issues if issue.category == category]

# ===============================================================================
# Structural Validation Framework
# ===============================================================================

class StructuralValidator:
    """Validates model structure and consistency"""
    
    def __init__(self):
        self.validation_rules = [
            self._check_mass_balance,
            self._check_circular_dependencies,
            self._check_dimensional_consistency,
            self._check_unreachable_stocks,
            self._check_unbounded_growth,
            self._check_formula_dependencies,
            self._check_parameter_usage,
            self._check_unit_consistency
        ]
    
    def validate_structure(self, model: 'Model', level: ValidationLevel = ValidationLevel.STANDARD) -> List[ValidationIssue]:
        """Run structural validation checks"""
        issues = []
        
        print(f"🔍 Running structural validation ({level.value} level)...")
        
        for rule in self.validation_rules:
            try:
                rule_issues = rule(model, level)
                issues.extend(rule_issues)
            except Exception as e:
                issues.append(ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    category="validation_error",
                    message=f"Validation rule {rule.__name__} failed: {str(e)}",
                    suggestion="Check model configuration and validation framework"
                ))
        
        return issues
    
    def _check_mass_balance(self, model: 'Model', level: ValidationLevel) -> List[ValidationIssue]:
        """Check for mass balance violations"""
        issues = []
        
        try:
            # Check each stock for mass balance
            for stock in model.stocks:
                inflow_count = len(stock.inflows)
                outflow_count = len(stock.outflows)
                
                # Check for isolated stocks
                if inflow_count == 0 and outflow_count == 0:
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        category="mass_balance",
                        message=f"Stock '{stock.name}' has no flows",
                        suggestion="Add inflows/outflows or remove unused stock",
                        element_name=stock.name,
                        element_type="stock"
                    ))
                
                # Check for sink-only stocks (only outflows)
                elif inflow_count == 0 and outflow_count > 0:
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.INFO,
                        category="mass_balance",
                        message=f"Stock '{stock.name}' is a pure sink (only outflows)",
                        suggestion="Verify this is intentional for your model",
                        element_name=stock.name,
                        element_type="stock"
                    ))
                
                # Check for source-only stocks (only inflows)
                elif inflow_count > 0 and outflow_count == 0:
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.INFO,
                        category="mass_balance",
                        message=f"Stock '{stock.name}' is a pure source (only inflows)",
                        suggestion="Verify this is intentional for your model",
                        element_name=stock.name,
                        element_type="stock"
                    ))
            
            # Global mass balance check (if level >= STANDARD)
            if level.value in ['standard', 'comprehensive', 'deep']:
                mass_conservation = model.validate_mass_conservation()
                if not mass_conservation.get('is_conserved', True):
                    error_type = mass_conservation.get('type', 'unknown')
                    relative_error = mass_conservation.get('relative_error', 0)
                    
                    if error_type in ['pure_source_system', 'pure_sink_system']:
                        severity = IssueSeverity.INFO
                    elif relative_error > 0.1:
                        severity = IssueSeverity.ERROR
                    else:
                        severity = IssueSeverity.WARNING
                    
                    issues.append(ValidationIssue(
                        severity=severity,
                        category="mass_balance",
                        message=f"Mass conservation violation: {error_type} with {relative_error:.2e} relative error",
                        suggestion="Check flow formulas and ensure proper conservation",
                        details=mass_conservation
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity=IssueSeverity.ERROR,
                category="mass_balance",
                message=f"Mass balance check failed: {str(e)}",
                suggestion="Check model structure and flow definitions"
            ))
        
        return issues
    
    def _check_circular_dependencies(self, model: 'Model', level: ValidationLevel) -> List[ValidationIssue]:
        """Detect circular dependencies in formulas and flows"""
        issues = []
        
        try:
            # Build dependency graph
            G = nx.DiGraph()
            
            # Add nodes for stocks, flows, and auxiliaries
            for stock in model.stocks:
                G.add_node(f"stock_{stock.name}", type="stock")
            
            for flow in model.flows:
                G.add_node(f"flow_{flow.name}", type="flow")
            
            # Add auxiliaries if model has them
            if hasattr(model, 'auxiliaries'):
                for aux_name in model.auxiliaries:
                    G.add_node(f"aux_{aux_name}", type="auxiliary")
            
            # Add edges based on flow connections
            for stock in model.stocks:
                for flow_name, flow in stock.inflows.items():
                    G.add_edge(f"flow_{flow.name}", f"stock_{stock.name}", relationship="inflow")
                
                for flow_name, flow in stock.outflows.items():
                    G.add_edge(f"stock_{stock.name}", f"flow_{flow.name}", relationship="outflow")
            
            # Add formula dependencies (if enhanced model)
            if hasattr(model, 'formula_engine'):
                self._add_formula_dependencies(G, model)
            
            # Check for cycles
            try:
                cycles = list(nx.simple_cycles(G))
                for cycle in cycles:
                    cycle_str = ' → '.join(cycle)
                    
                    # Determine severity based on cycle type
                    has_stock = any('stock_' in node for node in cycle)
                    has_aux = any('aux_' in node for node in cycle)
                    
                    if has_aux and len(cycle) <= 3:
                        severity = IssueSeverity.WARNING
                        suggestion = "Check auxiliary calculations for logical dependencies"
                    elif has_stock:
                        severity = IssueSeverity.ERROR
                        suggestion = "Remove circular references in stock-flow relationships"
                    else:
                        severity = IssueSeverity.WARNING
                        suggestion = "Review formula dependencies"
                    
                    issues.append(ValidationIssue(
                        severity=severity,
                        category="circular_dependency",
                        message=f"Circular dependency detected: {cycle_str}",
                        suggestion=suggestion,
                        details={'cycle': cycle, 'cycle_length': len(cycle)}
                    ))
                    
            except nx.NetworkXError:
                pass  # No cycles found
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity=IssueSeverity.ERROR,
                category="circular_dependency",
                message=f"Circular dependency check failed: {str(e)}",
                suggestion="Check model structure and dependencies"
            ))
        
        return issues
    
    def _add_formula_dependencies(self, G: nx.DiGraph, model: 'Model'):
        """Add formula dependencies to dependency graph"""
        try:
            # Add auxiliary dependencies
            if hasattr(model, 'auxiliaries'):
                for aux_name, aux_data in model.auxiliaries.items():
                    formula = aux_data.get('formula', '')
                    if formula and hasattr(model, 'formula_engine'):
                        deps = model.formula_engine.get_formula_dependencies(formula)
                        for dep in deps:
                            if f"stock_{dep}" in G.nodes:
                                G.add_edge(f"stock_{dep}", f"aux_{aux_name}", relationship="formula_dep")
                            elif f"aux_{dep}" in G.nodes:
                                G.add_edge(f"aux_{dep}", f"aux_{aux_name}", relationship="formula_dep")
            
            # Add flow formula dependencies
            for flow in model.flows:
                if hasattr(flow, '_rate_formula') and hasattr(model, 'formula_engine'):
                    deps = model.formula_engine.get_formula_dependencies(flow._rate_formula)
                    for dep in deps:
                        if f"stock_{dep}" in G.nodes:
                            G.add_edge(f"stock_{dep}", f"flow_{flow.name}", relationship="formula_dep")
                        elif f"aux_{dep}" in G.nodes:
                            G.add_edge(f"aux_{dep}", f"flow_{flow.name}", relationship="formula_dep")
        except Exception:
            pass  # Skip formula dependencies if they can't be analyzed
    
    def _check_dimensional_consistency(self, model: 'Model', level: ValidationLevel) -> List[ValidationIssue]:
        """Check dimensional consistency using unit analysis"""
        issues = []
        
        try:
            # Check basic unit consistency
            for stock in model.stocks:
                stock_units = getattr(stock, 'units', '')
                
                # Check inflows
                for flow in stock.inflows.values():
                    flow_units = getattr(flow, 'units', '')
                    if stock_units and flow_units:
                        if not self._units_compatible(stock_units, flow_units, "inflow"):
                            issues.append(ValidationIssue(
                                severity=IssueSeverity.WARNING,
                                category="dimensional_consistency",
                                message=f"Unit mismatch: stock '{stock.name}' ({stock_units}) with inflow '{flow.name}' ({flow_units})",
                                suggestion="Ensure flow units are stock_units/time",
                                element_name=flow.name,
                                element_type="flow"
                            ))
                
                # Check outflows
                for flow in stock.outflows.values():
                    flow_units = getattr(flow, 'units', '')
                    if stock_units and flow_units:
                        if not self._units_compatible(stock_units, flow_units, "outflow"):
                            issues.append(ValidationIssue(
                                severity=IssueSeverity.WARNING,
                                category="dimensional_consistency",
                                message=f"Unit mismatch: stock '{stock.name}' ({stock_units}) with outflow '{flow.name}' ({flow_units})",
                                suggestion="Ensure flow units are stock_units/time",
                                element_name=flow.name,
                                element_type="flow"
                            ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                category="dimensional_consistency",
                message=f"Dimensional consistency check failed: {str(e)}",
                suggestion="Check unit definitions"
            ))
        
        return issues
    
    def _check_unreachable_stocks(self, model: 'Model', level: ValidationLevel) -> List[ValidationIssue]:
        """Check for unreachable or disconnected model elements"""
        issues = []
        
        try:
            # Build connectivity graph
            G = nx.Graph()  # Undirected for connectivity analysis
            
            for stock in model.stocks:
                G.add_node(stock.name, type="stock")
            
            for flow in model.flows:
                G.add_node(flow.name, type="flow")
                
                # Add connections
                for stock in model.stocks:
                    if flow in stock.inflows.values():
                        G.add_edge(flow.name, stock.name)
                    if flow in stock.outflows.values():
                        G.add_edge(stock.name, flow.name)
            
            # Find connected components
            components = list(nx.connected_components(G))
            
            if len(components) > 1:
                # Multiple disconnected components
                for i, component in enumerate(components):
                    if len(component) == 1:
                        element_name = list(component)[0]
                        issues.append(ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            category="connectivity",
                            message=f"Isolated element: '{element_name}'",
                            suggestion="Connect to other elements or remove if unused",
                            element_name=element_name
                        ))
                    elif i > 0:  # Not the largest component
                        issues.append(ValidationIssue(
                            severity=IssueSeverity.INFO,
                            category="connectivity",
                            message=f"Disconnected subsystem with {len(component)} elements: {list(component)}",
                            suggestion="Verify if this separation is intentional",
                            details={'component': list(component)}
                        ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                category="connectivity",
                message=f"Connectivity check failed: {str(e)}",
                suggestion="Check model structure"
            ))
        
        return issues
    
    def _check_unbounded_growth(self, model: 'Model', level: ValidationLevel) -> List[ValidationIssue]:
        """Check for potential unbounded growth patterns"""
        issues = []
        
        try:
            # Look for stocks with only positive inflows and no negative feedback
            for stock in model.stocks:
                has_limiting_factor = False
                has_outflows = len(stock.outflows) > 0
                
                # Check for limiting factors in formulas (simplified heuristic)
                if hasattr(model, 'auxiliaries'):
                    for aux_name, aux_data in model.auxiliaries.items():
                        formula = aux_data.get('formula', '')
                        if stock.name in formula and ('/' in formula or 'capacity' in formula.lower()):
                            has_limiting_factor = True
                            break
                
                # Check flows for limiting patterns
                for flow in stock.inflows.values():
                    if hasattr(flow, '_rate_formula'):
                        formula = flow._rate_formula
                        if ('/' in formula and stock.name in formula) or 'capacity' in formula.lower():
                            has_limiting_factor = True
                            break
                
                if not has_limiting_factor and not has_outflows:
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        category="unbounded_growth",
                        message=f"Stock '{stock.name}' may have unbounded growth (no limiting factors or outflows)",
                        suggestion="Add carrying capacity, outflows, or negative feedback",
                        element_name=stock.name,
                        element_type="stock"
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                category="unbounded_growth",
                message=f"Unbounded growth check failed: {str(e)}",
                suggestion="Check growth patterns manually"
            ))
        
        return issues
    
    def _check_formula_dependencies(self, model: 'Model', level: ValidationLevel) -> List[ValidationIssue]:
        """Check formula dependencies and validation"""
        issues = []
        
        if not hasattr(model, 'formula_engine'):
            return issues
        
        try:
            # Validate all formulas if enhanced model
            if hasattr(model, 'validate_all_formulas'):
                validation = model.validate_all_formulas()
                
                if not validation.get('valid', True):
                    for error in validation.get('errors', []):
                        issues.append(ValidationIssue(
                            severity=IssueSeverity.ERROR,
                            category="formula_error",
                            message=f"Formula error: {error}",
                            suggestion="Fix formula syntax and dependencies"
                        ))
                
                for warning in validation.get('warnings', []):
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        category="formula_warning",
                        message=f"Formula warning: {warning}",
                        suggestion="Review formula for potential issues"
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                category="formula_dependencies",
                message=f"Formula dependency check failed: {str(e)}",
                suggestion="Check formula engine configuration"
            ))
        
        return issues
    
    def _check_parameter_usage(self, model: 'Model', level: ValidationLevel) -> List[ValidationIssue]:
        """Check parameter usage and definitions"""
        issues = []
        
        try:
            defined_params = set()
            used_params = set()
            
            # Get defined parameters
            if hasattr(model, 'parameters'):
                defined_params.update(model.parameters.keys())
            
            # Find used parameters in formulas
            if hasattr(model, 'formula_engine'):
                for flow in model.flows:
                    if hasattr(flow, '_rate_formula'):
                        deps = model.formula_engine.get_formula_dependencies(flow._rate_formula)
                        used_params.update(deps)
                
                if hasattr(model, 'auxiliaries'):
                    for aux_data in model.auxiliaries.values():
                        formula = aux_data.get('formula', '')
                        if formula:
                            deps = model.formula_engine.get_formula_dependencies(formula)
                            used_params.update(deps)
            
            # Check for unused parameters
            unused_params = defined_params - used_params
            for param in unused_params:
                issues.append(ValidationIssue(
                    severity=IssueSeverity.INFO,
                    category="parameter_usage",
                    message=f"Parameter '{param}' is defined but not used",
                    suggestion="Remove unused parameter or use in formulas",
                    element_name=param,
                    element_type="parameter"
                ))
            
            # Check for undefined parameters
            undefined_params = used_params - defined_params - {stock.name for stock in model.stocks}
            for param in undefined_params:
                if param not in ['TIME', 'DT', 'STARTTIME', 'STOPTIME']:  # Exclude system variables
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        category="parameter_usage",
                        message=f"Parameter '{param}' is used but not defined",
                        suggestion="Define parameter value or check formula",
                        element_name=param,
                        element_type="parameter"
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                category="parameter_usage",
                message=f"Parameter usage check failed: {str(e)}",
                suggestion="Check parameter definitions"
            ))
        
        return issues
    
    def _check_unit_consistency(self, model: 'Model', level: ValidationLevel) -> List[ValidationIssue]:
        """Check unit consistency across the model"""
        issues = []
        
        # This would implement a full unit analysis system
        # For now, basic checks are done in dimensional_consistency
        
        return issues
    
    def _units_compatible(self, stock_units: str, flow_units: str, flow_type: str) -> bool:
        """Check if units are compatible (simplified)"""
        if not stock_units or not flow_units:
            return True  # Can't check without units
        
        # Basic compatibility checks
        flow_units_lower = flow_units.lower()
        stock_units_lower = stock_units.lower()
        
        # Check for time-based flow units
        time_indicators = ['per_time', '/time', '/year', '/month', '/day', '/hour']
        has_time = any(indicator in flow_units_lower for indicator in time_indicators)
        
        if has_time:
            # Flow should be stock_units/time
            base_flow_unit = flow_units_lower.split('/')[0] if '/' in flow_units_lower else flow_units_lower
            return base_flow_unit in stock_units_lower or stock_units_lower in base_flow_unit
        
        return True  # Default to compatible for unknown patterns

# ===============================================================================
# Behavioral Analysis Framework  
# ===============================================================================

class BehaviorAnalyzer:
    """Analyzes model behavior and dynamics"""
    
    def __init__(self):
        self.analysis_methods = [
            self._find_equilibria,
            self._analyze_stability,
            self._detect_oscillations,
            self._analyze_growth_patterns,
            self._check_boundary_behavior
        ]
    
    def analyze_behavior(self, model: 'Model', level: ValidationLevel = ValidationLevel.STANDARD) -> Dict[str, Any]:
        """Run behavioral analysis"""
        
        print(f"🔬 Running behavioral analysis ({level.value} level)...")
        
        analysis_results = {
            'equilibria': [],
            'stability': {},
            'oscillations': {},
            'growth_patterns': {},
            'boundary_behavior': {},
            'analysis_success': True,
            'analysis_errors': []
        }
        
        for method in self.analysis_methods:
            try:
                if level == ValidationLevel.BASIC and method.__name__ in ['_analyze_stability', '_detect_oscillations']:
                    continue  # Skip complex analysis for basic level
                
                result = method(model, level)
                
                method_name = method.__name__.replace('_', '').replace('analyze', '').replace('find', '').replace('detect', '').replace('check', '')
                analysis_results[method_name] = result
                
            except Exception as e:
                analysis_results['analysis_errors'].append(f"{method.__name__}: {str(e)}")
                analysis_results['analysis_success'] = False
        
        return analysis_results
    
    def _find_equilibria(self, model: 'Model', level: ValidationLevel) -> Dict[str, Any]:
        """Find equilibrium points where all derivatives are zero"""
        
        try:
            # Only attempt if model has integration support
            if not hasattr(model, 'compute_derivatives'):
                return {'equilibria': [], 'method': 'not_available'}
            
            equilibria = []
            initial_state = model.get_flattened_state()
            
            def derivatives_zero(state_vector):
                try:
                    return model.compute_derivatives(model.time, state_vector)
                except Exception:
                    return np.full_like(state_vector, np.inf)  # Invalid state
            
            # Try multiple initial guesses
            initial_guesses = [
                initial_state,  # Current state
                np.zeros_like(initial_state),  # Zero state
                np.ones_like(initial_state) * np.mean(initial_state),  # Mean state
            ]
            
            # Add some perturbed states
            if level.value in ['comprehensive', 'deep']:
                for i in range(3):
                    noise = np.random.normal(0, np.std(initial_state) * 0.1, initial_state.shape)
                    initial_guesses.append(initial_state + noise)
            
            for i, guess in enumerate(initial_guesses):
                try:
                    equilibrium = fsolve(derivatives_zero, guess, xtol=1e-10)
                    
                    # Verify it's actually an equilibrium
                    derivatives = derivatives_zero(equilibrium)
                    if np.allclose(derivatives, 0, atol=1e-8):
                        # Check if we already found this equilibrium
                        is_new = True
                        for existing_eq in equilibria:
                            if np.allclose(equilibrium, existing_eq['state'], atol=1e-6):
                                is_new = False
                                break
                        
                        if is_new and np.all(equilibrium >= 0):  # Ensure physical validity
                            equilibria.append({
                                'state': equilibrium,
                                'total': np.sum(equilibrium),
                                'convergence_quality': np.max(np.abs(derivatives))
                            })
                
                except Exception:
                    continue
            
            return {
                'equilibria': equilibria,
                'count': len(equilibria),
                'method': 'fsolve',
                'search_attempts': len(initial_guesses)
            }
        
        except Exception as e:
            return {'equilibria': [], 'error': str(e), 'method': 'failed'}
    
    def _analyze_stability(self, model: 'Model', level: ValidationLevel) -> Dict[str, Any]:
        """Analyze stability of equilibrium points"""
        
        try:
            if not hasattr(model, 'compute_derivatives'):
                return {'stability': 'analysis_not_available'}
            
            # Get equilibria from previous analysis
            equilibria_result = self._find_equilibria(model, level)
            equilibria = equilibria_result.get('equilibria', [])
            
            stability_results = []
            
            for eq_data in equilibria:
                equilibrium = eq_data['state']
                
                try:
                    # Compute Jacobian matrix at equilibrium
                    jacobian = self._compute_jacobian(model, equilibrium)
                    
                    # Find eigenvalues
                    eigenvalues = eigvals(jacobian)
                    
                    # Analyze stability
                    max_real_part = np.max(np.real(eigenvalues))
                    is_stable = max_real_part < -1e-10  # Slightly negative for stability
                    
                    stability_results.append({
                        'equilibrium_state': equilibrium,
                        'eigenvalues': eigenvalues.tolist(),
                        'is_stable': is_stable,
                        'max_real_eigenvalue': max_real_part,
                        'has_oscillations': np.any(np.abs(np.imag(eigenvalues)) > 1e-10),
                        'stability_margin': -max_real_part if is_stable else max_real_part
                    })
                
                except Exception as e:
                    stability_results.append({
                        'equilibrium_state': equilibrium,
                        'analysis_failed': True,
                        'error': str(e)
                    })
            
            return {
                'stability_analysis': stability_results,
                'stable_equilibria': sum(1 for r in stability_results if r.get('is_stable', False)),
                'total_equilibria': len(stability_results)
            }
        
        except Exception as e:
            return {'stability': 'analysis_failed', 'error': str(e)}
    
    def _detect_oscillations(self, model: 'Model', level: ValidationLevel) -> Dict[str, Any]:
        """Detect oscillatory behavior in model"""
        
        try:
            # Run a short simulation to analyze behavior
            if not hasattr(model, 'run'):
                return {'oscillations': 'simulation_not_available'}
            
            # Save current state
            original_time = model.time
            original_stocks = {stock.name: stock.values.copy() for stock in model.stocks}
            
            # Run simulation
            model.reset()
            if hasattr(model, 'integration_engine'):
                result = model.integration_engine.integrate(model, method='rk4', end_time=50, dt=0.1)
            else:
                model.run(duration=50)
                result = {'success': True, 'y': np.array([[stock.total() for stock in model.stocks] for _ in model.time_history]).T}
            
            oscillation_info = {}
            
            if result.get('success', False) and 'y' in result:
                y = result['y']
                
                for i, stock in enumerate(model.stocks):
                    if i < y.shape[0]:
                        state_trajectory = y[i, :]
                        
                        # Simple oscillation detection using autocorrelation
                        if len(state_trajectory) > 10:
                            detrended = state_trajectory - np.mean(state_trajectory)
                            
                            if np.std(detrended) > 1e-6:  # Avoid division by zero
                                # Compute autocorrelation
                                autocorr = np.correlate(detrended, detrended, mode='full')
                                autocorr = autocorr[autocorr.size // 2:]
                                autocorr = autocorr / autocorr[0]  # Normalize
                                
                                # Look for peaks
                                peaks = []
                                for j in range(1, min(len(autocorr) - 1, 50)):
                                    if (autocorr[j] > autocorr[j-1] and 
                                        autocorr[j] > autocorr[j+1] and 
                                        autocorr[j] > 0.3):
                                        peaks.append(j)
                                
                                if peaks:
                                    period_estimate = peaks[0] * 0.1  # dt = 0.1
                                    oscillation_info[stock.name] = {
                                        'is_oscillating': True,
                                        'estimated_period': period_estimate,
                                        'amplitude': np.std(detrended),
                                        'first_peak_correlation': autocorr[peaks[0]]
                                    }
                                else:
                                    oscillation_info[stock.name] = {'is_oscillating': False}
                            else:
                                oscillation_info[stock.name] = {'is_oscillating': False, 'reason': 'constant_value'}
                        else:
                            oscillation_info[stock.name] = {'is_oscillating': False, 'reason': 'insufficient_data'}
            
            # Restore original state
            model.time = original_time
            for stock in model.stocks:
                if stock.name in original_stocks:
                    stock.values = original_stocks[stock.name]
            
            return oscillation_info
        
        except Exception as e:
            return {'oscillations': 'analysis_failed', 'error': str(e)}
    
    def _analyze_growth_patterns(self, model: 'Model', level: ValidationLevel) -> Dict[str, Any]:
        """Analyze growth patterns and trends"""
        
        try:
            growth_analysis = {}
            
            # Analyze each stock's potential growth
            for stock in model.stocks:
                stock_analysis = {
                    'has_inflows': len(stock.inflows) > 0,
                    'has_outflows': len(stock.outflows) > 0,
                    'flow_balance': len(stock.inflows) - len(stock.outflows),
                    'growth_type': 'unknown'
                }
                
                # Classify growth type based on flows
                if len(stock.inflows) > 0 and len(stock.outflows) == 0:
                    stock_analysis['growth_type'] = 'accumulator'
                elif len(stock.inflows) == 0 and len(stock.outflows) > 0:
                    stock_analysis['growth_type'] = 'depleting'
                elif len(stock.inflows) > 0 and len(stock.outflows) > 0:
                    stock_analysis['growth_type'] = 'balanced'
                else:
                    stock_analysis['growth_type'] = 'isolated'
                
                # Check for growth limiting factors
                limiting_factors = []
                if hasattr(model, 'auxiliaries'):
                    for aux_name, aux_data in model.auxiliaries.items():
                        formula = aux_data.get('formula', '')
                        if stock.name in formula and ('capacity' in formula.lower() or '/' in formula):
                            limiting_factors.append(aux_name)
                
                stock_analysis['limiting_factors'] = limiting_factors
                stock_analysis['has_limits'] = len(limiting_factors) > 0
                
                growth_analysis[stock.name] = stock_analysis
            
            return growth_analysis
        
        except Exception as e:
            return {'growth_patterns': 'analysis_failed', 'error': str(e)}
    
    def _check_boundary_behavior(self, model: 'Model', level: ValidationLevel) -> Dict[str, Any]:
        """Check behavior at boundary conditions"""
        
        try:
            boundary_results = {}
            
            # Check behavior when stocks approach zero
            for stock in model.stocks:
                # Check if outflows can drive stock negative
                can_go_negative = True
                
                # Check bounds
                if hasattr(stock, 'min_value'):
                    can_go_negative = stock.min_value < 0
                
                boundary_results[stock.name] = {
                    'can_go_negative': can_go_negative,
                    'has_bounds': hasattr(stock, 'min_value') and hasattr(stock, 'max_value'),
                    'min_bound': getattr(stock, 'min_value', None),
                    'max_bound': getattr(stock, 'max_value', None)
                }
            
            return boundary_results
        
        except Exception as e:
            return {'boundary_behavior': 'analysis_failed', 'error': str(e)}
    
    def _compute_jacobian(self, model: 'Model', state: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Compute Jacobian matrix using finite differences"""
        
        n = len(state)
        jacobian = np.zeros((n, n))
        
        f0 = model.compute_derivatives(model.time, state)
        
        for i in range(n):
            state_plus = state.copy()
            state_plus[i] += epsilon
            f_plus = model.compute_derivatives(model.time, state_plus)
            jacobian[:, i] = (f_plus - f0) / epsilon
        
        return jacobian

# ===============================================================================
# Main Model Validator
# ===============================================================================

class ModelValidator:
    """Main model validation orchestrator"""
    
    def __init__(self):
        self.structural_validator = StructuralValidator()
        self.behavior_analyzer = BehaviorAnalyzer()
    
    def validate_model(self, model: 'Model', 
                      level: ValidationLevel = ValidationLevel.STANDARD,
                      include_behavioral: bool = True,
                      include_performance: bool = False) -> ValidationReport:
        """Comprehensive model validation"""
        
        print(f"🔍 Starting comprehensive model validation ({level.value} level)")
        print(f"   Model: {getattr(model, 'name', 'Unnamed')}")
        print(f"   Stocks: {len(model.stocks)}, Flows: {len(model.flows)}")
        
        # Create validation report
        report = ValidationReport(
            model_name=getattr(model, 'name', 'Unnamed Model'),
            validation_level=level
        )
        
        # 1. Structural Validation
        try:
            structural_issues = self.structural_validator.validate_structure(model, level)
            for issue in structural_issues:
                report.add_issue(issue)
            
            report.structural_analysis = {
                'completed': True,
                'issues_found': len(structural_issues),
                'categories': list(set(issue.category for issue in structural_issues))
            }
            
        except Exception as e:
            report.add_issue(ValidationIssue(
                severity=IssueSeverity.ERROR,
                category="validation_framework",
                message=f"Structural validation failed: {str(e)}",
                suggestion="Check model configuration"
            ))
        
        # 2. Behavioral Analysis (if requested)
        if include_behavioral and level.value in ['standard', 'comprehensive', 'deep']:
            try:
                behavioral_results = self.behavior_analyzer.analyze_behavior(model, level)
                report.behavioral_analysis = behavioral_results
                
                # Add issues based on behavioral analysis
                self._interpret_behavioral_results(behavioral_results, report)
                
            except Exception as e:
                report.add_issue(ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    category="behavioral_analysis",
                    message=f"Behavioral analysis failed: {str(e)}",
                    suggestion="Check if model supports dynamic analysis"
                ))
        
        # 3. Performance Analysis (if requested)
        if include_performance and level.value in ['comprehensive', 'deep']:
            try:
                performance_results = self._analyze_performance(model)
                report.performance_analysis = performance_results
                
            except Exception as e:
                report.add_issue(ValidationIssue(
                    severity=IssueSeverity.INFO,
                    category="performance_analysis",
                    message=f"Performance analysis failed: {str(e)}",
                    suggestion="Performance analysis is optional"
                ))
        
        # Final validation summary
        self._print_validation_summary(report)
        
        return report
    
    def _interpret_behavioral_results(self, behavioral_results: Dict[str, Any], report: ValidationReport):
        """Interpret behavioral analysis results and add issues"""
        
        # Check equilibria
        equilibria_data = behavioral_results.get('equilibria', {})
        if isinstance(equilibria_data, dict):
            equilibria_count = equilibria_data.get('count', 0)
            
            if equilibria_count == 0:
                report.add_issue(ValidationIssue(
                    severity=IssueSeverity.INFO,
                    category="behavioral_analysis",
                    message="No equilibrium points found",
                    suggestion="Model may have unbounded dynamics or complex behavior"
                ))
            elif equilibria_count > 3:
                report.add_issue(ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    category="behavioral_analysis",
                    message=f"Multiple equilibria found ({equilibria_count})",
                    suggestion="Model may have complex multi-stable behavior"
                ))
        
        # Check stability
        stability_data = behavioral_results.get('stability', {})
        if isinstance(stability_data, dict):
            stable_count = stability_data.get('stable_equilibria', 0)
            total_count = stability_data.get('total_equilibria', 0)
            
            if total_count > 0 and stable_count == 0:
                report.add_issue(ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    category="behavioral_analysis",
                    message="No stable equilibria found",
                    suggestion="Model may be inherently unstable"
                ))
        
        # Check oscillations
        oscillations_data = behavioral_results.get('oscillations', {})
        oscillating_stocks = [name for name, data in oscillations_data.items() 
                            if isinstance(data, dict) and data.get('is_oscillating', False)]
        
        if len(oscillating_stocks) > 0:
            report.add_issue(ValidationIssue(
                severity=IssueSeverity.INFO,
                category="behavioral_analysis",
                message=f"Oscillatory behavior detected in: {', '.join(oscillating_stocks)}",
                suggestion="Verify if oscillations are expected or indicate instability"
            ))
    
    def _analyze_performance(self, model: 'Model') -> Dict[str, Any]:
        """Analyze model performance characteristics"""
        
        performance_analysis = {
            'model_size': {
                'stocks': len(model.stocks),
                'flows': len(model.flows),
                'total_state_variables': sum(getattr(stock.values, 'size', 1) for stock in model.stocks)
            },
            'complexity_score': self._calculate_complexity_score(model),
            'estimated_performance': 'unknown'
        }
        
        # Estimate performance category
        total_vars = performance_analysis['model_size']['total_state_variables']
        if total_vars < 10:
            performance_analysis['estimated_performance'] = 'fast'
        elif total_vars < 100:
            performance_analysis['estimated_performance'] = 'medium'
        elif total_vars < 1000:
            performance_analysis['estimated_performance'] = 'slow'
        else:
            performance_analysis['estimated_performance'] = 'very_slow'
        
        return performance_analysis
    
    def _calculate_complexity_score(self, model: 'Model') -> int:
        """Calculate model complexity score"""
        
        score = 0
        
        # Base complexity from elements
        score += len(model.stocks) * 2
        score += len(model.flows) * 3
        
        # Multi-dimensional complexity
        for stock in model.stocks:
            if hasattr(stock, 'dimensions') and stock.dimensions:
                score += len(stock.dimensions) * 5
        
        # Formula complexity
        if hasattr(model, 'auxiliaries'):
            score += len(model.auxiliaries) * 4
        
        # Enhanced flow complexity
        for stock in model.stocks:
            if hasattr(stock, 'md_flows'):
                score += len(stock.md_flows) * 6
        
        return score
    
    def _print_validation_summary(self, report: ValidationReport):
        """Print validation summary"""
        
        print(f"\n{'='*60}")
        print("📊 VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        status = "✅ VALID" if report.is_valid else "❌ INVALID"
        print(f"Model Status: {status}")
        print(f"Total Issues: {report.total_issues}")
        
        if report.error_count > 0:
            print(f"  🔴 Errors: {report.error_count}")
        if report.warning_count > 0:
            print(f"  🟡 Warnings: {report.warning_count}")
        if report.info_count > 0:
            print(f"  🔵 Info: {report.info_count}")
        
        # Print critical issues
        critical_issues = report.get_issues_by_severity(IssueSeverity.CRITICAL)
        error_issues = report.get_issues_by_severity(IssueSeverity.ERROR)
        
        if critical_issues or error_issues:
            print(f"\n🚨 Critical Issues:")
            for issue in critical_issues + error_issues:
                print(f"   • {issue.message}")
                if issue.suggestion:
                    print(f"     💡 {issue.suggestion}")
        
        # Behavioral analysis summary
        if report.behavioral_analysis:
            print(f"\n🔬 Behavioral Analysis:")
            equilibria = report.behavioral_analysis.get('equilibria', {})
            if isinstance(equilibria, dict):
                eq_count = equilibria.get('count', 0)
                print(f"   Equilibria found: {eq_count}")
            
            stability = report.behavioral_analysis.get('stability', {})
            if isinstance(stability, dict):
                stable = stability.get('stable_equilibria', 0)
                total = stability.get('total_equilibria', 0)
                if total > 0:
                    print(f"   Stable equilibria: {stable}/{total}")
        
        print(f"\n⏱️  Validation completed in {time.time() - report.timestamp:.2f}s")

# ===============================================================================
# Convenience Functions
# ===============================================================================

def validate_model(model: 'Model', 
                  level: str = "standard",
                  include_behavioral: bool = True) -> ValidationReport:
    """Convenience function for model validation"""
    
    validator = ModelValidator()
    validation_level = ValidationLevel(level.lower())
    
    return validator.validate_model(
        model, 
        level=validation_level,
        include_behavioral=include_behavioral
    )

def quick_validate(model: 'Model') -> bool:
    """Quick validation - returns True if model is valid"""
    
    report = validate_model(model, level="basic", include_behavioral=False)
    return report.is_valid

def export_validation_report(report: ValidationReport, filepath: str):
    """Export validation report to JSON"""
    
    report_dict = {
        'model_name': report.model_name,
        'validation_level': report.validation_level.value,
        'timestamp': report.timestamp,
        'is_valid': report.is_valid,
        'summary': {
            'total_issues': report.total_issues,
            'error_count': report.error_count,
            'warning_count': report.warning_count,
            'info_count': report.info_count
        },
        'issues': [
            {
                'severity': issue.severity.value,
                'category': issue.category,
                'message': issue.message,
                'suggestion': issue.suggestion,
                'element_name': issue.element_name,
                'element_type': issue.element_type,
                'details': issue.details
            } for issue in report.issues
        ],
        'structural_analysis': report.structural_analysis,
        'behavioral_analysis': report.behavioral_analysis,
        'performance_analysis': report.performance_analysis
    }
    
    with open(filepath, 'w') as f:
        json.dump(report_dict, f, indent=2, default=str)
    
    print(f"📄 Validation report exported to: {filepath}")

# ===============================================================================
# Testing Functions
# ===============================================================================

def test_model_validation():
    """Test model validation framework"""
    
    print("🧪 Testing Model Validation Framework")
    print("=" * 50)
    
    try:
        from simulation import Model, Stock, Flow
        
        # Create test model
        model = Model(name="Validation Test Model")
        
        # Add stocks
        population = Stock(values=1000.0, name="Population", units="people")
        resources = Stock(values=500.0, name="Resources", units="units")
        
        # Add flows
        growth = Flow(rate_expression=lambda: population.values * 0.02, name="Growth", units="people/year")
        consumption = Flow(rate_expression=lambda: population.values * 0.1, name="Consumption", units="units/year")
        
        # Connect flows
        population.add_inflow(growth)
        resources.add_outflow(consumption)
        
        model.add_stock(population)
        model.add_stock(resources)
        model.add_flow(growth)
        model.add_flow(consumption)
        
        # Test validation
        validator = ModelValidator()
        report = validator.validate_model(model, ValidationLevel.STANDARD)
        
        print(f"✅ Validation completed!")
        print(f"   Model valid: {report.is_valid}")
        print(f"   Issues found: {report.total_issues}")
        print(f"   Categories tested: {len(set(issue.category for issue in report.issues))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

if __name__ == "__main__":
    test_model_validation()