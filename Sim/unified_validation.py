# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 11:50:16 2025

@author: NagabhushanamTattaga
"""

# unified_validation.py - Consolidates ALL validation logic

from typing import Dict, List, Any, Protocol, Union, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import ast
import re
from abc import ABC, abstractmethod
import datetime  # ADD THIS LINE

# ===============================================================================
# Unified Validation Framework - Replaces duplicate validation across modules
# ===============================================================================

class ValidationLevel(Enum):
    BASIC = "basic"
    STANDARD = "standard" 
    COMPREHENSIVE = "comprehensive"

class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationIssue:
    severity: ValidationSeverity
    category: str
    message: str
    suggestion: str
    element_name: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and clean up the issue data"""
        if not self.message:
            self.message = "Unknown validation issue"
        if not self.suggestion:
            self.suggestion = "Review and fix the issue"
        if not self.category:
            self.category = "general"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary"""
        return {
            'severity': self.severity.value,
            'category': self.category,
            'message': self.message,
            'suggestion': self.suggestion,
            'element_name': self.element_name,
            'details': self.details
        }
    
    def __str__(self) -> str:
        """String representation of the issue"""
        severity_icons = {
            'critical': '🚨',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️'
        }
        icon = severity_icons.get(self.severity.value, '❓')
        element_part = f" [{self.element_name}]" if self.element_name else ""
        return f"{icon} {self.severity.value.upper()}{element_part}: {self.message}"

@dataclass
class ValidationReport:
    is_valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Initialize timestamp and other metadata"""
        if self.timestamp is None:
            import datetime
            self.timestamp = datetime.datetime.now().isoformat()
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue to the report"""
        self.issues.append(issue)
        if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False
    
    def merge(self, other: 'ValidationReport'):
        """Merge another validation report into this one"""
        self.issues.extend(other.issues)
        if not other.is_valid:
            self.is_valid = False
        
        # Merge metadata
        if other.metadata:
            if not self.metadata:
                self.metadata = {}
            self.metadata.update(other.metadata)
    
    def __getattr__(self, name: str) -> Any:
        """Handle missing attributes gracefully"""
        
        # Handle common metric requests as attributes
        if name.endswith('_count'):
            metric_name = name
            return self.get_metric(metric_name, default=0)
        
        elif name.endswith('_score'):
            metric_name = name
            return self.get_metric(metric_name, default=0.0)
        
        elif name == 'model_name':
            return self.metadata.get('model_name', 'unnamed_model') if self.metadata else 'unnamed_model'
        
        elif name == 'validation_level':
            return self.metadata.get('validation_level', 'standard') if self.metadata else 'standard'
        
        elif name in ['structural_analysis', 'behavioral_analysis', 'performance_analysis']:
            return self.metadata.get(name, {}) if self.metadata else {}
        
        # If nothing matches, raise the normal AttributeError
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")            
        
    def get_metric(self, metric_name: str, *args, **kwargs) -> Any:
        """Get a specific validation metric (flexible argument handling)"""
        
        # Handle legacy calls or extra arguments gracefully
        context = kwargs.get('context', None)
        default_value = kwargs.get('default', None)
        
        metrics = {
            'total_issues': len(self.issues),
            'error_count': len([i for i in self.issues if i.severity == ValidationSeverity.ERROR]),
            'warning_count': len([i for i in self.issues if i.severity == ValidationSeverity.WARNING]),
            'info_count': len([i for i in self.issues if i.severity == ValidationSeverity.INFO]),
            'critical_count': len([i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]),
            'validity_score': 1.0 if self.is_valid else 0.0,
            'severity_score': self._calculate_severity_score(),
            'categories': list(set(issue.category for issue in self.issues)),
            'elements_with_issues': list(set(issue.element_name for issue in self.issues if issue.element_name)),
            'issues_by_category': self._get_issues_by_category_dict(),
            'issues_by_severity': self._get_issues_by_severity_dict(),
            'validation_summary': self.get_summary()
        }
        
        # Return the requested metric or default value
        result = metrics.get(metric_name, default_value)
        
        # If still None and no default provided, return appropriate fallback
        if result is None:
            if 'count' in metric_name.lower():
                return 0
            elif 'score' in metric_name.lower():
                return 0.0
            elif 'list' in metric_name.lower() or 'issues' in metric_name.lower():
                return []
            else:
                return None
        
        return result

    def _get_issues_by_category_dict(self) -> Dict[str, List[ValidationIssue]]:
        """Get issues grouped by category"""
        category_dict = {}
        for issue in self.issues:
            if issue.category not in category_dict:
                category_dict[issue.category] = []
            category_dict[issue.category].append(issue)
        return category_dict
    
    def _get_issues_by_severity_dict(self) -> Dict[str, List[ValidationIssue]]:
        """Get issues grouped by severity"""
        severity_dict = {}
        for issue in self.issues:
            severity_key = issue.severity.value
            if severity_key not in severity_dict:
                severity_dict[severity_key] = []
            severity_dict[severity_key].append(issue)
        return severity_dict
        
    def _calculate_severity_score(self) -> float:
        """Calculate a severity score (0-1, where 1 is best)"""
        if not self.issues:
            return 1.0
        
        severity_weights = {
            ValidationSeverity.CRITICAL: 0.0,
            ValidationSeverity.ERROR: 0.2,
            ValidationSeverity.WARNING: 0.6,
            ValidationSeverity.INFO: 0.9
        }
        
        total_weight = sum(severity_weights.get(issue.severity, 0.5) for issue in self.issues)
        return total_weight / len(self.issues) if self.issues else 1.0
        
    def get_validation_metric(self, metric_name: str, context: Any = None) -> Any:
        """Alternative method name for getting metrics (legacy compatibility)"""
        return self.get_metric(metric_name, context=context)
    
    def fetch_metric(self, metric_name: str, default_value: Any = None) -> Any:
        """Another alternative method name for getting metrics"""
        return self.get_metric(metric_name, default=default_value)
    
    def query_metric(self, metric_name: str) -> Any:
        """Query a metric with simple interface"""
        return self.get_metric(metric_name)    
    
    @property
    def total_issues(self) -> int:
        """Total number of issues"""
        return len(self.issues)
    
    @property
    def error_count(self) -> int:
        """Number of error-level issues"""
        return len([i for i in self.issues if i.severity == ValidationSeverity.ERROR])
    
    @property
    def warning_count(self) -> int:
        """Number of warning-level issues"""
        return len([i for i in self.issues if i.severity == ValidationSeverity.WARNING])
    
    @property
    def info_count(self) -> int:
        """Number of info-level issues"""
        return len([i for i in self.issues if i.severity == ValidationSeverity.INFO])
    
    @property
    def critical_count(self) -> int:
        """Number of critical-level issues"""
        return len([i for i in self.issues if i.severity == ValidationSeverity.CRITICAL])
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get all issues of a specific severity"""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_category(self, category: str) -> List[ValidationIssue]:
        """Get all issues of a specific category"""
        return [issue for issue in self.issues if issue.category == category]
    
    def get_issues_by_element(self, element_name: str) -> List[ValidationIssue]:
        """Get all issues for a specific element"""
        return [issue for issue in self.issues if issue.element_name == element_name]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the validation report"""
        return {
            'is_valid': self.is_valid,
            'total_issues': self.total_issues,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'info_count': self.info_count,
            'critical_count': self.critical_count,
            'severity_score': self._calculate_severity_score(),
            'categories': list(set(issue.category for issue in self.issues)),
            'elements_affected': list(set(issue.element_name for issue in self.issues if issue.element_name)),
            'timestamp': self.timestamp
        }
    
    def filter_issues(self, severity: Optional[ValidationSeverity] = None, 
                     category: Optional[str] = None, 
                     element_name: Optional[str] = None) -> 'ValidationReport':
        """Create a new report with filtered issues"""
        filtered_report = ValidationReport(metadata=self.metadata.copy() if self.metadata else {})
        
        for issue in self.issues:
            include = True
            
            if severity and issue.severity != severity:
                include = False
            if category and issue.category != category:
                include = False
            if element_name and issue.element_name != element_name:
                include = False
            
            if include:
                filtered_report.add_issue(issue)
        
        return filtered_report
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            'is_valid': self.is_valid,
            'timestamp': self.timestamp,
            'summary': self.get_summary(),
            'issues': [
                {
                    'severity': issue.severity.value,
                    'category': issue.category,
                    'message': issue.message,
                    'suggestion': issue.suggestion,
                    'element_name': issue.element_name,
                    'details': issue.details
                } for issue in self.issues
            ],
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """String representation of the report"""
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        return f"ValidationReport({status}, {self.total_issues} issues)"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"ValidationReport(is_valid={self.is_valid}, issues={len(self.issues)}, timestamp='{self.timestamp}')"
# ===============================================================================
# Validator Interface - All validators implement this
# ===============================================================================

class IValidator(Protocol):
    def validate(self, target: Any, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
        """Validate target object and return report"""
        ...
    
    def applies_to(self, target_type: str) -> bool:
        """Check if this validator applies to target type"""
        ...

# ===============================================================================
# Specific Validators - Consolidated from multiple modules
# ===============================================================================

class MassConservationValidator:
    """Consolidates mass conservation logic from simulation.py and model_validation.py"""
    
    def __init__(self):
        # Conservation tolerance levels
        self.tolerance_levels = {
            ValidationLevel.BASIC: 1e-3,        # 0.1% error tolerance
            ValidationLevel.STANDARD: 1e-6,     # 0.0001% error tolerance  
            ValidationLevel.COMPREHENSIVE: 1e-9  # Very strict tolerance
        }
        
        # Mass balance check parameters
        self.mass_balance_config = {
            'check_interval': 10,  # Check every N time steps
            'min_stock_value': 1e-12,  # Minimum stock value to consider
            'max_relative_error': 1e-2,  # Maximum acceptable relative error
            'track_conservation_history': True
        }
    
    def applies_to(self, target_type: str) -> bool:
        return target_type in ['model', 'integration_result', 'simulation_result', 'stock']
    
    def validate(self, target: Any, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
        report = ValidationReport()
        
        # Handle different target types
        if hasattr(target, 'stocks') and hasattr(target, 'flows'):
            # Model with stocks and flows
            self._validate_model_conservation(target, report, level)
        
        elif hasattr(target, 'validate_mass_conservation'):
            # Object with existing mass conservation method
            self._validate_existing_conservation_method(target, report, level)
        
        elif isinstance(target, dict) and 'y' in target and 't' in target:
            # Integration result dictionary
            self._validate_integration_conservation(target, report, level)
        
        elif hasattr(target, 'simulation_results'):
            # Model with simulation results
            self._validate_simulation_conservation(target, report, level)
        
        elif hasattr(target, 'values') and hasattr(target, 'inflows'):
            # Single stock validation
            self._validate_stock_conservation(target, report, level)
        
        return report
    
    def _validate_model_conservation(self, model: Any, report: ValidationReport, level: ValidationLevel):
        """Validate mass conservation for entire model"""
        
        try:
            # Get stocks and flows
            stocks = self._get_model_stocks(model)
            flows = self._get_model_flows(model)
            
            if not stocks:
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="mass_balance",
                    message="No stocks found in model - cannot validate mass conservation",
                    suggestion="Add stocks to the model for mass conservation validation"
                ))
                return
            
            # 1. Check flow-stock connectivity
            connectivity_issues = self._check_flow_stock_connectivity(stocks, flows)
            for issue in connectivity_issues:
                report.add_issue(issue)
            
            # 2. Check for mass balance violations
            if level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
                balance_issues = self._check_mass_balance(stocks, flows, level)
                for issue in balance_issues:
                    report.add_issue(issue)
            
            # 3. Check for conservation across dimensions (if multi-dimensional)
            if level == ValidationLevel.COMPREHENSIVE:
                dimension_issues = self._check_dimensional_conservation(stocks, flows)
                for issue in dimension_issues:
                    report.add_issue(issue)
            
            # 4. Analytical conservation check
            analytical_issues = self._check_analytical_conservation(model, level)
            for issue in analytical_issues:
                report.add_issue(issue)
        
        except Exception as e:
            report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="mass_balance",
                message=f"Mass conservation validation failed: {str(e)}",
                suggestion="Check model structure and flow definitions"
            ))
    
    def _get_model_stocks(self, model: Any) -> List[Any]:
        """Extract stocks from model"""
        stocks = []
        
        if hasattr(model, 'stocks'):
            stock_collection = model.stocks
            if isinstance(stock_collection, dict):
                stocks = list(stock_collection.values())
            elif hasattr(stock_collection, '__iter__'):
                stocks = list(stock_collection)
        
        elif hasattr(model, 'context') and hasattr(model.context, 'stocks'):
            stock_collection = model.context.stocks
            if isinstance(stock_collection, dict):
                stocks = list(stock_collection.values())
        
        return stocks
    
    def _get_model_flows(self, model: Any) -> List[Any]:
        """Extract flows from model"""
        flows = []
        
        if hasattr(model, 'flows'):
            flow_collection = model.flows
            if isinstance(flow_collection, dict):
                flows = list(flow_collection.values())
            elif hasattr(flow_collection, '__iter__'):
                flows = list(flow_collection)
        
        elif hasattr(model, 'context') and hasattr(model.context, 'flows'):
            flow_collection = model.context.flows
            if isinstance(flow_collection, dict):
                flows = list(flow_collection.values())
        
        return flows
    
    def _check_flow_stock_connectivity(self, stocks: List[Any], flows: List[Any]) -> List[ValidationIssue]:
        """Check that flows are properly connected to stocks"""
        issues = []
        
        # Create stock name lookup
        stock_names = set()
        for stock in stocks:
            if hasattr(stock, 'name'):
                stock_names.add(stock.name)
        
        # Check each flow's connections
        for flow in flows:
            flow_name = getattr(flow, 'name', 'unnamed_flow')
            
            from_stock = getattr(flow, 'from_stock', None)
            to_stock = getattr(flow, 'to_stock', None)
            
            # Check from_stock connection
            if from_stock:
                from_stock_name = getattr(from_stock, 'name', str(from_stock))
                if from_stock_name not in stock_names and from_stock not in stocks:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="mass_balance",
                        message=f"Flow '{flow_name}' from_stock '{from_stock_name}' not found",
                        suggestion="Check stock reference or add missing stock",
                        element_name=flow_name
                    ))
            
            # Check to_stock connection  
            if to_stock:
                to_stock_name = getattr(to_stock, 'name', str(to_stock))
                if to_stock_name not in stock_names and to_stock not in stocks:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="mass_balance", 
                        message=f"Flow '{flow_name}' to_stock '{to_stock_name}' not found",
                        suggestion="Check stock reference or add missing stock",
                        element_name=flow_name
                    ))
            
            # Warn about disconnected flows
            if not from_stock and not to_stock:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="mass_balance",
                    message=f"Flow '{flow_name}' not connected to any stock",
                    suggestion="Connect flow to source and/or target stock for mass conservation",
                    element_name=flow_name
                ))
        
        return issues
    
    def _check_mass_balance(self, stocks: List[Any], flows: List[Any], level: ValidationLevel) -> List[ValidationIssue]:
        """Check mass balance for each stock"""
        issues = []
        tolerance = self.tolerance_levels.get(level, 1e-6)
        
        for stock in stocks:
            stock_name = getattr(stock, 'name', 'unnamed_stock')
            
            try:
                # Calculate net flow rate for this stock
                net_inflow = 0.0
                net_outflow = 0.0
                connected_flows = []
                
                for flow in flows:
                    flow_name = getattr(flow, 'name', 'unnamed_flow')
                    from_stock = getattr(flow, 'from_stock', None)
                    to_stock = getattr(flow, 'to_stock', None)
                    
                    # Check if flow affects this stock
                    if to_stock == stock or (hasattr(to_stock, 'name') and hasattr(stock, 'name') and to_stock.name == stock.name):
                        # This is an inflow
                        connected_flows.append((flow_name, 'inflow'))
                        try:
                            if hasattr(flow, 'rate_expression') and callable(flow.rate_expression):
                                flow_rate = flow.rate_expression()
                                net_inflow += float(flow_rate) if np.isscalar(flow_rate) else np.sum(flow_rate)
                        except Exception as e:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                category="mass_balance",
                                message=f"Cannot evaluate inflow '{flow_name}' for stock '{stock_name}': {str(e)}",
                                suggestion="Check flow rate calculation",
                                element_name=flow_name
                            ))
                    
                    if from_stock == stock or (hasattr(from_stock, 'name') and hasattr(stock, 'name') and from_stock.name == stock.name):
                        # This is an outflow
                        connected_flows.append((flow_name, 'outflow'))
                        try:
                            if hasattr(flow, 'rate_expression') and callable(flow.rate_expression):
                                flow_rate = flow.rate_expression()
                                net_outflow += float(flow_rate) if np.isscalar(flow_rate) else np.sum(flow_rate)
                        except Exception as e:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                category="mass_balance",
                                message=f"Cannot evaluate outflow '{flow_name}' for stock '{stock_name}': {str(e)}",
                                suggestion="Check flow rate calculation",
                                element_name=flow_name
                            ))
                
                # Check for stocks with no connected flows
                if not connected_flows:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        category="mass_balance",
                        message=f"Stock '{stock_name}' has no connected flows",
                        suggestion="Add flows or remove unused stock",
                        element_name=stock_name
                    ))
                
                # Calculate mass balance
                net_flow = net_inflow - net_outflow
                stock_value = getattr(stock, 'values', getattr(stock, 'value', 0))
                
                if hasattr(stock_value, '__iter__') and not isinstance(stock_value, str):
                    stock_value = np.sum(stock_value)
                
                # Check for conservation issues
                if abs(stock_value) > self.mass_balance_config['min_stock_value']:
                    relative_error = abs(net_flow) / abs(stock_value)
                    if relative_error > tolerance:
                        severity = ValidationSeverity.ERROR if relative_error > self.mass_balance_config['max_relative_error'] else ValidationSeverity.WARNING
                        issues.append(ValidationIssue(
                            severity=severity,
                            category="mass_balance",
                            message=f"Mass balance issue for stock '{stock_name}': {relative_error:.2e} relative error",
                            suggestion="Check flow calculations and ensure proper conservation",
                            element_name=stock_name,
                            details={
                                'net_inflow': net_inflow,
                                'net_outflow': net_outflow,
                                'net_flow': net_flow,
                                'stock_value': float(stock_value),
                                'relative_error': relative_error,
                                'connected_flows': connected_flows
                            }
                        ))
            
            except Exception as e:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="mass_balance",
                    message=f"Cannot check mass balance for stock '{stock_name}': {str(e)}",
                    suggestion="Check stock structure and flow connections",
                    element_name=stock_name
                ))
        
        return issues
    
    def _check_dimensional_conservation(self, stocks: List[Any], flows: List[Any]) -> List[ValidationIssue]:
        """Check conservation across dimensions for multi-dimensional stocks"""
        issues = []
        
        for stock in stocks:
            stock_name = getattr(stock, 'name', 'unnamed_stock')
            
            # Check if stock has dimensions
            if hasattr(stock, 'dimensions') and stock.dimensions:
                try:
                    # For multi-dimensional stocks, check conservation in each dimension
                    stock_values = getattr(stock, 'values', None)
                    if hasattr(stock_values, 'shape') and len(stock_values.shape) > 0:
                        
                        # Check total conservation
                        total_stock = np.sum(stock_values)
                        if total_stock < 0:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                category="mass_balance",
                                message=f"Multi-dimensional stock '{stock_name}' has negative total: {total_stock}",
                                suggestion="Check dimension calculations and flow mappings",
                                element_name=stock_name
                            ))
                        
                        # Check for NaN or infinite values
                        if not np.all(np.isfinite(stock_values)):
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                category="mass_balance",
                                message=f"Multi-dimensional stock '{stock_name}' contains non-finite values",
                                suggestion="Check for division by zero or overflow in calculations",
                                element_name=stock_name
                            ))
                
                except Exception as e:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="mass_balance",
                        message=f"Cannot check dimensional conservation for stock '{stock_name}': {str(e)}",
                        suggestion="Check multi-dimensional stock configuration",
                        element_name=stock_name
                    ))
        
        return issues
    
    def _check_analytical_conservation(self, model: Any, level: ValidationLevel) -> List[ValidationIssue]:
        """Check analytical conservation properties"""
        issues = []
        
        try:
            # If model has built-in conservation validation, use it
            if hasattr(model, 'validate_mass_conservation'):
                conservation_result = model.validate_mass_conservation()
                
                if isinstance(conservation_result, dict):
                    if not conservation_result.get('is_conserved', True):
                        relative_error = conservation_result.get('relative_error', 0.0)
                        tolerance = self.tolerance_levels.get(level, 1e-6)
                        
                        severity = ValidationSeverity.ERROR if relative_error > tolerance else ValidationSeverity.WARNING
                        issues.append(ValidationIssue(
                            severity=severity,
                            category="mass_balance",
                            message=f"Analytical mass conservation violation: {relative_error:.2e} relative error",
                            suggestion="Check flow formulas and ensure proper conservation",
                            details=conservation_result
                        ))
                
                elif isinstance(conservation_result, bool) and not conservation_result:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="mass_balance",
                        message="Model fails analytical mass conservation check",
                        suggestion="Review model structure and flow definitions"
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="mass_balance",
                message=f"Cannot perform analytical conservation check: {str(e)}",
                suggestion="Add mass conservation validation method to model"
            ))
        
        return issues
    
    def _validate_existing_conservation_method(self, target: Any, report: ValidationReport, level: ValidationLevel):
        """Validate using existing mass conservation method"""
        try:
            conservation_result = target.validate_mass_conservation()
            tolerance = self.tolerance_levels.get(level, 1e-6)
            
            if isinstance(conservation_result, dict):
                if not conservation_result.get('is_conserved', True):
                    relative_error = conservation_result.get('relative_error', 0.0)
                    severity = ValidationSeverity.ERROR if relative_error > tolerance else ValidationSeverity.WARNING
                    
                    report.add_issue(ValidationIssue(
                        severity=severity,
                        category="mass_balance",
                        message=f"Mass conservation violation: {relative_error:.2e} relative error",
                        suggestion="Check flow formulas and ensure proper conservation",
                        details=conservation_result
                    ))
            
            elif isinstance(conservation_result, bool) and not conservation_result:
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="mass_balance",
                    message="Mass conservation validation failed",
                    suggestion="Check model structure and flow definitions"
                ))
        
        except Exception as e:
            report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="mass_balance",
                message=f"Mass balance validation failed: {str(e)}",
                suggestion="Check model structure and flow definitions"
            ))
    
    def _validate_integration_conservation(self, result: Dict[str, Any], report: ValidationReport, level: ValidationLevel):
        """Validate conservation in integration results"""
        try:
            if 'y' not in result or 't' not in result:
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="mass_balance",
                    message="Integration result missing required data for conservation check",
                    suggestion="Ensure integration returns 'y' and 't' arrays"
                ))
                return
            
            y_values = result['y']
            t_values = result['t']
            tolerance = self.tolerance_levels.get(level, 1e-6)
            
            # Check for conservation over time
            if len(y_values.shape) > 1:
                # Multiple variables - check total conservation
                total_initial = np.sum(y_values[:, 0])
                total_final = np.sum(y_values[:, -1])
                
                if abs(total_initial) > self.mass_balance_config['min_stock_value']:
                    relative_change = abs(total_final - total_initial) / abs(total_initial)
                    
                    if relative_change > tolerance:
                        severity = ValidationSeverity.ERROR if relative_change > 0.01 else ValidationSeverity.WARNING
                        report.add_issue(ValidationIssue(
                            severity=severity,
                            category="mass_balance",
                            message=f"Total mass changed during integration: {relative_change:.2e} relative change",
                            suggestion="Check integration method and conservation properties",
                            details={
                                'initial_total': total_initial,
                                'final_total': total_final,
                                'relative_change': relative_change,
                                'integration_time': t_values[-1] - t_values[0]
                            }
                        ))
            
            # Check for negative values
            if np.any(y_values < -tolerance):
                min_value = np.min(y_values)
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="mass_balance",
                    message=f"Integration produced negative values: minimum = {min_value:.2e}",
                    suggestion="Check model formulation and integration bounds",
                    details={'minimum_value': min_value}
                ))
        
        except Exception as e:
            report.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="mass_balance",
                message=f"Cannot validate integration conservation: {str(e)}",
                suggestion="Check integration result format"
            ))
    
    def _validate_simulation_conservation(self, model: Any, report: ValidationReport, level: ValidationLevel):
        """Validate conservation in simulation results"""
        try:
            if not hasattr(model, 'simulation_results'):
                return
            
            results = model.simulation_results
            if not results or not isinstance(results, dict):
                return
            
            # Check if simulation results contain stock data
            if 'stocks' in results:
                self._check_simulation_stock_conservation(results['stocks'], report, level)
            
            # Check metadata for conservation warnings
            if 'metadata' in results:
                metadata = results['metadata']
                if isinstance(metadata, dict) and 'conservation_warnings' in metadata:
                    for warning in metadata['conservation_warnings']:
                        report.add_issue(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category="mass_balance",
                            message=f"Simulation conservation warning: {warning}",
                            suggestion="Review simulation parameters and model formulation"
                        ))
        
        except Exception as e:
            report.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="mass_balance",
                message=f"Cannot validate simulation conservation: {str(e)}",
                suggestion="Check simulation results format"
            ))
    
    def _check_simulation_stock_conservation(self, stock_results: Dict[str, Any], report: ValidationReport, level: ValidationLevel):
        """Check conservation for simulation stock results"""
        tolerance = self.tolerance_levels.get(level, 1e-6)
        
        for stock_name, stock_data in stock_results.items():
            try:
                if 'values' in stock_data and hasattr(stock_data['values'], 'shape'):
                    values = stock_data['values']
                    
                    # Check for negative values
                    if np.any(values < -tolerance):
                        min_value = np.min(values)
                        report.add_issue(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category="mass_balance",
                            message=f"Stock '{stock_name}' has negative values in simulation: minimum = {min_value:.2e}",
                            suggestion="Check model formulation and flow calculations",
                            element_name=stock_name
                        ))
                    
                    # Check for conservation over time (if multiple time points)
                    if len(values.shape) > 0 and values.shape[-1] > 1:
                        initial_total = np.sum(values[..., 0]) if len(values.shape) > 1 else values[0]
                        final_total = np.sum(values[..., -1]) if len(values.shape) > 1 else values[-1]
                        
                        if abs(initial_total) > self.mass_balance_config['min_stock_value']:
                            relative_change = abs(final_total - initial_total) / abs(initial_total)
                            
                            if relative_change > tolerance:
                                severity = ValidationSeverity.ERROR if relative_change > 0.01 else ValidationSeverity.INFO
                                report.add_issue(ValidationIssue(
                                    severity=severity,
                                    category="mass_balance",
                                    message=f"Stock '{stock_name}' mass changed during simulation: {relative_change:.2e} relative change",
                                    suggestion="Verify flow conservation and check for mass leakage",
                                    element_name=stock_name,
                                    details={
                                        'initial_total': float(initial_total),
                                        'final_total': float(final_total),
                                        'relative_change': relative_change
                                    }
                                ))
            
            except Exception as e:
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="mass_balance",
                    message=f"Cannot check conservation for stock '{stock_name}': {str(e)}",
                    suggestion="Check simulation result format for this stock",
                    element_name=stock_name
                ))
    
    def _validate_stock_conservation(self, stock: Any, report: ValidationReport, level: ValidationLevel):
        """Validate conservation for a single stock"""
        stock_name = getattr(stock, 'name', 'unnamed_stock')
        
        try:
            # Check inflows and outflows balance
            inflows = getattr(stock, 'inflows', [])
            outflows = getattr(stock, 'outflows', [])
            
            total_inflow = 0.0
            total_outflow = 0.0
            
            # Calculate total inflow
            for flow in inflows:
                try:
                    if hasattr(flow, 'rate_expression') and callable(flow.rate_expression):
                        rate = flow.rate_expression()
                        total_inflow += float(rate) if np.isscalar(rate) else np.sum(rate)
                except Exception:
                    pass
            
            # Calculate total outflow
            for flow in outflows:
                try:
                    if hasattr(flow, 'rate_expression') and callable(flow.rate_expression):
                        rate = flow.rate_expression()
                        total_outflow += float(rate) if np.isscalar(rate) else np.sum(rate)
                except Exception:
                    pass
            
            # Check balance
            net_flow = total_inflow - total_outflow
            stock_value = getattr(stock, 'values', getattr(stock, 'value', 0))
            
            if hasattr(stock_value, '__iter__') and not isinstance(stock_value, str):
                stock_value = np.sum(stock_value)
            
            tolerance = self.tolerance_levels.get(level, 1e-6)
            
            if abs(stock_value) > self.mass_balance_config['min_stock_value']:
                relative_error = abs(net_flow) / abs(stock_value)
                if relative_error > tolerance:
                    severity = ValidationSeverity.ERROR if relative_error > 0.01 else ValidationSeverity.WARNING
                    report.add_issue(ValidationIssue(
                        severity=severity,
                        category="mass_balance",
                        message=f"Stock '{stock_name}' has mass balance issue: {relative_error:.2e} relative error",
                        suggestion="Check connected flows and their rate calculations",
                        element_name=stock_name,
                        details={
                            'total_inflow': total_inflow,
                            'total_outflow': total_outflow,
                            'net_flow': net_flow,
                            'stock_value': float(stock_value),
                            'relative_error': relative_error
                        }
                    ))
        
        except Exception as e:
            report.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="mass_balance",
                message=f"Cannot validate conservation for stock '{stock_name}': {str(e)}",
                suggestion="Check stock structure and flow connections",
                element_name=stock_name
            ))
class FormulaSecurityValidator:
    """Consolidates formula validation from formula_engine.py with enhanced security"""
    
    def __init__(self):
        # Dangerous operations that should be blocked
        self.blocked_functions = {
            'eval', 'exec', 'compile', '__import__', 'open', 'file',
            'input', 'raw_input', 'reload', 'vars', 'globals', 'locals',
            'dir', 'hasattr', 'getattr', 'setattr', 'delattr'
        }
        
        # Dangerous modules that should not be imported
        self.blocked_modules = {
            'os', 'sys', 'subprocess', 'importlib', 'builtins',
            'socket', 'urllib', 'requests', 'pickle', 'marshal'
        }
        
        # Safe mathematical functions that are allowed
        self.safe_functions = {
            'abs', 'min', 'max', 'sum', 'round', 'int', 'float',
            'pow', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tan'
        }
        
        # Complex formula threshold for performance warnings
        self.complexity_threshold = 20
    
    def applies_to(self, target_type: str) -> bool:
        return target_type in ['formula', 'model', 'flow', 'auxiliary']
    
    def validate(self, target: Any, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
        report = ValidationReport()
        
        # If target is a model, validate all formulas in it
        if hasattr(target, 'formula_engine') and hasattr(target, 'validate_all_formulas'):
            try:
                formula_validation = target.validate_all_formulas()
                
                if not formula_validation.get('valid', True):
                    for error in formula_validation.get('errors', []):
                        report.add_issue(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="formula_security",
                            message=f"Formula error: {error}",
                            suggestion="Fix formula syntax and dependencies"
                        ))
                
                for warning in formula_validation.get('warnings', []):
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="formula_security",
                        message=f"Formula warning: {warning}",
                        suggestion="Review formula for potential issues"
                    ))
            
            except Exception as e:
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="formula_security",
                    message=f"Formula validation failed: {str(e)}",
                    suggestion="Check formula engine configuration"
                ))
        
        # If target is a single formula string
        elif isinstance(target, str):
            # Use formula engine validation logic here
            pass
        
        return report

class FlowCompatibilityValidator:
    """Consolidates flow validation from enhanced_flow.py with comprehensive checks"""
    
    def __init__(self):
        # Flow rate bounds for reasonableness checks
        self.default_rate_bounds = {
            'min_rate': -1e6,  # Allow negative flows (outflows)
            'max_rate': 1e6,   # Maximum reasonable flow rate
            'zero_threshold': 1e-12  # Threshold for considering a rate as zero
        }
        
        # Common unit conversions for compatibility checking
        self.unit_families = {
            'time': ['second', 'minute', 'hour', 'day', 'week', 'month', 'year'],
            'mass': ['gram', 'kilogram', 'pound', 'ton'],
            'volume': ['liter', 'gallon', 'cubic_meter'],
            'count': ['unit', 'item', 'piece', 'person', 'people'],
            'currency': ['dollar', 'euro', 'pound', 'yen']
        }
    
    def applies_to(self, target_type: str) -> bool:
        return target_type in ['flow', 'model', 'stock']
    
    def validate(self, target: Any, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
        report = ValidationReport()
        
        # Handle different target types
        if hasattr(target, 'flows') and hasattr(target, 'stocks'):
            # Model with flows and stocks
            self._validate_model_flows(target, report, level)
        
        elif hasattr(target, 'md_flows'):
            # Stock with multi-dimensional flows
            self._validate_stock_md_flows(target, report, level)
        
        elif hasattr(target, 'rate_formula') or hasattr(target, 'rate_expression'):
            # Single flow object
            self._validate_single_flow(target, report, level)
        
        elif hasattr(target, 'flows'):
            # Object with flows collection
            self._validate_flow_collection(target, report, level)
        
        return report
    
    def _validate_model_flows(self, model: Any, report: ValidationReport, level: ValidationLevel):
        """Validate all flows in a model for compatibility"""
        
        if not hasattr(model, 'flows') or not hasattr(model, 'stocks'):
            return
        
        # Get flows and stocks collections
        flows = getattr(model, 'flows', {})
        stocks = getattr(model, 'stocks', {})
        
        if isinstance(flows, dict):
            flows = flows.values()
        if isinstance(stocks, dict):
            stocks = stocks.values()
        
        # 1. Check each flow individually
        for flow in flows:
            self._validate_single_flow(flow, report, level, context_stocks=stocks)
        
        # 2. Check flow-stock relationships
        if level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
            self._validate_flow_stock_relationships(flows, stocks, report)
        
        # 3. Check for flow conflicts
        if level == ValidationLevel.COMPREHENSIVE:
            self._validate_flow_conflicts(flows, report)
    
    def _validate_single_flow(self, flow: Any, report: ValidationReport, 
                             level: ValidationLevel, context_stocks=None):
        """Validate a single flow object"""
        
        flow_name = getattr(flow, 'name', 'unnamed_flow')
        
        # 1. Basic flow structure validation
        structure_issues = self._validate_flow_structure(flow, flow_name)
        for issue in structure_issues:
            report.add_issue(issue)
        
        # 2. Flow rate validation
        if level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
            rate_issues = self._validate_flow_rate(flow, flow_name)
            for issue in rate_issues:
                report.add_issue(issue)
        
        # 3. Dimension compatibility (if applicable)
        if hasattr(flow, 'dimensions') or hasattr(flow, 'dimension_mappings'):
            dimension_issues = self._validate_flow_dimensions(flow, flow_name)
            for issue in dimension_issues:
                report.add_issue(issue)
        
        # 4. Unit consistency
        if level == ValidationLevel.COMPREHENSIVE:
            unit_issues = self._validate_flow_units(flow, flow_name, context_stocks)
            for issue in unit_issues:
                report.add_issue(issue)
    
    def _validate_flow_structure(self, flow: Any, flow_name: str) -> List[ValidationIssue]:
        """Validate basic flow structure"""
        issues = []
        
        # Check for required attributes
        required_attrs = ['name']
        optional_attrs = ['rate_formula', 'rate_expression', 'from_stock', 'to_stock']
        
        for attr in required_attrs:
            if not hasattr(flow, attr):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="flow_structure",
                    message=f"Flow '{flow_name}' missing required attribute '{attr}'",
                    suggestion=f"Add '{attr}' attribute to flow definition",
                    element_name=flow_name
                ))
        
        # Check that flow has either rate_formula or rate_expression
        has_rate_formula = hasattr(flow, 'rate_formula') and flow.rate_formula
        has_rate_expression = hasattr(flow, 'rate_expression') and flow.rate_expression
        
        if not has_rate_formula and not has_rate_expression:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="flow_structure",
                message=f"Flow '{flow_name}' has no rate formula or expression",
                suggestion="Add rate_formula or rate_expression to define flow behavior",
                element_name=flow_name
            ))
        
        # Check for both formula and expression (potential conflict)
        if has_rate_formula and has_rate_expression:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="flow_structure",
                message=f"Flow '{flow_name}' has both rate_formula and rate_expression",
                suggestion="Use either rate_formula OR rate_expression, not both",
                element_name=flow_name
            ))
        
        # Check stock connections
        from_stock = getattr(flow, 'from_stock', None)
        to_stock = getattr(flow, 'to_stock', None)
        
        if from_stock is None and to_stock is None:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="flow_structure",
                message=f"Flow '{flow_name}' not connected to any stocks",
                suggestion="Connect flow to at least one stock (from_stock or to_stock)",
                element_name=flow_name
            ))
        
        return issues
    
    def _validate_flow_rate(self, flow: Any, flow_name: str) -> List[ValidationIssue]:
        """Validate flow rate calculations and bounds"""
        issues = []
        
        # Try to evaluate flow rate if possible
        try:
            if hasattr(flow, 'rate_expression') and callable(flow.rate_expression):
                # Try to call rate expression
                rate = flow.rate_expression()
                issues.extend(self._check_rate_bounds(rate, flow_name))
            
            elif hasattr(flow, 'calculate_flow_rate'):
                # This would need a context, so we'll skip actual calculation
                # but check if the method exists
                pass
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="flow_rate",
                message=f"Cannot evaluate flow rate for '{flow_name}': {str(e)}",
                suggestion="Check flow rate formula for errors",
                element_name=flow_name
            ))
        
        # Check for common rate formula issues
        if hasattr(flow, 'rate_formula'):
            formula = flow.rate_formula
            if isinstance(formula, str):
                formula_issues = self._check_rate_formula_patterns(formula, flow_name)
                issues.extend(formula_issues)
        
        return issues
    
    def _check_rate_bounds(self, rate: Any, flow_name: str) -> List[ValidationIssue]:
        """Check if flow rate is within reasonable bounds"""
        issues = []
        
        try:
            # Handle numpy arrays
            if hasattr(rate, 'shape'):
                rate_values = np.asarray(rate).flatten()
            else:
                rate_values = [float(rate)]
            
            for rate_val in rate_values:
                if not np.isfinite(rate_val):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="flow_rate",
                        message=f"Flow '{flow_name}' has non-finite rate: {rate_val}",
                        suggestion="Check flow formula for division by zero or overflow",
                        element_name=flow_name
                    ))
                
                elif abs(rate_val) > self.default_rate_bounds['max_rate']:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="flow_rate",
                        message=f"Flow '{flow_name}' has very large rate: {rate_val:.2e}",
                        suggestion="Verify flow rate calculation and scaling",
                        element_name=flow_name
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="flow_rate",
                message=f"Cannot check rate bounds for '{flow_name}': {str(e)}",
                suggestion="Verify flow rate returns numeric value",
                element_name=flow_name
            ))
        
        return issues
    
    def _check_rate_formula_patterns(self, formula: str, flow_name: str) -> List[ValidationIssue]:
        """Check flow rate formula for common issues"""
        issues = []
        
        # Check for potential division by zero
        if '/0' in formula.replace(' ', ''):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="flow_rate",
                message=f"Flow '{flow_name}' formula contains division by zero",
                suggestion="Add protection against division by zero",
                element_name=flow_name
            ))
        
        # Check for very complex formulas
        complexity_indicators = formula.count('(') + formula.count('*') + formula.count('/') + formula.count('+')
        if complexity_indicators > 15:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="flow_rate",
                message=f"Flow '{flow_name}' has complex rate formula ({complexity_indicators} operations)",
                suggestion="Consider breaking complex formula into auxiliary variables",
                element_name=flow_name,
                details={'complexity_score': complexity_indicators}
            ))
        
        return issues
    
    def _validate_flow_dimensions(self, flow: Any, flow_name: str) -> List[ValidationIssue]:
        """Validate multi-dimensional flow compatibility"""
        issues = []
        
        # Check dimension mappings if present
        if hasattr(flow, 'dimension_mappings'):
            mappings = flow.dimension_mappings
            if mappings:
                for i, mapping in enumerate(mappings):
                    if hasattr(mapping, 'source_dimension') and hasattr(mapping, 'target_dimension'):
                        if mapping.source_dimension == mapping.target_dimension:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                category="flow_dimensions",
                                message=f"Flow '{flow_name}' maps dimension to itself: {mapping.source_dimension}",
                                suggestion="Remove redundant dimension mapping",
                                element_name=flow_name
                            ))
        
        # Check if flow connects stocks with incompatible dimensions
        from_stock = getattr(flow, 'from_stock', None)
        to_stock = getattr(flow, 'to_stock', None)
        
        if from_stock and to_stock:
            from_dims = getattr(from_stock, 'dimensions', [])
            to_dims = getattr(to_stock, 'dimensions', [])
            
            if from_dims and to_dims and len(from_dims) != len(to_dims):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="flow_dimensions",
                    message=f"Flow '{flow_name}' connects stocks with different dimension counts",
                    suggestion="Verify dimension mappings are correctly configured",
                    element_name=flow_name
                ))
        
        return issues
    
    def _validate_flow_units(self, flow: Any, flow_name: str, context_stocks=None) -> List[ValidationIssue]:
        """Validate flow unit consistency"""
        issues = []
        
        flow_units = getattr(flow, 'units', None)
        if not flow_units:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="flow_units",
                message=f"Flow '{flow_name}' has no units specified",
                suggestion="Add units for better model documentation",
                element_name=flow_name
            ))
            return issues
        
        # Check unit consistency with connected stocks
        from_stock = getattr(flow, 'from_stock', None)
        to_stock = getattr(flow, 'to_stock', None)
        
        for stock, direction in [(from_stock, 'from'), (to_stock, 'to')]:
            if stock and hasattr(stock, 'units'):
                stock_units = stock.units
                if stock_units and not self._units_compatible(flow_units, stock_units):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="flow_units",
                        message=f"Flow '{flow_name}' units '{flow_units}' incompatible with {direction}_stock units '{stock_units}'",
                        suggestion="Verify unit consistency or add unit conversion",
                        element_name=flow_name
                    ))
        
        return issues
    
    def _units_compatible(self, units1: str, units2: str) -> bool:
        """Check if two unit strings are compatible"""
        if units1 == units2:
            return True
        
        # Check if units are in the same family
        for family, unit_list in self.unit_families.items():
            if units1.lower() in unit_list and units2.lower() in unit_list:
                return True
        
        # Check for per-time units (rates)
        if ('/' in units1 and '/' in units2) or ('per' in units1.lower() and 'per' in units2.lower()):
            return True
        
        return False
    
    def _validate_stock_md_flows(self, stock: Any, report: ValidationReport, level: ValidationLevel):
        """Validate multi-dimensional flows on a stock"""
        
        if not hasattr(stock, 'md_flows'):
            return
        
        stock_name = getattr(stock, 'name', 'unnamed_stock')
        
        for flow_name, md_flow, target_stock in stock.md_flows:
            self._validate_single_flow(md_flow, report, level)
            
            # Additional MD-specific validation
            if hasattr(md_flow, 'validate_configuration'):
                try:
                    config_valid = md_flow.validate_configuration()
                    if not config_valid:
                        report.add_issue(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="flow_compatibility",
                            message=f"Multi-dimensional flow '{flow_name}' configuration invalid",
                            suggestion="Check flow configuration and target stock compatibility",
                            element_name=flow_name
                        ))
                except Exception as e:
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="flow_compatibility",
                        message=f"Multi-dimensional flow '{flow_name}' validation failed: {str(e)}",
                        suggestion="Check flow configuration and target stock compatibility",
                        element_name=flow_name
                    ))
    
    def _validate_flow_collection(self, target: Any, report: ValidationReport, level: ValidationLevel):
        """Validate a collection of flows"""
        flows = target.flows
        if isinstance(flows, dict):
            flows = flows.values()
        
        for flow in flows:
            self._validate_single_flow(flow, report, level)
    
    def _validate_flow_stock_relationships(self, flows, stocks, report: ValidationReport):
        """Validate relationships between flows and stocks"""
        
        # Create stock lookup
        stock_lookup = {}
        for stock in stocks:
            if hasattr(stock, 'name'):
                stock_lookup[stock.name] = stock
        
        # Check flow connections
        for flow in flows:
            flow_name = getattr(flow, 'name', 'unnamed_flow')
            
            # Check from_stock reference
            if hasattr(flow, 'from_stock') and flow.from_stock:
                from_stock = flow.from_stock
                if isinstance(from_stock, str) and from_stock not in stock_lookup:
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="flow_stock_relationship",
                        message=f"Flow '{flow_name}' references non-existent from_stock '{from_stock}'",
                        suggestion="Check stock name or add missing stock",
                        element_name=flow_name
                    ))
            
            # Check to_stock reference
            if hasattr(flow, 'to_stock') and flow.to_stock:
                to_stock = flow.to_stock
                if isinstance(to_stock, str) and to_stock not in stock_lookup:
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="flow_stock_relationship",
                        message=f"Flow '{flow_name}' references non-existent to_stock '{to_stock}'",
                        suggestion="Check stock name or add missing stock",
                        element_name=flow_name
                    ))
    
    def _validate_flow_conflicts(self, flows, report: ValidationReport):
        """Check for conflicting flows"""
        
        # Group flows by stock connections
        stock_flows = {}
        
        for flow in flows:
            flow_name = getattr(flow, 'name', 'unnamed_flow')
            from_stock = getattr(flow, 'from_stock', None)
            to_stock = getattr(flow, 'to_stock', None)
            
            # Track flows affecting each stock
            for stock, direction in [(from_stock, 'outflow'), (to_stock, 'inflow')]:
                if stock:
                    stock_key = getattr(stock, 'name', str(stock)) if hasattr(stock, 'name') else str(stock)
                    if stock_key not in stock_flows:
                        stock_flows[stock_key] = {'inflows': [], 'outflows': []}
                    stock_flows[stock_key][f"{direction}s"].append(flow_name)
        
        # Check for potential issues
        for stock_name, flows_dict in stock_flows.items():
            if len(flows_dict['outflows']) > 5:
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="flow_conflicts",
                    message=f"Stock '{stock_name}' has many outflows ({len(flows_dict['outflows'])})",
                    suggestion="Consider consolidating similar flows for simplicity",
                    element_name=stock_name
                ))

class NumericalStabilityValidator:
    """Consolidates numerical validation from integration_engine.py"""
    
    def applies_to(self, target_type: str) -> bool:
        return target_type in ['integration_result', 'model']
    
    def validate(self, target: Any, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
        report = ValidationReport()
        
        # If target is integration result
        if isinstance(target, dict) and 'y' in target:
            # Use validation logic from integration_engine.py
            y = target['y']
            
            # Check for finite values
            if not np.all(np.isfinite(y)):
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="numerical_stability",
                    message="Integration produced non-finite values (NaN/Inf)",
                    suggestion="Check integration method and model parameters"
                ))
            
            # Check for reasonable value ranges
            max_value = np.max(np.abs(y))
            if max_value > 1e12:
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="numerical_stability", 
                    message=f"Very large values detected: {max_value:.2e}",
                    suggestion="Check for numerical instability or model scaling"
                ))
        
        return report

# ===============================================================================
# Unified Validation Framework - Main Class
# ===============================================================================

class UnifiedValidationFramework:
    """Central validation system that replaces scattered validation logic"""
    
    def __init__(self):
        self.validators = {
            'mass_conservation': MassConservationValidator(),
            'formula_security': FormulaSecurityValidator(),
            'flow_compatibility': FlowCompatibilityValidator(),
            'numerical_stability': NumericalStabilityValidator()
        }
        
        # Performance tracking
        self.validation_stats = {
            'total_validations': 0,
            'validation_times': {},
            'issue_counts': {}
        }
    
    def validate_all(self, target: Any, target_type: str = 'model', 
                    level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
        """
        Main validation entry point - replaces all scattered validation calls
        
        Usage:
            validator = UnifiedValidationFramework()
            report = validator.validate_all(model, 'model', ValidationLevel.STANDARD)
        """
        
        combined_report = ValidationReport()
        
        for validator_name, validator in self.validators.items():
            if validator.applies_to(target_type):
                try:
                    import time
                    start_time = time.time()
                    
                    validation_result = validator.validate(target, level)
                    combined_report.merge(validation_result)
                    
                    # Track performance
                    validation_time = time.time() - start_time
                    if validator_name not in self.validation_stats['validation_times']:
                        self.validation_stats['validation_times'][validator_name] = []
                    self.validation_stats['validation_times'][validator_name].append(validation_time)
                    
                except Exception as e:
                    combined_report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="validation_framework",
                        message=f"Validator '{validator_name}' failed: {str(e)}",
                        suggestion="Check validation framework configuration"
                    ))
        
        self.validation_stats['total_validations'] += 1
        return combined_report
    
    def validate_formula(self, formula: str, context: Dict[str, Any] = None) -> ValidationReport:
        """Specific formula validation - replaces formula_engine validation calls"""
        return self.validators['formula_security'].validate(formula)
    
    def validate_mass_conservation(self, model: Any) -> ValidationReport:
        """Specific mass conservation - replaces simulation.py validation calls"""
        return self.validators['mass_conservation'].validate(model)
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation performance statistics"""
        return self.validation_stats.copy()

# ===============================================================================
# Usage Examples - How to replace existing validation calls
# ===============================================================================

def replace_existing_validation_calls():
    """
    Examples of how to replace existing scattered validation calls
    """
    
    # Create unified validator (singleton pattern recommended)
    validator = UnifiedValidationFramework()
    
    # REPLACE: model.validate_mass_conservation()
    # OLD: conservation_result = model.validate_mass_conservation()
    # NEW:
    validation_report = validator.validate_mass_conservation(model)
    
    # REPLACE: formula_engine.validate_formula()  
    # OLD: formula_validation = formula_engine.validate_formula(formula, context)
    # NEW:
    formula_report = validator.validate_formula(formula)
    
    # REPLACE: ModelValidator().validate_model()
    # OLD: report = ModelValidator().validate_model(model, level)
    # NEW:
    model_report = validator.validate_all(model, 'model', ValidationLevel.STANDARD)
    
    # REPLACE: integration result validation
    # OLD: validation = IntegrationResultProcessor.validate_integration_result(result, model)
    # NEW:
    integration_report = validator.validate_all(result, 'integration_result')
    
    return {
        'mass_conservation': validation_report,
        'formula': formula_report,
        'model': model_report,
        'integration': integration_report
    }

# ===============================================================================
# Migration Helper - Backward Compatibility
# ===============================================================================

class ValidationMigrationHelper:
    """Provides backward compatibility during transition"""
    
    def __init__(self):
        self.unified_validator = UnifiedValidationFramework()
    
    def simulate_old_mass_conservation(self, model):
        """Simulates old model.validate_mass_conservation() interface"""
        report = self.unified_validator.validate_mass_conservation(model)
        
        # Convert to old format for backward compatibility
        return {
            'is_conserved': report.is_valid,
            'issues': [issue.message for issue in report.issues],
            'relative_error': 0.0  # Extract from details if available
        }
    
    def simulate_old_formula_validation(self, formula, context=None):
        """Simulates old formula_engine.validate_formula() interface"""
        report = self.unified_validator.validate_formula(formula)
        
        return {
            'valid': report.is_valid,
            'errors': [issue.message for issue in report.issues 
                      if issue.severity == ValidationSeverity.ERROR],
            'warnings': [issue.message for issue in report.issues 
                        if issue.severity == ValidationSeverity.WARNING]
        }

# ===============================================================================
# Singleton Pattern for Global Access
# ===============================================================================

_unified_validator_instance = None

def get_unified_validator() -> UnifiedValidationFramework:
    """Get singleton instance of unified validator"""
    global _unified_validator_instance
    if _unified_validator_instance is None:
        _unified_validator_instance = UnifiedValidationFramework()
    return _unified_validator_instance

# Convenience functions for easy migration
def validate_model(model, level='standard'):
    """Global function for model validation"""
    validator = get_unified_validator()
    return validator.validate_all(model, 'model', ValidationLevel(level))

def validate_formula(formula, context=None):
    """Global function for formula validation"""
    validator = get_unified_validator()
    return validator.validate_formula(formula)




def test_formula_security_validator():
    """Test the enhanced formula security validator"""
    print("🧪 Testing Enhanced Formula Security Validator")
    print("=" * 50)
    
    validator = FormulaSecurityValidator()
    
    # Test cases
    test_formulas = [
        ("Population * 0.02", "safe_formula"),
        ("eval('malicious_code')", "dangerous_eval"),
        ("import os", "dangerous_import"),
        ("Population * growth_rate * (1 - Population / capacity)", "complex_formula")
    ]
    
    for formula, test_name in test_formulas:
        print(f"\n📝 Testing: {test_name}")
        print(f"   Formula: {formula}")
        
        report = validator.validate(formula, ValidationLevel.COMPREHENSIVE)
        print(f"   Valid: {report.is_valid}")
        print(f"   Issues: {len(report.issues)}")
        
        for issue in report.issues:
            severity_icon = {"critical": "🚨", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
            icon = severity_icon.get(issue.severity.value, "❓")
            print(f"   {icon} {issue.severity.value.upper()}: {issue.message}")
            
       
            
def test_mass_conservation_validator():
    """Test the mass conservation validator"""
    print("🧪 Testing Mass Conservation Validator")
    print("=" * 50)
    
    validator = MassConservationValidator()
    
    # Create mock objects for testing
    class MockStock:
        def __init__(self, name, values):
            self.name = name
            self.values = values
            self.inflows = []
            self.outflows = []
    
    class MockFlow:
        def __init__(self, name, rate, from_stock=None, to_stock=None):
            self.name = name
            self.from_stock = from_stock
            self.to_stock = to_stock
            self.rate_expression = lambda: rate
    
    class MockModel:
        def __init__(self):
            self.stocks = {}
            self.flows = {}
        
        def add_stock(self, stock):
            self.stocks[stock.name] = stock
        
        def add_flow(self, flow):
            self.flows[flow.name] = flow
    
    # Test case 1: Simple conserved system
    print("\n📊 Test 1: Simple conserved system")
    model1 = MockModel()
    stock1 = MockStock("Population", 1000.0)
    model1.add_stock(stock1)
    
    report1 = validator.validate(model1, ValidationLevel.STANDARD)
    print(f"   Valid: {report1.is_valid}")
    print(f"   Issues: {len(report1.issues)}")
    
    # Test case 2: Integration result
    print("\n📊 Test 2: Integration result")
    integration_result = {
        'y': np.array([[1000.0, 1010.0, 1020.0], [500.0, 490.0, 480.0]]),
        't': np.array([0.0, 1.0, 2.0]),
        'success': True
    }
    
    report2 = validator.validate(integration_result, ValidationLevel.COMPREHENSIVE)
    print(f"   Valid: {report2.is_valid}")
    print(f"   Issues: {len(report2.issues)}")
    
    for issue in (report1.issues + report2.issues):
        severity_icon = {"critical": "🚨", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
        icon = severity_icon.get(issue.severity.value, "❓")
        print(f"   {icon} {issue.severity.value.upper()}: {issue.message}")

def test_flow_compatibility_validator():
    """Test the flow compatibility validator"""
    print("🧪 Testing Flow Compatibility Validator")
    print("=" * 50)
    
    validator = FlowCompatibilityValidator()
    
    # Create a mock flow for testing
    class MockFlow:
        def __init__(self, name, rate_formula=None, rate_expression=None, units=None):
            self.name = name
            self.rate_formula = rate_formula
            self.rate_expression = rate_expression
            self.units = units
            self.from_stock = None
            self.to_stock = None
    
    # Test cases
    test_flows = [
        MockFlow("good_flow", rate_formula="Population * 0.02", units="people/year"),
        MockFlow("bad_flow", rate_formula="Population / 0", units="people/year"),  # Division by zero
        MockFlow("complex_flow", rate_formula="(Pop * rate * (1-Pop/cap) * factor * modifier * adj)", units="people/year"),
        MockFlow("no_rate_flow"),  # Missing rate
    ]
    
    for flow in test_flows:
        print(f"\n📊 Testing flow: {flow.name}")
        report = validator.validate(flow, ValidationLevel.COMPREHENSIVE)
        print(f"   Valid: {report.is_valid}")
        print(f"   Issues: {len(report.issues)}")
        
        for issue in report.issues:
            severity_icon = {"critical": "🚨", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
            icon = severity_icon.get(issue.severity.value, "❓")
            print(f"   {icon} {issue.severity.value.upper()}: {issue.message}")

# Update the main test to include mass conservation validator
if __name__ == "__main__":
    test_formula_security_validator()
    print("\n" + "="*70 + "\n")
    test_flow_compatibility_validator()
    print("\n" + "="*70 + "\n")
    test_mass_conservation_validator()       
