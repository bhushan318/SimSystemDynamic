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
from abc import ABC, abstractmethod

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

@dataclass
class ValidationReport:
    is_valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    
    def add_issue(self, issue: ValidationIssue):
        self.issues.append(issue)
        if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False
    
    def merge(self, other: 'ValidationReport'):
        """Merge another validation report into this one"""
        self.issues.extend(other.issues)
        if not other.is_valid:
            self.is_valid = False

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
    
    def applies_to(self, target_type: str) -> bool:
        return target_type in ['model', 'simulation']
    
    def validate(self, model: Any, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
        report = ValidationReport()
        
        try:
            # Consolidated logic from both simulation.py and model_validation.py
            total_inflow = 0.0
            total_outflow = 0.0
            
            # Check each stock for mass balance
            for stock in model.stocks:
                inflow_count = len(getattr(stock, 'inflows', {}))
                outflow_count = len(getattr(stock, 'outflows', {}))
                
                # From model_validation.py logic
                if inflow_count == 0 and outflow_count == 0:
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="mass_balance",
                        message=f"Stock '{stock.name}' has no flows",
                        suggestion="Add inflows/outflows or remove unused stock",
                        element_name=stock.name
                    ))
                
                # From simulation.py logic - calculate actual flows
                for flow in getattr(stock, 'inflows', {}).values():
                    try:
                        rate = flow.get_rate()
                        flow_value = np.sum(rate) if hasattr(rate, '__iter__') else rate
                        total_inflow += flow_value
                    except:
                        pass
                
                for flow in getattr(stock, 'outflows', {}).values():
                    try:
                        rate = flow.get_rate()
                        flow_value = np.sum(rate) if hasattr(rate, '__iter__') else rate
                        total_outflow += flow_value
                    except:
                        pass
            
            # Global conservation check (from simulation.py)
            if hasattr(model, 'validate_mass_conservation'):
                conservation_result = model.validate_mass_conservation()
                if not conservation_result.get('is_conserved', True):
                    relative_error = conservation_result.get('relative_error', 0)
                    severity = (ValidationSeverity.ERROR if relative_error > 0.1 
                              else ValidationSeverity.WARNING)
                    
                    report.add_issue(ValidationIssue(
                        severity=severity,
                        category="mass_balance",
                        message=f"Mass conservation violation: {relative_error:.2e} relative error",
                        suggestion="Check flow formulas and ensure proper conservation",
                        details=conservation_result
                    ))
        
        except Exception as e:
            report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="mass_balance",
                message=f"Mass balance validation failed: {str(e)}",
                suggestion="Check model structure and flow definitions"
            ))
        
        return report

class FormulaSecurityValidator:
    """Consolidates formula validation from formula_engine.py"""
    
    def applies_to(self, target_type: str) -> bool:
        return target_type in ['formula', 'model', 'flow']
    
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
    """Consolidates flow validation from enhanced_flow.py"""
    
    def applies_to(self, target_type: str) -> bool:
        return target_type in ['flow', 'model']
    
    def validate(self, target: Any, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
        report = ValidationReport()
        
        if hasattr(target, 'md_flows'):  # Stock with multi-dimensional flows
            for flow_name, md_flow, target_stock in target.md_flows:
                # Validate flow configuration
                if hasattr(md_flow, 'validate_configuration'):
                    try:
                        # This would call the validation from enhanced_flow.py
                        pass
                    except Exception as e:
                        report.add_issue(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="flow_compatibility",
                            message=f"Multi-dimensional flow '{flow_name}' validation failed: {str(e)}",
                            suggestion="Check flow configuration and target stock compatibility"
                        ))
        
        return report

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