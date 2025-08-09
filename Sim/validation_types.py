# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 11:57:21 2025

@author: NagabhushanamTattaga
"""

# validation_types.py - Shared validation data types and enums

from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json

# ===============================================================================
# Core Validation Enums
# ===============================================================================

class ValidationLevel(Enum):
    """Validation thoroughness levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    DEEP = "deep"

class ValidationSeverity(Enum):
    """Issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationCategory(Enum):
    """Validation category types"""
    MASS_BALANCE = "mass_balance"
    FORMULA_SECURITY = "formula_security"
    FLOW_COMPATIBILITY = "flow_compatibility"
    NUMERICAL_STABILITY = "numerical_stability"
    DIMENSIONAL_CONSISTENCY = "dimensional_consistency"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    PARAMETER_USAGE = "parameter_usage"
    PERFORMANCE = "performance"
    STRUCTURE = "structure"
    BEHAVIORAL = "behavioral"

# ===============================================================================
# Core Validation Data Structures
# ===============================================================================

@dataclass
class ValidationIssue:
    """Individual validation issue with comprehensive metadata"""
    severity: ValidationSeverity
    category: ValidationCategory
    message: str
    suggestion: str
    element_name: Optional[str] = None
    element_type: Optional[str] = None  # 'stock', 'flow', 'parameter', etc.
    location: Optional[str] = None      # File/line information
    details: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'severity': self.severity.value,
            'category': self.category.value,
            'message': self.message,
            'suggestion': self.suggestion,
            'element_name': self.element_name,
            'element_type': self.element_type,
            'location': self.location,
            'details': self.details,
            'timestamp': self.timestamp
        }

@dataclass 
class ValidationMetrics:
    """Validation performance and quality metrics"""
    execution_time: float = 0.0
    validators_run: List[str] = field(default_factory=list)
    issues_by_severity: Dict[str, int] = field(default_factory=dict)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def add_metric(self, name: str, value: Any):
        """Add custom metric"""
        self.custom_metrics[name] = value
    
    def get_metric(self, name: str, default: Any = None) -> Any:
        """Get custom metric with default"""
        return self.custom_metrics.get(name, default)

@dataclass
class ValidationReport:
    """Comprehensive validation report with metrics and statistics"""
    is_valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    validation_level: Optional[ValidationLevel] = None
    target_type: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def add_issue(self, issue: ValidationIssue):
        """Add validation issue and update statistics"""
        self.issues.append(issue)
        
        # Update validity based on severity
        if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False
        
        # Update metrics
        severity_key = issue.severity.value
        if severity_key not in self.metrics.issues_by_severity:
            self.metrics.issues_by_severity[severity_key] = 0
        self.metrics.issues_by_severity[severity_key] += 1
    
    def merge(self, other: 'ValidationReport'):
        """Merge another validation report into this one"""
        self.issues.extend(other.issues)
        
        # Update validity
        if not other.is_valid:
            self.is_valid = False
        
        # Merge metrics
        for severity, count in other.metrics.issues_by_severity.items():
            if severity not in self.metrics.issues_by_severity:
                self.metrics.issues_by_severity[severity] = 0
            self.metrics.issues_by_severity[severity] += count
        
        # Merge custom metrics
        self.metrics.custom_metrics.update(other.metrics.custom_metrics)
        
        # Merge validator lists
        self.metrics.validators_run.extend(other.metrics.validators_run)
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues of specific severity"""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_category(self, category: ValidationCategory) -> List[ValidationIssue]:
        """Get issues of specific category"""
        return [issue for issue in self.issues if issue.category == category]
    
    def get_metric(self, name: str, default: Any = None) -> Any:
        """Get custom metric from report"""
        return self.metrics.get_metric(name, default)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all custom metrics"""
        return self.metrics.custom_metrics.copy()
    
    @property
    def error_count(self) -> int:
        """Get number of errors"""
        return self.metrics.issues_by_severity.get('error', 0)
    
    @property
    def warning_count(self) -> int:
        """Get number of warnings"""
        return self.metrics.issues_by_severity.get('warning', 0)
    
    @property
    def critical_count(self) -> int:
        """Get number of critical issues"""
        return self.metrics.issues_by_severity.get('critical', 0)
    
    @property
    def total_issues(self) -> int:
        """Get total number of issues"""
        return len(self.issues)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization"""
        return {
            'is_valid': self.is_valid,
            'validation_level': self.validation_level.value if self.validation_level else None,
            'target_type': self.target_type,
            'timestamp': self.timestamp,
            'summary': {
                'total_issues': self.total_issues,
                'error_count': self.error_count,
                'warning_count': self.warning_count,
                'critical_count': self.critical_count
            },
            'issues': [issue.to_dict() for issue in self.issues],
            'metrics': {
                'execution_time': self.metrics.execution_time,
                'validators_run': self.metrics.validators_run,
                'issues_by_severity': self.metrics.issues_by_severity,
                'custom_metrics': self.metrics.custom_metrics
            }
        }
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export report as JSON"""
        json_data = json.dumps(self.to_dict(), indent=2, default=str)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_data)
        
        return json_data

# ===============================================================================
# Validation Context Classes
# ===============================================================================

@dataclass
class ValidationContext:
    """Context information for validation operations"""
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    target_type: str = "unknown"
    custom_validators: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Performance settings
    enable_caching: bool = True
    enable_parallel: bool = False
    timeout_seconds: Optional[float] = None
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.configuration.get(key, default)
    
    def set_config(self, key: str, value: Any):
        """Set configuration value"""
        self.configuration[key] = value

# ===============================================================================
# Validator Interface Protocol
# ===============================================================================

from typing import Protocol

class IValidator(Protocol):
    """Interface that all validators must implement"""
    
    def validate(self, target: Any, context: ValidationContext) -> ValidationReport:
        """Validate target object and return report"""
        ...
    
    def applies_to(self, target_type: str) -> bool:
        """Check if this validator applies to target type"""
        ...
    
    def get_name(self) -> str:
        """Get validator name"""
        ...
    
    def get_supported_levels(self) -> List[ValidationLevel]:
        """Get supported validation levels"""
        ...

# ===============================================================================
# Validation Configuration
# ===============================================================================

@dataclass
class ValidationConfiguration:
    """Global validation system configuration"""
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 1000
    enable_parallel_validation: bool = False
    max_validation_time: float = 30.0  # seconds
    
    # Validation behavior
    stop_on_first_error: bool = False
    include_performance_metrics: bool = True
    auto_fix_issues: bool = False
    
    # Reporting settings
    include_suggestions: bool = True
    include_element_details: bool = True
    max_issues_per_category: int = 100
    
    # Security settings
    enable_formula_sandboxing: bool = True
    max_formula_complexity: int = 1000
    
    # Custom validation settings
    custom_validators: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def default(cls) -> 'ValidationConfiguration':
        """Get default configuration"""
        return cls()
    
    @classmethod
    def production(cls) -> 'ValidationConfiguration':
        """Get production-optimized configuration"""
        return cls(
            enable_caching=True,
            enable_parallel_validation=True,
            stop_on_first_error=False,
            max_validation_time=60.0,
            include_performance_metrics=True
        )
    
    @classmethod
    def development(cls) -> 'ValidationConfiguration':
        """Get development-optimized configuration"""
        return cls(
            enable_caching=False,
            include_element_details=True,
            include_suggestions=True,
            auto_fix_issues=False
        )

# ===============================================================================
# Utility Functions
# ===============================================================================

def create_validation_issue(severity: str, category: str, message: str, 
                           suggestion: str, **kwargs) -> ValidationIssue:
    """Convenience function to create validation issues"""
    return ValidationIssue(
        severity=ValidationSeverity(severity),
        category=ValidationCategory(category) if hasattr(ValidationCategory, category.upper()) else ValidationCategory.STRUCTURE,
        message=message,
        suggestion=suggestion,
        **kwargs
    )

def merge_validation_reports(*reports: ValidationReport) -> ValidationReport:
    """Merge multiple validation reports into one"""
    if not reports:
        return ValidationReport()
    
    merged = ValidationReport()
    for report in reports:
        merged.merge(report)
    
    return merged

# ===============================================================================
# Backward Compatibility Helpers
# ===============================================================================

class LegacyValidationAdapter:
    """Adapter to convert between old and new validation formats"""
    
    @staticmethod
    def convert_old_mass_conservation(old_result: Dict[str, Any]) -> ValidationReport:
        """Convert old mass conservation result to new format"""
        report = ValidationReport()
        
        if not old_result.get('is_conserved', True):
            issue = create_validation_issue(
                severity='error',
                category='mass_balance',
                message=f"Mass conservation violation: {old_result.get('relative_error', 0):.2e}",
                suggestion="Check flow formulas and ensure proper conservation"
            )
            report.add_issue(issue)
        
        # Add metrics
        for key in ['relative_error', 'total_inflow', 'total_outflow', 'net_flow']:
            if key in old_result:
                report.metrics.add_metric(key, old_result[key])
        
        return report
    
    @staticmethod
    def convert_old_formula_validation(old_result: Dict[str, Any]) -> ValidationReport:
        """Convert old formula validation result to new format"""
        report = ValidationReport()
        
        # Convert errors
        for error in old_result.get('errors', []):
            issue = create_validation_issue(
                severity='error',
                category='formula_security',
                message=error,
                suggestion="Fix formula syntax and dependencies"
            )
            report.add_issue(issue)
        
        # Convert warnings
        for warning in old_result.get('warnings', []):
            issue = create_validation_issue(
                severity='warning',
                category='formula_security',
                message=warning,
                suggestion="Review formula for potential issues"
            )
            report.add_issue(issue)
        
        # Add metrics
        for key in ['dependencies', 'complexity_score']:
            if key in old_result:
                report.metrics.add_metric(key, old_result[key])
        
        return report