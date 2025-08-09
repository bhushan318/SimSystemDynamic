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


import json
from unified_validation import UnifiedValidationFramework, ValidationLevel
from validation_types import ValidationReport, ValidationIssue, ValidationSeverity



# Keep only the main entry point function
def validate_model(model, level: str = "standard", include_behavioral: bool = True) -> ValidationReport:
    """Main model validation entry point - now delegates to unified framework"""
    
    validator = UnifiedValidationFramework()
    validation_level = ValidationLevel(level)
    
    # Use unified validation framework
    report = validator.validate_all(model, 'model', validation_level)
    
    # Add behavioral analysis if requested and not basic level
    if include_behavioral and level != 'basic':
        behavioral_report = validator.validate_all(model, 'behavioral', validation_level)
        report.merge(behavioral_report)
    
    return report

def quick_validate(model) -> bool:
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
        report = validate_model(model, ValidationLevel.STANDARD)
        
        print(f"✅ Validation completed!")
        print(f"   Model valid: {report.is_valid}")
        # print(f"   Issues found: {report.total_issues}")
        # print(f"   Categories tested: {len(set(issue.category for issue in report.issues))}")
        print(report)
        return True
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

if __name__ == "__main__":
    test_model_validation()