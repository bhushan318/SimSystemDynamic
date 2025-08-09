# -*- coding: utf-8 -*-
"""
Model Validation Framework - Unified Interface
Now serves as a clean interface to the unified validation framework
"""

import json
from typing import Dict, Any, Optional
from unified_validation import (
    get_unified_validator, 
    ValidationLevel, 
    ValidationReport, 
    ValidationIssue, 
    ValidationSeverity
)

# ===============================================================================
# Main Validation Interface - Simplified and Unified
# ===============================================================================

def validate_model(model: Any, level: str = "standard", include_behavioral: bool = True) -> ValidationReport:
    """
    Main model validation entry point - now fully unified
    
    Args:
        model: Model object to validate
        level: Validation level ("basic", "standard", "comprehensive")
        include_behavioral: Whether to include behavioral analysis
    
    Returns:
        ValidationReport with all validation results
    """
    
    try:
        # Get unified validator
        validator = get_unified_validator()
        validation_level = ValidationLevel(level)
        
        # Primary validation using unified framework
        report = validator.validate_all(model, 'model', validation_level)
        
        # Add behavioral analysis if requested and level permits
        if include_behavioral and level != 'basic':
            try:
                behavioral_report = validator.validate_all(model, 'behavioral', validation_level)
                report.merge(behavioral_report)
            except Exception as behavioral_error:
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="behavioral_analysis",
                    message=f"Behavioral analysis skipped: {str(behavioral_error)}",
                    suggestion="Behavioral analysis requires enhanced model features"
                ))
        
        # Add validation metadata
        report.metadata = {
            'validation_level': level,
            'include_behavioral': include_behavioral,
            'framework_version': 'unified_v1.0',
            'validator_count': len(validator.validators)
        }
        
        return report
        
    except Exception as e:
        # Create error report if validation fails
        error_report = ValidationReport()
        error_report.add_issue(ValidationIssue(
            severity=ValidationSeverity.CRITICAL,
            category="validation_framework",
            message=f"Validation framework failed: {str(e)}",
            suggestion="Check unified validation framework installation and configuration"
        ))
        return error_report

def quick_validate(model: Any) -> bool:
    """Quick validation - returns True if model passes basic validation"""
    try:
        report = validate_model(model, level="basic", include_behavioral=False)
        return report.is_valid
    except Exception:
        return False

def comprehensive_validate(model: Any) -> ValidationReport:
    """Comprehensive validation with all checks enabled"""
    return validate_model(model, level="comprehensive", include_behavioral=True)

def validate_model_changes(old_model: Any, new_model: Any, level: str = "standard") -> ValidationReport:
    """
    Validate changes between model versions
    
    Args:
        old_model: Previous model version
        new_model: Updated model version  
        level: Validation level
    
    Returns:
        ValidationReport focusing on changes and compatibility
    """
    
    try:
        validator = get_unified_validator()
        validation_level = ValidationLevel(level)
        
        # Validate new model
        new_report = validator.validate_all(new_model, 'model', validation_level)
        
        # Add change-specific validations
        change_issues = _validate_model_compatibility(old_model, new_model)
        for issue in change_issues:
            new_report.add_issue(issue)
        
        return new_report
        
    except Exception as e:
        error_report = ValidationReport()
        error_report.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            category="model_comparison",
            message=f"Model change validation failed: {str(e)}",
            suggestion="Check model compatibility and validation framework"
        ))
        return error_report

def _validate_model_compatibility(old_model: Any, new_model: Any) -> list[ValidationIssue]:
    """Check compatibility between model versions"""
    issues = []
    
    try:
        # Check if major structure changed
        old_stocks = _count_model_elements(old_model, 'stocks')
        new_stocks = _count_model_elements(new_model, 'stocks')
        old_flows = _count_model_elements(old_model, 'flows')
        new_flows = _count_model_elements(new_model, 'flows')
        
        if old_stocks != new_stocks:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="model_changes",
                message=f"Stock count changed: {old_stocks} → {new_stocks}",
                suggestion="Verify model changes are intentional"
            ))
        
        if old_flows != new_flows:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="model_changes", 
                message=f"Flow count changed: {old_flows} → {new_flows}",
                suggestion="Verify model changes are intentional"
            ))
        
        # Check for significant structural changes
        structure_change_ratio = abs(new_stocks + new_flows - old_stocks - old_flows) / max(old_stocks + old_flows, 1)
        if structure_change_ratio > 0.5:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="model_changes",
                message=f"Major structural change detected ({structure_change_ratio:.1%})",
                suggestion="Consider comprehensive re-validation and testing"
            ))
    
    except Exception as e:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.INFO,
            category="model_comparison",
            message=f"Cannot compare model structures: {str(e)}",
            suggestion="Manual review of model changes recommended"
        ))
    
    return issues

def _count_model_elements(model: Any, element_type: str) -> int:
    """Helper to count model elements"""
    try:
        if hasattr(model, element_type):
            elements = getattr(model, element_type)
            if isinstance(elements, dict):
                return len(elements)
            elif hasattr(elements, '__len__'):
                return len(elements)
        return 0
    except Exception:
        return 0

# ===============================================================================
# Enhanced Report Export and Analysis
# ===============================================================================

def export_validation_report(report: ValidationReport, filepath: str, format: str = "json"):
    """
    Export validation report in various formats
    
    Args:
        report: ValidationReport to export
        filepath: Output file path
        format: Export format ("json", "html", "csv")
    """
    
    try:
        if format.lower() == "json":
            _export_json_report(report, filepath)
        elif format.lower() == "html":
            _export_html_report(report, filepath)
        elif format.lower() == "csv":
            _export_csv_report(report, filepath)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        print(f"📄 Validation report exported to: {filepath}")
        
    except Exception as e:
        print(f"❌ Failed to export report: {str(e)}")

def _export_json_report(report: ValidationReport, filepath: str):
    """Export report as JSON"""
    
    report_dict = {
        'framework_version': 'unified_v1.0',
        'timestamp': getattr(report, 'timestamp', None),
        'is_valid': report.is_valid,
        'summary': {
            'total_issues': len(report.issues),
            'error_count': len([i for i in report.issues if i.severity == ValidationSeverity.ERROR]),
            'warning_count': len([i for i in report.issues if i.severity == ValidationSeverity.WARNING]),
            'info_count': len([i for i in report.issues if i.severity == ValidationSeverity.INFO]),
            'critical_count': len([i for i in report.issues if i.severity == ValidationSeverity.CRITICAL])
        },
        'issues': [
            {
                'severity': issue.severity.value,
                'category': issue.category,
                'message': issue.message,
                'suggestion': issue.suggestion,
                'element_name': issue.element_name,
                'details': issue.details
            } for issue in report.issues
        ],
        'metadata': getattr(report, 'metadata', {})
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, default=str, ensure_ascii=False)

def _export_html_report(report: ValidationReport, filepath: str):
    """Export report as HTML"""
    
    severity_colors = {
        'critical': '#dc3545',  # Red
        'error': '#fd7e14',     # Orange  
        'warning': '#ffc107',   # Yellow
        'info': '#17a2b8'       # Blue
    }
    
    severity_icons = {
        'critical': '🚨',
        'error': '❌', 
        'warning': '⚠️',
        'info': 'ℹ️'
    }
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
            .metric {{ background-color: #e9ecef; padding: 10px; border-radius: 5px; text-align: center; }}
            .issue {{ margin: 10px 0; padding: 10px; border-left: 4px solid; border-radius: 3px; }}
            .valid {{ color: green; font-weight: bold; }}
            .invalid {{ color: red; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Model Validation Report</h1>
            <p>Status: <span class="{'valid' if report.is_valid else 'invalid'}">
                {'✅ VALID' if report.is_valid else '❌ INVALID'}
            </span></p>
        </div>
        
        <div class="summary">
            <div class="metric">
                <div>Total Issues</div>
                <div><strong>{len(report.issues)}</strong></div>
            </div>
            <div class="metric">
                <div>Critical</div>
                <div><strong>{len([i for i in report.issues if i.severity == ValidationSeverity.CRITICAL])}</strong></div>
            </div>
            <div class="metric">
                <div>Errors</div>
                <div><strong>{len([i for i in report.issues if i.severity == ValidationSeverity.ERROR])}</strong></div>
            </div>
            <div class="metric">
                <div>Warnings</div>
                <div><strong>{len([i for i in report.issues if i.severity == ValidationSeverity.WARNING])}</strong></div>
            </div>
        </div>
        
        <h2>Issues</h2>
    """
    
    for issue in report.issues:
        color = severity_colors.get(issue.severity.value, '#6c757d')
        icon = severity_icons.get(issue.severity.value, '❓')
        
        html_content += f"""
        <div class="issue" style="border-left-color: {color};">
            <strong>{icon} {issue.severity.value.upper()}</strong> - {issue.category}<br>
            <strong>Message:</strong> {issue.message}<br>
            <strong>Suggestion:</strong> {issue.suggestion}
            {f'<br><strong>Element:</strong> {issue.element_name}' if issue.element_name else ''}
        </div>
        """
    
    html_content += """
        </body>
    </html>
    """
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)

def _export_csv_report(report: ValidationReport, filepath: str):
    """Export report as CSV"""
    import csv
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Severity', 'Category', 'Message', 'Suggestion', 'Element', 'Details'])
        
        # Issues
        for issue in report.issues:
            writer.writerow([
                issue.severity.value,
                issue.category,
                issue.message,
                issue.suggestion,
                issue.element_name or '',
                str(issue.details) if issue.details else ''
            ])

# ===============================================================================
# Validation Analysis and Insights
# ===============================================================================

def analyze_validation_patterns(reports: list[ValidationReport]) -> Dict[str, Any]:
    """
    Analyze patterns across multiple validation reports
    
    Args:
        reports: List of ValidationReport objects
    
    Returns:
        Dictionary with pattern analysis
    """
    
    if not reports:
        return {"error": "No reports provided"}
    
    analysis = {
        'total_reports': len(reports),
        'overall_validity_rate': sum(1 for r in reports if r.is_valid) / len(reports),
        'common_issues': {},
        'severity_distribution': {},
        'category_distribution': {}
    }
    
    # Collect all issues
    all_issues = []
    for report in reports:
        all_issues.extend(report.issues)
    
    # Analyze issue patterns
    issue_messages = {}
    for issue in all_issues:
        if issue.message not in issue_messages:
            issue_messages[issue.message] = 0
        issue_messages[issue.message] += 1
    
    # Top 10 most common issues
    analysis['common_issues'] = dict(sorted(issue_messages.items(), 
                                          key=lambda x: x[1], reverse=True)[:10])
    
    # Severity distribution
    for issue in all_issues:
        severity = issue.severity.value
        if severity not in analysis['severity_distribution']:
            analysis['severity_distribution'][severity] = 0
        analysis['severity_distribution'][severity] += 1
    
    # Category distribution
    for issue in all_issues:
        category = issue.category
        if category not in analysis['category_distribution']:
            analysis['category_distribution'][category] = 0
        analysis['category_distribution'][category] += 1
    
    return analysis

# ===============================================================================
# Testing and Validation Framework Tests
# ===============================================================================

def test_validation_framework():
    """Test the unified validation framework integration"""
    print("🧪 Testing Unified Model Validation Framework")
    print("=" * 60)
    
    try:
        # Test 1: Basic validation
        print("\n📊 Test 1: Quick validation")
        result = quick_validate(None)  # This should handle gracefully
        print(f"   Quick validate with None: {result}")
        
        # Test 2: Validation levels
        print("\n📊 Test 2: Validation levels")
        for level in ["basic", "standard", "comprehensive"]:
            try:
                report = validate_model(None, level=level, include_behavioral=False)
                print(f"   Level '{level}': {len(report.issues)} issues")
            except Exception as e:
                print(f"   Level '{level}': Failed - {str(e)}")
        
        # Test 3: Export functionality
        print("\n📊 Test 3: Export functionality")
        dummy_report = ValidationReport()
        dummy_report.add_issue(ValidationIssue(
            severity=ValidationSeverity.INFO,
            category="test",
            message="Test issue",
            suggestion="Test suggestion"
        ))
        
        try:
            export_validation_report(dummy_report, "test_report.json", "json")
            print("   JSON export: ✅")
        except Exception as e:
            print(f"   JSON export: ❌ {str(e)}")
        
        print("\n✅ Validation framework integration test completed!")
        
    except Exception as e:
        print(f"\n❌ Validation framework test failed: {str(e)}")

if __name__ == "__main__":
    test_validation_framework()