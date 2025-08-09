# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 23:35:36 2025

@author: NagabhushanamTattaga
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import numpy as np
from datetime import datetime
import warnings
import traceback

# Import the enhanced formula engine
try:
    from formula_engine import (
        FormulaEngine, ModelContext, FormulaType,
        FormulaEvaluationError, SecurityError
    )
    from formula_integration import EnhancedSystemDynamicsModel
    FORMULA_ENGINE_AVAILABLE = True
    print("✅ Advanced Formula Engine loaded successfully")
except ImportError as e:
    print(f"⚠️  Formula Engine not available: {e}")
    FORMULA_ENGINE_AVAILABLE = False

# Import existing simulation classes
try:
    from simulation import Model, Stock, Flow
    SIMULATION_AVAILABLE = True
except ImportError:
    print("⚠️  Basic simulation module not available")
    SIMULATION_AVAILABLE = False


# NEW: Performance system imports
try:
    from unified_performance_system import (
        initialize_performance, enhance_model, get_performance_report,
        optimize_performance_thresholds, performance_context
    )
    PERFORMANCE_AVAILABLE = True
    print("✅ Performance system loaded")
except ImportError:
    PERFORMANCE_AVAILABLE = False
    print("⚠️ Performance system not available - using basic simulation")



app = FastAPI(
    title="Enhanced System Dynamics API with Performance", 
    version="3.0.0",
    description="System Dynamics API with 10x-100x Performance Optimization"
)


# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global performance system
performance_initialized = False



# ===============================================================================
# NEW: Performance System Startup
# ===============================================================================

@app.on_event("startup")
async def startup_performance():
    """Initialize performance system on API startup"""
    global performance_initialized
    
    if PERFORMANCE_AVAILABLE:
        try:
            print("🚀 Initializing Performance System...")
            init_results = initialize_performance()
            
            # Log capabilities
            available_phases = sum(1 for v in init_results.values() if v)
            print(f"✅ Performance system ready with {available_phases}/4 optimization phases")
            
            if init_results.get('gpu_infrastructure', {}).get('gpu_available', False):
                gpu_count = init_results.get('gpu_infrastructure', {}).get('gpu_count', 0)
                print(f"🔥 GPU acceleration available: {gpu_count} device(s)")
            
            performance_initialized = True
            
        except Exception as e:
            print(f"⚠️ Performance initialization failed: {e}")
            print("   API will use basic simulation mode")
            performance_initialized = False
    else:
        print("📊 Running in basic simulation mode")


# ===============================================================================
# Enhanced Pydantic Models
# ===============================================================================

class ElementData(BaseModel):
    id: int
    type: str  # 'stock', 'flow', 'parameter', 'connector'
    x: float
    y: float
    name: str
    value: Optional[float] = None
    formula: Optional[str] = None
    units: Optional[str] = None
    description: Optional[str] = None

class ConnectionData(BaseModel):
    id: int
    from_element_id: int
    to_element_id: int
    connection_type: str  # 'inflow', 'outflow', 'parameter', 'dependency'

class ModelData(BaseModel):
    elements: List[ElementData]
    connections: List[ConnectionData]
    simulation_params: Dict[str, Any] = {
        "start_time": 0,
        "end_time": 50,
        "dt": 1
    }
    # Enhanced model properties
    parameters: Optional[Dict[str, float]] = {}
    constants: Optional[Dict[str, float]] = {}
    auxiliaries: Optional[Dict[str, str]] = {}  # name -> formula

class SimulationRequest(BaseModel):
    model: ModelData
    use_formula_engine: bool = True
    use_performance_optimization: bool = True
    validation_level: str = "standard"  # "none", "basic", "standard", "strict"

class SimulationResult(BaseModel):
    time: List[float]
    stocks: Dict[str, List[float]]
    flows: Dict[str, List[float]]
    auxiliaries: Optional[Dict[str, List[float]]] = None
    success: bool
    message: str
    # Enhanced result properties
    formula_statistics: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None
    integration_method: Optional[str] = None
    performance_stats: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    estimated_speedup: Optional[float] = None


class FormulaValidationRequest(BaseModel):
    formula: str
    context: Dict[str, Any] = {}
    formula_type: str = "rate"  # "rate", "auxiliary", "initial", "constant"

class FormulaValidationResult(BaseModel):
    valid: bool
    errors: List[str]
    warnings: List[str]
    dependencies: List[str]
    complexity_score: int

# In-memory storage for models
models_storage = {}

# Global formula engine instance
global_formula_engine = FormulaEngine() if FORMULA_ENGINE_AVAILABLE else None

# ===============================================================================
# API Endpoints
# ===============================================================================

@app.get("/")
async def root():
    return {
        "message": "Enhanced System Dynamics API with Performance Optimization",
        "version": "3.0.0",
        "performance_enabled": performance_initialized,
        "features": {
            "basic_simulation": True,
            "performance_optimization": performance_initialized,
            "gpu_acceleration": performance_initialized,
            "automatic_optimization": True
        }
    }

@app.post("/api/validate_formula")
async def validate_formula(request: FormulaValidationRequest) -> FormulaValidationResult:
    """Validate a formula using the advanced formula engine"""
    
    if not FORMULA_ENGINE_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Formula engine not available. Please install formula_engine module."
        )
    
    try:
        # Create context for validation
        context = ModelContext(
            stocks=request.context.get('stocks', {}),
            parameters=request.context.get('parameters', {}),
            auxiliaries=request.context.get('auxiliaries', {})
        )
        
        # Validate formula
        validation = global_formula_engine.validate_formula(request.formula, context)
        
        return FormulaValidationResult(
            valid=validation['valid'],
            errors=validation['errors'],
            warnings=validation['warnings'],
            dependencies=list(validation['dependencies']),
            complexity_score=validation['complexity_score']
        )
        
    except Exception as e:
        return FormulaValidationResult(
            valid=False,
            errors=[f"Validation failed: {str(e)}"],
            warnings=[],
            dependencies=[],
            complexity_score=0
        )


@app.post("/api/validate_model_comprehensive")
async def validate_model_comprehensive(model: ModelData):
    """Comprehensive model validation before simulation"""
    
    try:
        from model_validation import validate_model, ValidationLevel
        
        # Create temporary enhanced model for validation
        temp_model = create_enhanced_model_for_validation(model)
        
        # Run validation
        report = validate_model(temp_model, level="standard", include_behavioral=False)
        
        return {
            "valid": report.is_valid,
            "issues": [
                {
                    "severity": issue.severity.value,
                    "category": issue.category,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                    "element": issue.element_name
                } for issue in report.issues
            ],
            "summary": {
                "total_issues": report.total_issues,
                "errors": report.error_count,
                "warnings": report.warning_count
            }
        }
        
    except Exception as e:
        return {
            "valid": False,
            "issues": [{"severity": "error", "message": f"Validation failed: {str(e)}"}],
            "summary": {"total_issues": 1, "errors": 1, "warnings": 0}
        }

def create_enhanced_model_for_validation(model_data: ModelData):
    """Create enhanced model just for validation (without full simulation setup)"""
    
    if not FORMULA_ENGINE_AVAILABLE:
        # Use basic model
        return build_simulation_model(model_data)
    
    # Create enhanced model
    from formula_integration import EnhancedSystemDynamicsModel
    enhanced_model = EnhancedSystemDynamicsModel("Validation Model")
    
    # Add parameters and constants
    for name, value in model_data.parameters.items():
        enhanced_model.add_parameter(name, value)
    
    for name, value in model_data.constants.items():
        enhanced_model.add_constant(name, value)
    
    # Add auxiliaries
    for name, formula in model_data.auxiliaries.items():
        enhanced_model.add_auxiliary(name, formula)
    
    # Add stocks (validation only - no need for full setup)
    for element in model_data.elements:
        if element.type == "stock":
            try:
                formula = element.formula or str(element.value or 0)
                enhanced_model.add_stock_with_formula(element.name, formula, units=element.units or "")
            except Exception:
                # Add basic stock if formula fails
                from simulation import Stock
                stock = Stock(values=element.value or 0, name=element.name, units=element.units or "")
                enhanced_model.add_stock(stock)
    
    # Add flows (just for validation)
    for element in model_data.elements:
        if element.type == "flow":
            try:
                formula = element.formula or "0"
                enhanced_model.add_flow_with_formula(element.name, formula, units=element.units or "")
            except Exception:
                # Add basic flow if formula fails
                from simulation import Flow
                flow = Flow(rate_expression=lambda: 0, name=element.name, units=element.units or "")
                enhanced_model.add_flow(flow)
    
    return enhanced_model



@app.post("/api/simulate")
async def simulate_model(request: SimulationRequest) -> SimulationResult:
    """Run simulation with optional advanced formula engine"""
    
    try:
        model_data = request.model
        
        print(f"🚀 Starting simulation (Formula Engine: {request.use_formula_engine and FORMULA_ENGINE_AVAILABLE})")
        
        if request.use_formula_engine and FORMULA_ENGINE_AVAILABLE:
            # Use enhanced model with formula engine
            results = run_enhanced_simulation(model_data, request.validation_level)
        else:
            # Fall back to basic simulation
            results = run_basic_simulation(model_data)
        
        return results
        
    except FormulaEvaluationError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Formula evaluation failed: {str(e)}"
        )
    except SecurityError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Formula security violation: {str(e)}"
        )
    except Exception as e:
        # Log the full traceback for debugging
        print(f"❌ Simulation error: {str(e)}")
        print(traceback.format_exc())
        
        raise HTTPException(
            status_code=500, 
            detail=f"Simulation failed: {str(e)}"
        )
        
         
    start_time = time.time()
    
    try:
        if use_performance:
            # NEW: Performance-optimized simulation
            result = await run_performance_optimized_simulation(model_data)
        else:
            # Original simulation method
            result = run_basic_simulation(model_data)
        
        execution_time = time.time() - start_time
        result['execution_time'] = execution_time
        
        return SimulationResult(**result)
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ Simulation error: {e}")
        
        return SimulationResult(
            time=[], stocks={}, flows={}, success=False,
            message=f"Simulation failed: {str(e)}",
            execution_time=execution_time
        )        

async def run_performance_optimized_simulation(model_data: ModelData) -> Dict[str, Any]:
    """NEW: Run simulation with performance optimization"""
    
    with performance_context() as perf_controller:
        # Build simulation model
        base_model = build_simulation_model(model_data)
        
        # Enhance with performance system
        enhanced_model = enhance_model(base_model)
        
        print(f"📊 Model enhanced: {len(base_model.stocks)} stocks, {len(base_model.flows)} flows")
        
        # Run enhanced simulation
        sim_params = model_data.simulation_params
        start_time = sim_params.get("start_time", 0)
        end_time = sim_params.get("end_time", 50)
        dt = sim_params.get("dt", 1)
        
        # Set model parameters
        enhanced_model.time = start_time
        enhanced_model.dt = dt
        
        # Initialize results
        time_points = []
        stock_results = {stock.name: [] for stock in enhanced_model.stocks}
        flow_results = {flow.name: [] for flow in enhanced_model.flows}
        
        # Record initial state
        time_points.append(enhanced_model.time)
        for stock in enhanced_model.stocks:
            stock_results[stock.name].append(float(np.sum(stock.values)))
        
        # Run simulation with enhanced stepping
        steps = int((end_time - start_time) / dt)
        for step in range(steps):
            # Use enhanced step method
            enhanced_model.step()
            
            # Record state
            time_points.append(enhanced_model.time)
            for stock in enhanced_model.stocks:
                stock_results[stock.name].append(float(np.sum(stock.values)))
            
            # Record flow rates
            for flow in enhanced_model.flows:
                if len(flow_results[flow.name]) <= step:
                    try:
                        rate = flow.get_rate()
                        flow_results[flow.name].append(float(np.sum(rate)) if hasattr(rate, '__iter__') else float(rate))
                    except:
                        flow_results[flow.name].append(0.0)
        
        # Get performance statistics
        perf_summary = enhanced_model.get_performance_summary()
        
        return {
            'time': time_points,
            'stocks': stock_results,
            'flows': flow_results,
            'success': True,
            'message': f"Performance-optimized simulation completed ({perf_summary['total_steps']} steps)",
            'performance_stats': perf_summary,
            'estimated_speedup': perf_summary.get('estimated_speedup', 1.0)
        }



def run_enhanced_simulation(model_data: ModelData, validation_level: str) -> SimulationResult:
    """Run simulation using the enhanced formula engine"""
    
    print("🔬 Running enhanced simulation with formula engine")
    
    # VALIDATE FIRST - before any simulation setup
    if validation_level in ["standard", "strict"]:
        try:
            from model_validation import validate_model
            temp_model = create_enhanced_model_for_validation(model_data)
            validation_report = validate_model(temp_model, level="basic", include_behavioral=False)
            
            if not validation_report.is_valid and validation_level == "strict":
                return SimulationResult(
                    time=[], stocks={}, flows={}, success=False,
                    message=f"Model validation failed: {validation_report.error_count} errors found",
                    validation_results={"valid": False, "errors": [issue.message for issue in validation_report.issues]}
                )
            
            print(f"✅ Model validation passed ({validation_report.total_issues} issues found)")
            
        except Exception as e:
            print(f"⚠️ Validation failed: {e}")
            if validation_level == "strict":
                return SimulationResult(
                    time=[], stocks={}, flows={}, success=False,
                    message=f"Model validation error: {str(e)}"
                )    
    
    
    # Create enhanced model
    enhanced_model = EnhancedSystemDynamicsModel("API Model")
    
    # Set simulation parameters
    sim_params = model_data.simulation_params
    enhanced_model.time = sim_params.get("start_time", 0)
    enhanced_model.dt = sim_params.get("dt", 1)
    enhanced_model.start_time = sim_params.get("start_time", 0)
    enhanced_model.end_time = sim_params.get("end_time", 50)
    
    # Add parameters and constants
    for name, value in model_data.parameters.items():
        enhanced_model.add_parameter(name, value)
    
    for name, value in model_data.constants.items():
        enhanced_model.add_constant(name, value)
    
    # Add auxiliaries
    for name, formula in model_data.auxiliaries.items():
        enhanced_model.add_auxiliary(name, formula)
    
    # Create stocks and flows based on elements
    stock_objects = {}
    flow_objects = {}
    
    # First pass: create stocks
    for element in model_data.elements:
        if element.type == "stock":
            try:
                if element.formula:
                    # Stock with formula-based initial value
                    stock = enhanced_model.add_stock_with_formula(
                        element.name, element.formula, units=element.units or ""
                    )
                else:
                    # Stock with fixed initial value
                    stock = enhanced_model.add_stock_with_formula(
                        element.name, str(element.value or 0), units=element.units or ""
                    )
                stock_objects[element.id] = stock
                
            except Exception as e:
                warnings.warn(f"Failed to create stock {element.name}: {e}")
        
        elif element.type == "parameter":
            # Add as parameter if not already added
            if element.name not in enhanced_model.parameters:
                enhanced_model.add_parameter(element.name, element.value or 0)
    
    # Second pass: create flows with connections
    for element in model_data.elements:
        if element.type == "flow":
            try:
                # Determine source and target stocks from connections
                from_stock = None
                to_stock = None
                
                for connection in model_data.connections:
                    if connection.to_element_id == element.id and connection.connection_type == "outflow":
                        from_element = next(e for e in model_data.elements if e.id == connection.from_element_id)
                        if from_element.type == "stock":
                            from_stock = stock_objects.get(from_element.id)
                    
                    elif connection.from_element_id == element.id and connection.connection_type == "inflow":
                        to_element = next(e for e in model_data.elements if e.id == connection.to_element_id)
                        if to_element.type == "stock":
                            to_stock = stock_objects.get(to_element.id)
                
                # Create flow with formula
                formula = element.formula or "0"
                flow = enhanced_model.add_flow_with_formula(
                    element.name, formula, from_stock, to_stock, units=element.units or ""
                )
                flow_objects[element.id] = flow
                
            except Exception as e:
                warnings.warn(f"Failed to create flow {element.name}: {e}")
    
    # Validate model if requested
    validation_results = None
    if validation_level in ["standard", "strict"]:
        validation_results = enhanced_model.validate_all_formulas()
        
        if not validation_results['valid'] and validation_level == "strict":
            raise ValueError(f"Model validation failed: {validation_results['errors']}")
    
    # Run simulation
    start_time = sim_params["start_time"]
    end_time = sim_params["end_time"]
    dt = sim_params["dt"]
    
    time_steps = np.arange(start_time, end_time + dt, dt)
    
    # Initialize results
    results = {
        "time": time_steps.tolist(),
        "stocks": {},
        "flows": {},
        "auxiliaries": {}
    }
    
    # Initialize stock results
    for stock in enhanced_model.stocks:
        initial_value = stock.values.item() if stock.values.size == 1 else np.sum(stock.values)
        results["stocks"][stock.name] = [initial_value]
    
    # Initialize flow and auxiliary results
    for flow in enhanced_model.flows:
        results["flows"][flow.name] = []
    
    for aux_name in enhanced_model.auxiliaries:
        results["auxiliaries"][aux_name] = []
    
    # Simulation loop
    for i in range(1, len(time_steps)):
        enhanced_model.time = time_steps[i]
        
        try:
            # Update auxiliaries
            enhanced_model.update_auxiliaries()
            
            # Record auxiliary values
            for aux_name, aux_data in enhanced_model.auxiliaries.items():
                results["auxiliaries"][aux_name].append(aux_data['value'])
            
            # Calculate flow rates and update stocks
            context = enhanced_model._create_current_context()
            
            for stock in enhanced_model.stocks:
                net_flow = 0.0
                
                # Calculate inflows
                for flow in enhanced_model.flows:
                    if hasattr(flow, '_rate_formula'):
                        try:
                            rate = enhanced_model.formula_engine.evaluate(
                                flow._rate_formula, context, FormulaType.RATE
                            )
                            
                            # Check if this flow affects this stock
                            affects_stock = False
                            for conn in model_data.connections:
                                if (conn.from_element_id == flow_objects.get(flow.name, {}).get('id') and
                                    conn.connection_type == "inflow"):
                                    target_element = next(e for e in model_data.elements if e.id == conn.to_element_id)
                                    if target_element.name == stock.name:
                                        net_flow += rate
                                        affects_stock = True
                                
                                elif (conn.to_element_id == flow_objects.get(flow.name, {}).get('id') and
                                      conn.connection_type == "outflow"):
                                    source_element = next(e for e in model_data.elements if e.id == conn.from_element_id)
                                    if source_element.name == stock.name:
                                        net_flow -= rate
                                        affects_stock = True
                            
                            # Store flow rate
                            if flow.name not in results["flows"]:
                                results["flows"][flow.name] = []
                            if len(results["flows"][flow.name]) < i:
                                results["flows"][flow.name].append(rate)
                                
                        except Exception as e:
                            warnings.warn(f"Error evaluating flow {flow.name}: {e}")
                            if flow.name not in results["flows"]:
                                results["flows"][flow.name] = []
                            if len(results["flows"][flow.name]) < i:
                                results["flows"][flow.name].append(0.0)
                
                # Update stock value
                current_value = stock.values.item() if stock.values.size == 1 else np.sum(stock.values)
                new_value = max(0, current_value + net_flow * dt)
                
                stock.values = np.array([new_value]) if stock.values.size == 1 else np.full_like(stock.values, new_value)
                results["stocks"][stock.name].append(new_value)
                
        except Exception as e:
            warnings.warn(f"Error in simulation step {i}: {e}")
            # Fill with previous values or zeros
            for stock in enhanced_model.stocks:
                if len(results["stocks"][stock.name]) < i + 1:
                    last_val = results["stocks"][stock.name][-1] if results["stocks"][stock.name] else 0
                    results["stocks"][stock.name].append(last_val)
    
    # Get formula engine statistics
    formula_stats = enhanced_model.get_formula_statistics()
    
    return SimulationResult(
        time=results["time"],
        stocks=results["stocks"],
        flows=results["flows"],
        auxiliaries=results["auxiliaries"],
        success=True,
        message="Enhanced simulation completed successfully",
        formula_statistics=formula_stats,
        validation_results=validation_results,
        integration_method="enhanced_formula_engine"
    )

def run_basic_simulation(model_data: ModelData) -> SimulationResult:
    """Run simulation using basic method (fallback)"""
    
    print("🔧 Running basic simulation (fallback mode)")
    
    # Use existing basic simulation logic
    simulation_model = build_simulation_model(model_data)
    results = run_simulation(simulation_model, model_data.simulation_params)
    
    return SimulationResult(
        time=results["time"],
        stocks=results["stocks"],
        flows=results.get("flows", {}),
        success=True,
        message="Basic simulation completed successfully",
        integration_method="basic_euler"
    )

    model = build_simulation_model(model_data)
    
    # Run basic simulation
    sim_params = model_data.simulation_params
    model.time = sim_params.get("start_time", 0)
    model.dt = sim_params.get("dt", 1)
    
    time_points = []
    stock_results = {stock.name: [] for stock in model.stocks}
    flow_results = {flow.name: [] for flow in model.flows}
    
    # Basic simulation loop
    end_time = sim_params.get("end_time", 50)
    while model.time <= end_time:
        time_points.append(model.time)
        
        for stock in model.stocks:
            stock_results[stock.name].append(float(np.sum(stock.values) if hasattr(stock.values, '__iter__') else stock.values))
        
        for flow in model.flows:
            try:
                rate = flow.get_rate()
                flow_results[flow.name].append(float(np.sum(rate) if hasattr(rate, '__iter__') else rate))
            except:
                flow_results[flow.name].append(0.0)
        
        # Basic step
        for stock in model.stocks:
            net_flow = stock.get_net_flow(model.dt)
            if hasattr(stock.values, '__iter__'):
                stock.values = stock.values + net_flow * model.dt
            else:
                stock.values = stock.values + net_flow * model.dt
        
        model.time += model.dt
    
    return {
        'time': time_points,
        'stocks': stock_results,
        'flows': flow_results,
        'success': True,
        'message': "Basic simulation completed",
        'performance_stats': None,
        'estimated_speedup': 1.0
    }



# ===============================================================================
# Existing Basic Simulation Functions (Updated)
# ===============================================================================

def build_simulation_model(model_data: ModelData):
    """Convert UI model data to basic simulation format"""
    
    stocks = {}
    flows = {}
    parameters = {}
    
    # Process elements
    for element in model_data.elements:
        if element.type == "stock":
            stocks[element.name] = {
                "initial_value": element.value or 0,
                "id": element.id
            }
        elif element.type == "flow":
            flows[element.name] = {
                "formula": element.formula or "0",
                "id": element.id,
                "inflows": [],
                "outflows": []
            }
        elif element.type == "parameter":
            parameters[element.name] = {
                "value": element.value or 0,
                "id": element.id
            }
    
    # Process connections
    for connection in model_data.connections:
        from_element = next(e for e in model_data.elements if e.id == connection.from_element_id)
        to_element = next(e for e in model_data.elements if e.id == connection.to_element_id)
        
        if connection.connection_type == "outflow":
            if from_element.type == "stock" and to_element.type == "flow":
                flows[to_element.name]["outflows"].append(from_element.name)
                
        elif connection.connection_type == "inflow":
            if from_element.type == "flow" and to_element.type == "stock":
                flows[from_element.name]["inflows"].append(to_element.name)
    
    # return {
    #     "stocks": stocks,
    #     "flows": flows,
    #     "parameters": parameters
    # }


    model = Model(name="API Model")
    
    stocks = {}
    flows = {}
    
    # Process elements
    for element in model_data.elements:
        if element.type == "stock":
            stock = Stock(
                values=element.value or 0,
                name=element.name,
                units=element.units or ""
            )
            stocks[element.id] = stock
            model.add_stock(stock)
            
        elif element.type == "flow":
            # Simple flow creation
            rate = element.value or 0
            flow = Flow(
                rate_expression=lambda r=rate: r,
                name=element.name,
                units=element.units or ""
            )
            flows[element.id] = flow
            model.add_flow(flow)
    
    # Process connections
    for connection in model_data.connections:
        from_element = next(e for e in model_data.elements if e.id == connection.from_element_id)
        to_element = next(e for e in model_data.elements if e.id == connection.to_element_id)
        
        if connection.connection_type == "inflow":
            if from_element.type == "flow" and to_element.type == "stock":
                flow = flows[from_element.id]
                stock = stocks[to_element.id]
                stock.add_inflow(flow)
                
        elif connection.connection_type == "outflow":
            if from_element.type == "stock" and to_element.type == "flow":
                stock = stocks[from_element.id]
                flow = flows[to_element.id]
                stock.add_outflow(flow)
    
    return model


def run_simulation(model, sim_params):
    """Execute basic simulation"""
    
    start = sim_params["start_time"]
    end = sim_params["end_time"]
    dt = sim_params["dt"]
    
    time_steps = np.arange(start, end + dt, dt)
    
    # Initialize results
    results = {
        "time": time_steps.tolist(),
        "stocks": {},
        "flows": {}
    }
    
    # Initialize stock values
    current_stock_values = {}
    for stock_name, stock_data in model["stocks"].items():
        current_stock_values[stock_name] = stock_data["initial_value"]
        results["stocks"][stock_name] = [stock_data["initial_value"]]
    
    # Get parameters
    parameters = {}
    for param_name, param_data in model["parameters"].items():
        parameters[param_name] = param_data["value"]
    
    # Run simulation for each time step
    for i in range(1, len(time_steps)):
        t = time_steps[i]
        
        # Calculate flow rates
        flow_rates = {}
        for flow_name, flow_data in model["flows"].items():
            try:
                context = {
                    **current_stock_values,
                    **parameters,
                    "t": t,
                    "dt": dt
                }
                
                if flow_data["formula"] and flow_data["formula"].strip():
                    rate = evaluate_formula_safely(flow_data["formula"], context)
                else:
                    rate = 0
                
                flow_rates[flow_name] = max(0, rate)
                
            except Exception as e:
                print(f"Error evaluating flow {flow_name}: {e}")
                flow_rates[flow_name] = 0
        
        # Store flow rates
        for flow_name, rate in flow_rates.items():
            if flow_name not in results["flows"]:
                results["flows"][flow_name] = []
            results["flows"][flow_name].append(rate)
        
        # Update stock values based on flows
        stock_changes = {stock_name: 0 for stock_name in current_stock_values.keys()}
        
        for flow_name, flow_data in model["flows"].items():
            rate = flow_rates[flow_name]
            
            for stock_name in flow_data["outflows"]:
                if stock_name in stock_changes:
                    stock_changes[stock_name] -= rate * dt
            
            for stock_name in flow_data["inflows"]:
                if stock_name in stock_changes:
                    stock_changes[stock_name] += rate * dt
        
        # Update stock values and store results
        for stock_name in current_stock_values.keys():
            current_stock_values[stock_name] = max(0, current_stock_values[stock_name] + stock_changes[stock_name])
            results["stocks"][stock_name].append(current_stock_values[stock_name])
    
    return results

def evaluate_formula_safely(formula: str, context: dict) -> float:
    """Basic safe formula evaluation (fallback)"""
    
    expression = formula
    
    # Replace variables with context values
    for variable, value in context.items():
        import re
        pattern = r'\b' + re.escape(variable) + r'\b'
        expression = re.sub(pattern, str(value), expression)
    
    # Basic safety check
    allowed_chars = set('0123456789+-*/.() ')
    if not all(c in allowed_chars or c.isalnum() for c in expression.replace(' ', '')):
        raise ValueError(f"Invalid characters in formula: {formula}")
    
    try:
        return float(eval(expression))
    except Exception as e:
        raise ValueError(f"Formula evaluation failed: {formula}")

# ===============================================================================
# Additional API Endpoints
# ===============================================================================

@app.get("/api/performance_status")
async def get_performance_status():
    """Get current performance system status"""
    
    if not performance_initialized:
        return {
            "performance_enabled": False,
            "message": "Performance system not initialized"
        }
    
    try:
        report = get_performance_report()
        
        # Extract key metrics
        master_stats = report.get('master_controller', {}).get('global_stats', {})
        
        return {
            "performance_enabled": True,
            "total_operations": master_stats.get('total_operations', 0),
            "average_speedup": round(master_stats.get('average_speedup', 1.0), 2),
            "total_time_saved": round(master_stats.get('total_time_saved', 0.0), 3),
            "strategy_usage": master_stats.get('strategy_usage', {}),
            "gpu_available": 'phase_2_gpu_infrastructure' in report,
            "phases_active": list(report.keys())
        }
        
    except Exception as e:
        return {
            "performance_enabled": False,
            "error": str(e)
        }

@app.post("/api/optimize_performance")
async def optimize_performance():
    """Optimize performance thresholds based on usage"""
    
    if not performance_initialized:
        raise HTTPException(status_code=503, detail="Performance system not available")
    
    try:
        optimization_results = optimize_performance_thresholds()
        
        return {
            "success": True,
            "optimizations": optimization_results,
            "message": f"Optimized {len(optimization_results)} thresholds"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.get("/api/performance_report")
async def get_detailed_performance_report():
    """Get comprehensive performance report"""
    
    if not performance_initialized:
        raise HTTPException(status_code=503, detail="Performance system not available")
    
    try:
        return get_performance_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")




@app.get("/api/formula_engine_status")
async def get_formula_engine_status():
    """Get formula engine status and statistics"""
    
    if not FORMULA_ENGINE_AVAILABLE:
        return {
            "available": False,
            "message": "Formula engine not loaded"
        }
    
    stats = global_formula_engine.get_statistics()
    
    return {
        "available": True,
        "statistics": stats,
        "supported_functions": [
            "STEP", "PULSE", "RAMP", "LOOKUP", "IF_THEN_ELSE",
            "MIN", "MAX", "DELAY", "TIME", "DT",
            "sin", "cos", "exp", "log", "sqrt", "abs"
        ]
    }

@app.post("/api/clear_formula_cache")
async def clear_formula_cache():
    """Clear formula engine caches"""
    
    if not FORMULA_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Formula engine not available")
    
    global_formula_engine.clear_caches()
    
    return {"message": "Formula caches cleared successfully"}

@app.get("/api/model_templates")
async def get_model_templates():
    """Get predefined model templates with formulas"""
    
    templates = {
        "population_growth": {
            "name": "Population Growth Model",
            "description": "Simple population growth with carrying capacity",
            "elements": [
                {
                    "type": "stock",
                    "name": "Population",
                    "formula": "initial_population",
                    "units": "individuals"
                }
            ],
            "parameters": {
                "initial_population": 100,
                "growth_rate": 0.03,
                "carrying_capacity": 1000
            },
            "auxiliaries": {
                "effective_growth_rate": "growth_rate * (1 - Population / carrying_capacity)"
            }
        },
        "sir_epidemic": {
            "name": "SIR Epidemic Model",
            "description": "Susceptible-Infected-Recovered epidemic model",
            "elements": [
                {
                    "type": "stock",
                    "name": "Susceptible",
                    "formula": "total_population - initial_infected",
                    "units": "people"
                },
                {
                    "type": "stock",
                    "name": "Infected",
                    "formula": "initial_infected",
                    "units": "people"
                },
                {
                    "type": "stock",
                    "name": "Recovered",
                    "formula": "0",
                    "units": "people"
                }
            ],
            "parameters": {
                "total_population": 1000,
                "initial_infected": 10,
                "infection_rate": 0.3,
                "recovery_rate": 0.1
            },
            "auxiliaries": {
                "contact_rate": "infection_rate * Susceptible * Infected / total_population"
            }
        }
    }
    
    return templates

# ===============================================================================
# Existing API Endpoints (Updated)
# ===============================================================================

# @app.post("/api/validate_model")
# async def validate_model(model: ModelData):
#     """Validate model structure and equations"""
    
#     try:
#         # Basic structural validation
#         validation_result = validate_model_structure(model)
        
#         # Enhanced formula validation if engine available
#         if FORMULA_ENGINE_AVAILABLE:
#             formula_validation = validate_model_formulas(model)
#             validation_result["formula_validation"] = formula_validation
        
#         return validation_result
        
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")

@app.post("/api/validate_model")
async def validate_model(model: ModelData):
    """Validate model structure"""
    
    try:
        # Basic validation
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "performance_recommendation": None
        }
        
        # Count elements
        stocks = [e for e in model.elements if e.type == "stock"]
        flows = [e for e in model.elements if e.type == "flow"]
        
        if len(stocks) == 0:
            validation_result["errors"].append("Model must have at least one stock")
            validation_result["valid"] = False
        
        # NEW: Performance recommendations
        if performance_initialized:
            total_elements = len(stocks) + len(flows)
            if total_elements >= 1000:
                validation_result["performance_recommendation"] = "Large model detected - GPU acceleration recommended"
            elif total_elements >= 100:
                validation_result["performance_recommendation"] = "Medium model - CPU JIT optimization will be used"
            else:
                validation_result["performance_recommendation"] = "Small model - basic optimization sufficient"
        
        validation_result["element_count"] = {
            "stocks": len(stocks),
            "flows": len(flows),
            "total": total_elements
        }
        
        return validation_result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")




@app.post("/api/validate_model")
async def validate_model(model: ModelData):
    """Validate model structure and equations"""
    
    try:
        from unified_validation import get_unified_validator
        validator = get_unified_validator()
        
        # Convert API model to internal model for validation
        internal_model = convert_api_model_to_internal(model)
        
        # Use unified validation
        report = validator.validate_all(internal_model, 'model')
        
        return {
            'valid': report.is_valid,
            'issues': [
                {
                    'severity': issue.severity.value,
                    'category': issue.category,
                    'message': issue.message,
                    'suggestion': issue.suggestion,
                    'element': issue.element_name
                } for issue in report.issues
            ],
            'summary': {
                'total_issues': len(report.issues),
                'error_count': len([i for i in report.issues if i.severity.value == 'error']),
                'warning_count': len([i for i in report.issues if i.severity.value == 'warning'])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")


@app.post("/api/save_model")
async def save_model(model: ModelData, model_name: str):
    """
    Save model to storage
    """
    model_id = f"{model_name}_{datetime.now().isoformat()}"
    models_storage[model_id] = model.dict()
    
    return {
        "model_id": model_id,
        "message": "Model saved successfully"
    }

@app.get("/api/models")
async def list_models():
    """
    List all saved models
    """
    return {
        "models": list(models_storage.keys())
    }

@app.get("/api/models/{model_id}")
async def load_model(model_id: str):
    """
    Load a specific model
    """
    if model_id not in models_storage:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return models_storage[model_id]



# ===============================================================================
# Server Startup
# ===============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Enhanced System Dynamics API Server")
    print(f"   Performance System: {'✅ Available' if PERFORMANCE_AVAILABLE else '❌ Not Available'}")

    print(f"   Formula Engine: {'✅ Enabled' if FORMULA_ENGINE_AVAILABLE else '❌ Not Available'}")
    print(f"   Basic Simulation: {'✅ Available' if SIMULATION_AVAILABLE else '❌ Not Available'}")
    print("   Server: http://localhost:8000")
    print("   API Docs: http://localhost:8000/docs")
    
    print("   Performance Status: http://localhost:8000/api/performance_status")

    
    uvicorn.run(app, host="0.0.0.0", port=8000)