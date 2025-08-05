from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import numpy as np
from datetime import datetime

# Import the original SimSystemDynamic library
from simulation import Model, Stock, Flow  # Uncomment when library is available

app = FastAPI(title="System Dynamics API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class ElementData(BaseModel):
    id: int
    type: str  # 'stock', 'flow', 'parameter', 'connector'
    x: float
    y: float
    name: str
    value: Optional[float] = None
    formula: Optional[str] = None

class ConnectionData(BaseModel):
    id: int
    from_element_id: int
    to_element_id: int
    connection_type: str  # 'flow', 'dependency'

class ModelData(BaseModel):
    elements: List[ElementData]
    connections: List[ConnectionData]
    simulation_params: Dict[str, Any] = {
        "start_time": 0,
        "end_time": 50,
        "dt": 1
    }

class SimulationRequest(BaseModel):
    model: ModelData

class SimulationResult(BaseModel):
    time: List[float]
    stocks: Dict[str, List[float]]
    flows: Dict[str, List[float]]
    success: bool
    message: str

# In-memory storage for models (use database in production)
models_storage = {}

@app.get("/")
async def root():
    return {"message": "System Dynamics API Server"}

@app.post("/api/simulate")
async def simulate_model(request: SimulationRequest) -> SimulationResult:
    """
    Run simulation on the provided model
    """
    try:
        model_data = request.model
        
        # Convert UI model to SimSystemDynamic format
        simulation_model = build_simulation_model(model_data)
        
        # Run the simulation
        results = run_simulation(simulation_model, model_data.simulation_params)
        
        return SimulationResult(
            time=results["time"],
            stocks=results["stocks"],
            flows=results.get("flows", {}),
            success=True,
            message="Simulation completed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Simulation failed: {str(e)}")

@app.post("/api/validate_model")
async def validate_model(model: ModelData):
    """
    Validate model structure and equations
    """
    try:
        validation_result = validate_model_structure(model)
        return validation_result
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

# def build_simulation_model(model_data: ModelData):
#     """
#     Convert UI model data to SimSystemDynamic model
#     """
#     # This would integrate with the original library
#     # For now, we'll create a mock structure
    
#     stocks = {}
#     flows = {}
#     parameters = {}
    
#     # Process elements
#     for element in model_data.elements:
#         if element.type == "stock":
#             stocks[element.name] = {
#                 "initial_value": element.value or 0,
#                 "id": element.id
#             }
#         elif element.type == "flow":
#             flows[element.name] = {
#                 "formula": element.formula or "0",
#                 "id": element.id
#             }
#         elif element.type == "parameter":
#             parameters[element.name] = {
#                 "value": element.value or 0,
#                 "id": element.id
#             }
    
#     # Process connections to determine flow directions
#     for connection in model_data.connections:
#         from_element = next(e for e in model_data.elements if e.id == connection.from_element_id)
#         to_element = next(e for e in model_data.elements if e.id == connection.to_element_id)
        
#         if from_element.type == "stock" and to_element.type == "flow":
#             # Stock to flow connection (outflow)
#             flows[to_element.name]["from_stock"] = from_element.name
#         elif from_element.type == "flow" and to_element.type == "stock":
#             # Flow to stock connection (inflow)
#             flows[from_element.name]["to_stock"] = to_element.name
    
#     return {
#         "stocks": stocks,
#         "flows": flows,
#         "parameters": parameters
#     }

# def run_simulation(model, sim_params):
#     """
#     Execute the simulation using SimSystemDynamic
#     """
#     try:
#         # This is where you'd integrate with the actual SimSystemDynamic library
#         # For demonstration, we'll create mock results
        
#         start = sim_params["start_time"]
#         end = sim_params["end_time"]
#         dt = sim_params["dt"]
        
#         time_steps = np.arange(start, end + dt, dt)
        
#         # Mock simulation results
#         results = {
#             "time": time_steps.tolist(),
#             "stocks": {},
#             "flows": {}
#         }
        
#         # Generate mock data for each stock
#         for stock_name, stock_data in model["stocks"].items():
#             initial_value = stock_data["initial_value"]
            
#             # Simple exponential growth model for demonstration
#             values = []
#             current_value = initial_value
            
#             for t in time_steps:
#                 # Apply some growth/decay based on connected flows
#                 growth_rate = 0.02  # 2% per time step
#                 current_value = current_value * (1 + growth_rate)
#                 values.append(current_value)
            
#             results["stocks"][stock_name] = values
        
#         # Generate flow data
#         for flow_name, flow_data in model["flows"].items():
#             # Calculate flow rates based on formula
#             flow_values = []
#             for t in time_steps:
#                 # Simple rate calculation for demonstration
#                 rate = 5 + np.sin(t * 0.1) * 2  # Oscillating flow
#                 flow_values.append(rate)
            
#             results["flows"][flow_name] = flow_values
        
#         return results
        
#     except Exception as e:
#         raise Exception(f"Simulation execution failed: {str(e)}")

def build_simulation_model(model_data: ModelData):
    """
    Convert UI model data to SimSystemDynamic model with proper connection handling
    """
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
                "inflows": [],   # Stocks this flow adds to
                "outflows": []   # Stocks this flow removes from
            }
        elif element.type == "parameter":
            parameters[element.name] = {
                "value": element.value or 0,
                "id": element.id
            }
    
    # Process connections to determine flow directions
    for connection in model_data.connections:
        from_element = next(e for e in model_data.elements if e.id == connection.from_element_id)
        to_element = next(e for e in model_data.elements if e.id == connection.to_element_id)
        
        if connection.connection_type == "outflow":
            # Stock → Flow (flow removes from stock)
            if from_element.type == "stock" and to_element.type == "flow":
                flows[to_element.name]["outflows"].append(from_element.name)
                
        elif connection.connection_type == "inflow":
            # Flow → Stock (flow adds to stock)
            if from_element.type == "flow" and to_element.type == "stock":
                flows[from_element.name]["inflows"].append(to_element.name)
    
    return {
        "stocks": stocks,
        "flows": flows,
        "parameters": parameters
    }

def run_simulation(model, sim_params):
    """
    Execute the simulation with proper inflow/outflow handling
    """
    try:
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
                    # Create evaluation context
                    context = {
                        **current_stock_values,
                        **parameters,
                        "t": t,
                        "dt": dt
                    }
                    
                    # Evaluate flow formula
                    if flow_data["formula"] and flow_data["formula"].strip():
                        rate = evaluate_formula_safely(flow_data["formula"], context)
                    else:
                        rate = 0
                    
                    flow_rates[flow_name] = max(0, rate)  # Ensure non-negative flow rates
                    
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
                
                # Apply outflows (subtract from stocks)
                for stock_name in flow_data["outflows"]:
                    if stock_name in stock_changes:
                        stock_changes[stock_name] -= rate * dt
                
                # Apply inflows (add to stocks)
                for stock_name in flow_data["inflows"]:
                    if stock_name in stock_changes:
                        stock_changes[stock_name] += rate * dt
            
            # Update stock values and store results
            for stock_name in current_stock_values.keys():
                current_stock_values[stock_name] = max(0, current_stock_values[stock_name] + stock_changes[stock_name])
                results["stocks"][stock_name].append(current_stock_values[stock_name])
        
        return results
        
    except Exception as e:
        raise Exception(f"Simulation execution failed: {str(e)}")

def evaluate_formula_safely(formula: str, context: dict) -> float:
    """
    Safely evaluate mathematical formulas with stock and parameter values
    """
    # Replace variable names with their values
    expression = formula
    
    # Replace variables with context values
    for variable, value in context.items():
        # Use word boundaries to avoid partial matches
        import re
        pattern = r'\b' + re.escape(variable) + r'\b'
        expression = re.sub(pattern, str(value), expression)
    
    # Basic safety check
    allowed_chars = set('0123456789+-*/.() ')
    if not all(c in allowed_chars for c in expression.replace(' ', '')):
        raise ValueError(f"Invalid characters in formula: {formula}")
    
    try:
        return float(eval(expression))
    except Exception as e:
        raise ValueError(f"Formula evaluation failed: {formula} -> {expression}")

def validate_model_structure(model: ModelData):
    """
    Validate the model structure for common issues
    """
    errors = []
    warnings = []
    
    # Check for stocks without initial values
    for element in model.elements:
        if element.type == "stock" and (element.value is None or element.value == 0):
            warnings.append(f"Stock '{element.name}' has no initial value")
        
        if element.type == "flow" and not element.formula:
            errors.append(f"Flow '{element.name}' has no rate formula")
    
    # Check for disconnected elements
    connected_elements = set()
    for connection in model.connections:
        connected_elements.add(connection.from_element_id)
        connected_elements.add(connection.to_element_id)
    
    for element in model.elements:
        if element.id not in connected_elements and element.type != "parameter":
            warnings.append(f"Element '{element.name}' is not connected to other elements")
    
    # Check for circular dependencies (basic check)
    # More sophisticated analysis would be needed for complex models
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

def parse_formula(formula: str, available_vars: Dict[str, float]) -> float:
    """
    Safely parse and evaluate mathematical formulas
    """
    # This would need proper expression parsing and security measures
    # For now, a simple eval with restricted namespace
    
    # Create safe namespace with mathematical functions
    safe_dict = {
        "__builtins__": {},
        "abs": abs,
        "min": min,
        "max": max,
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        **available_vars  # Add model variables
    }
    
    try:
        return eval(formula, safe_dict)
    except Exception as e:
        raise ValueError(f"Invalid formula: {formula}. Error: {str(e)}")

# Example integration with original SimSystemDynamic library
def create_real_simulation(model_data: ModelData):
    """
    Example of how to integrate with the actual SimSystemDynamic library
    """
    # Uncomment and modify when you have the library available
    
    # from simulation import Model, Stock, Flow
    
    # # Create model instance
    # sim_params = model_data.simulation_params
    # model = Model(
    #     start=sim_params["start_time"],
    #     end=sim_params["end_time"],
    #     dt=sim_params["dt"]
    # )
    
    # # Create stocks
    # stocks_dict = {}
    # for element in model_data.elements:
    #     if element.type == "stock":
    #         stock = Stock(name=element.name, initial_value=element.value)
    #         stocks_dict[element.name] = stock
    #         model.add_stock(stock)
    
    # # Create flows with rate functions
    # for element in model_data.elements:
    #     if element.type == "flow":
    #         # Parse connections to determine from/to stocks
    #         from_stock = None
    #         to_stock = None
    #         
    #         for connection in model_data.connections:
    #             if connection.to_element_id == element.id:
    #                 from_element = next(e for e in model_data.elements if e.id == connection.from_element_id)
    #                 if from_element.type == "stock":
    #                     from_stock = stocks_dict.get(from_element.name)
    #             elif connection.from_element_id == element.id:
    #                 to_element = next(e for e in model_data.elements if e.id == connection.to_element_id)
    #                 if to_element.type == "stock":
    #                     to_stock = stocks_dict.get(to_element.name)
    #         
    #         # Create rate function from formula
    #         def create_rate_func(formula):
    #             def rate_func(t, state):
    #                 return parse_formula(formula, state)
    #             return rate_func
    #         
    #         flow = Flow(
    #             name=element.name,
    #             from_stock=from_stock,
    #             to_stock=to_stock,
    #             rate_func=create_rate_func(element.formula)
    #         )
    #         model.add_flow(flow)
    
    # # Run simulation
    # results = model.run()
    # return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)