# # Usage Example - AnyLogic Style System Dynamics
# """
# This example shows how to use the sim_SD module 
# with AnyLogic-style syntax for stock and flow modeling
# """

# # Import the system dynamics components
# from sim_SD import stock, flow, model

# def example_1_simple_tanks():
#     """Example 1: Simple tank system like AnyLogic"""
#     print("=" * 50)
#     print("EXAMPLE 1: SIMPLE TANK SYSTEM")
#     print("=" * 50)
    
#     # Create stocks (AnyLogic style)
#     stock_1 = stock(values=100.0, name="Tank_1")
#     stock_2 = stock(values=0.0, name="Tank_2")
    
#     # Create flow with proportional rate (like AnyLogic)
#     flow_1 = stock_1 * 0.02  # 2% of stock_1 per time unit
#     flow_1.name = "Transfer_Flow"
    
#     # Connect stocks and flows (AnyLogic style)
#     stock_1.add_outflow(flow_1)
#     stock_2.add_inflow(flow_1)
    
#     # Alternative syntax (more Pythonic)
#     # flow_1.connect(from_stock=stock_1, to_stock=stock_2)
    
#     # Create and run model
#     sim_model = model(dt=1.0)
#     sim_model.add_stock(stock_1)
#     sim_model.add_stock(stock_2)
#     sim_model.add_flow(flow_1)
    
#     # Run simulation
#     sim_model.run(100)  # 100 time units
    
#     # Plot results
#     sim_model.plot()
    
#     return sim_model

# def example_2_constant_flow():
#     """Example 2: Constant flow rate"""
#     print("=" * 50)
#     print("EXAMPLE 2: CONSTANT FLOW RATE")
#     print("=" * 50)
    
#     # Create stocks
#     source = stock(values=1000.0, name="Source")
#     sink = stock(values=0.0, name="Sink")
    
#     # Create constant flow (2 units per month = 2/30 per day)
#     constant_flow = flow(rate=2.0/30.0, name="Constant_Transfer")
    
#     # Connect
#     source.add_outflow(constant_flow)
#     sink.add_inflow(constant_flow)
    
#     # Create and run model
#     sim_model = model(dt=1.0)  # 1 day time step
#     sim_model.add_stock(source)
#     sim_model.add_stock(sink)
#     sim_model.add_flow(constant_flow)
    
#     # Run for 3 months (90 days)
#     sim_model.run(90)
    
#     # Plot results
#     sim_model.plot()
    
#     return sim_model

# def example_3_multi_dimensional():
#     """Example 3: Multi-dimensional stocks (age groups)"""
#     print("=" * 50)
#     print("EXAMPLE 3: MULTI-DIMENSIONAL STOCKS")
#     print("=" * 50)
    
#     # Create 1D stocks for population by age groups [young, middle, old]
#     population = stock(dim=3, values=[1000, 800, 600], name="Population")
#     deceased = stock(dim=3, values=[0, 0, 0], name="Deceased")
    
#     # Create mortality flow with different rates for each age group
#     mortality_rates = [0.001, 0.005, 0.02]  # Daily death rates
#     mortality_flow = flow(rate=lambda: population.values * mortality_rates, name="Mortality")
    
#     # Connect
#     population.add_outflow(mortality_flow)
#     deceased.add_inflow(mortality_flow)
    
#     # Create and run model
#     sim_model = model(dt=1.0)
#     sim_model.add_stock(population)
#     sim_model.add_stock(deceased)
#     sim_model.add_flow(mortality_flow)
    
#     # Run for 1 year
#     sim_model.run(365)
    
#     # Plot results
#     sim_model.plot()
    
#     return sim_model

# def example_4_complex_system():
#     """Example 4: Complex system with multiple flows"""
#     print("=" * 50)
#     print("EXAMPLE 4: COMPLEX MULTI-FLOW SYSTEM")
#     print("=" * 50)
    
#     # Create stocks for supply chain
#     raw_materials = stock(values=500.0, name="Raw_Materials")
#     work_in_progress = stock(values=50.0, name="WIP") 
#     finished_goods = stock(values=100.0, name="Finished_Goods")
    
#     # Create flows
#     material_supply = flow(rate=15.0, name="Material_Supply")  # Constant supply
#     production = flow(rate=lambda: min(10.0, raw_materials.values/2.0), name="Production")
#     shipping = finished_goods * 0.1  # 10% of finished goods shipped daily
#     shipping.name = "Shipping"
    
#     # Connect flows - AnyLogic style connections
#     # Material supply goes to raw materials
#     raw_materials.add_inflow(material_supply)
    
#     # Production: raw materials -> WIP (consumes 2x raw materials)
#     raw_materials.add_outflow(flow(rate=lambda: production.get_rate() * 2.0, name="Material_Consumption"))
#     work_in_progress.add_inflow(production)
    
#     # Finishing: WIP -> Finished goods
#     finishing = work_in_progress * 0.2  # 20% of WIP finished daily
#     finishing.name = "Finishing"
#     work_in_progress.add_outflow(finishing)
#     finished_goods.add_inflow(finishing)
    
#     # Shipping: finished goods -> external
#     finished_goods.add_outflow(shipping)
    
#     # Create and run model
#     sim_model = model(dt=1.0)
    
#     # Add all components
#     for stock_item in [raw_materials, work_in_progress, finished_goods]:
#         sim_model.add_stock(stock_item)
    
#     for flow_item in [material_supply, production, finishing, shipping]:
#         sim_model.add_flow(flow_item)
    
#     # Run simulation
#     sim_model.run(60)  # 2 months
    
#     # Plot results
#     sim_model.plot()
    
#     print(f"\nFinal inventory levels:")
#     print(f"Raw Materials: {raw_materials.values:.2f}")
#     print(f"Work in Progress: {work_in_progress.values:.2f}")
#     print(f"Finished Goods: {finished_goods.values:.2f}")
    
#     return sim_model

# def example_5_anylogic_syntax():
#     """Example 5: Most AnyLogic-like syntax"""
#     print("=" * 50)
#     print("EXAMPLE 5: ANYLOGIC-LIKE SYNTAX")
#     print("=" * 50)
    
#     # Create model first
#     sim_model = model(dt=1.0)
    
#     # Create stocks
#     tank_a = stock(values=200.0, name="Tank_A")
#     tank_b = stock(values=50.0, name="Tank_B") 
#     tank_c = stock(values=0.0, name="Tank_C")
    
#     # Add to model
#     sim_model.add_stock(tank_a)
#     sim_model.add_stock(tank_b)
#     sim_model.add_stock(tank_c)
    
#     # Create flows with AnyLogic-style expressions
#     flow_a_to_b = tank_a * 0.05  # 5% of tank_a
#     flow_b_to_c = flow(rate=3.0, name="Constant_Flow_B_to_C")  # Constant 3 units
    
#     # Set names
#     flow_a_to_b.name = "A_to_B"
    
#     # Connect flows (AnyLogic style)
#     tank_a.add_outflow(flow_a_to_b)
#     tank_b.add_inflow(flow_a_to_b)
    
#     tank_b.add_outflow(flow_b_to_c)
#     tank_c.add_inflow(flow_b_to_c)
    
#     # Add flows to model
#     sim_model.add_flow(flow_a_to_b)
#     sim_model.add_flow(flow_b_to_c)
    
#     # Run simulation
#     print("Running simulation...")
#     sim_model.run(50)
    
#     # Plot
#     sim_model.plot()
    
#     return sim_model

# if __name__ == "__main__":
#     print("System Dynamics Simulation Examples")
#     print("Using AnyLogic-style syntax")
#     print("=" * 60)
    
#     # Run all examples
#     model1 = example_1_simple_tanks()
#     model2 = example_2_constant_flow() 
#     model3 = example_3_multi_dimensional()
#     model4 = example_4_complex_system()
#     model5 = example_5_anylogic_syntax()
    
#     print("\n" + "=" * 60)
#     print("ALL EXAMPLES COMPLETED")
#     print("=" * 60)
#     print("✓ Simple proportional flows")
#     print("✓ Constant rate flows") 
#     print("✓ Multi-dimensional stocks")
#     print("✓ Complex multi-flow systems")
#     print("✓ AnyLogic-style syntax")


from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Sim.sim_SD import stock, flow, model, TimeUnit, print_progress


unit = 'day'
# Create model with specific time unit
sim_model = model(time_unit=unit, name="Day Model")

# Create simple tank system
tank1 = stock(values=100.0, name="Tank_1", units="liters")
tank2 = stock(values=0.0, name="Tank_2", units="liters")

# Flow rate: 2% per time unit
transfer = flow(rate=tank1.values* 0.02, name="Transfer_Flow",units="liters/day")

# Connect
tank1.add_outflow(transfer)
tank2.add_inflow(transfer)

# Add to model
sim_model.add_stock(tank1)
sim_model.add_stock(tank2)
sim_model.add_flow(transfer)

# Run for 10 time units
sim_model.run(duration=10)

print(f"  Final Tank1: {tank1.values:.3f} liters")
print(f"  Final Tank2: {tank2.values:.3f} liters")
print(f"  Automatic dt: {sim_model.dt:.6f} {unit}s")

sim_model.print_summary()
sim_model.plot()