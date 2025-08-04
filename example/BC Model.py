import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from  Sim.sim_SD import  stock, flow, model


regions =  14
age_group = 12


p_setupSupport =  np.zeros((14, 12))
p_setupSupport[:, 0] = 1
GrowthRate = 0.322

print(p_setupSupport)


populationGrowth = p_setupSupport*(TotAsymp[Region,0]+_TotAsymp.sum(Region,0, INDEX_CAN_VARY))*(GrowthRate)

PopulationInitial = ''

Population = stock(
    values=PopulationInitial, 
    name="Population", 
    units="People",     
    min_value=0.0, 
    dim=(regions, age_group),
)




# def get_value(ratio, ratio_1, values_1):
#     """
#     Returns the interpolated value based on the second ratio-to-value table.
#     If the ratio is outside the range [0.0, 2.0], the value is extrapolated.
#     """
#     return float(np.interp(ratio, ratio_1, values_1))

# # Constants (these don't change)
# Fertility = 0.03
# PopulationInitial = 50000 
# ImmigrationNormal = 0.1
# EmigrationNormal = 0.07
# HouseholdSize = 4
# AverageLifetime = 67
# HousesInitial = 14000
# Area = 8000
# LandPerHouse = 0.1
# DemolitionNormal = 0.015
# ConstructionNormal = 0.07

# # Lookup tables (these don't change)
# HouseholdsToHousesRatio_list = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
# Construction_1 = np.array([0.2, 0.25, 0.35, 0.5, 0.7, 1.0, 1.35, 1.6, 1.8, 1.95, 2.0])
# attractions = np.array([1.4, 1.4, 1.35, 1.3, 1.15, 1.0, 0.8, 0.65, 0.5, 0.45, 0.4])
# LandAvailability = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# Construction_2 = np.array([1.0, 0.92, 0.83, 0.75, 0.66, 0.57, 0.47, 0.35, 0.24, 0.11, 0.0])

# # Create simulation model
# sim_model = model(
#     time_unit="year", 
#     start_time="2024-01-01", 
#     name="Population Model - Dynamic Rates"
# )

# # Create stocks


# Houses = stock(
#     values=HousesInitial,    
#     name="Houses",    
#     units="Units",    
#     min_value=0.0 
# )

# # ============================================================================
# # CRITICAL FIX: Define flow rate functions that are called every time step
# # These functions access current stock values dynamically
# # ============================================================================

# def get_birth_rate():
#     """Calculate birth rate based on current population - called every time step"""
#     return Fertility * Population.values

# def get_immigration_rate():
#     """Calculate immigration rate based on current conditions - called every time step"""
#     # Calculate current ratios dynamically
#     current_households_to_houses_ratio = Population.values / (Houses.values * HouseholdSize)
    
#     # Get attraction multiplier based on current ratio
#     attraction_due_to_housing = get_value(
#         current_households_to_houses_ratio, 
#         HouseholdsToHousesRatio_list, 
#         attractions
#     )
    
#     return Population.values * ImmigrationNormal * attraction_due_to_housing

# def get_emigration_rate():
#     """Calculate emigration rate - called every time step"""
#     return Population.values * EmigrationNormal

# def get_death_rate():
#     """Calculate death rate - called every time step"""
#     return Population.values / AverageLifetime

# def get_construction_rate():
#     """Calculate construction rate based on current conditions - called every time step"""
#     # Calculate current ratios dynamically
#     current_households_to_houses_ratio = Population.values / (Houses.values * HouseholdSize)
#     current_fraction_occupied_land = (Houses.values * LandPerHouse) / Area
    
#     # Get multipliers based on current conditions
#     construction_due_to_land_availability = get_value(
#         current_fraction_occupied_land,
#         LandAvailability,  
#         Construction_2
#     )
    
#     construction_due_to_housing_availability = get_value(
#         current_households_to_houses_ratio, 
#         HouseholdsToHousesRatio_list, 
#         Construction_1 
#     )
    
#     # Calculate total construction multiplier
#     construction_multiplier = (construction_due_to_housing_availability * 
#                              construction_due_to_land_availability)
    
#     return construction_multiplier * ConstructionNormal * Houses.values

# def get_demolition_rate():
#     """Calculate demolition rate - called every time step"""
#     return Houses.values * DemolitionNormal

# # ============================================================================
# # Create flows with FUNCTION REFERENCES (not function calls!)
# # This ensures the functions are called every simulation step
# # ============================================================================

# # Population flows
# Births = flow(
#     rate=get_birth_rate,  # Function reference, not get_birth_rate()
#     name="Births",
#     units="People/year"
# )

# Immigration = flow(
#     rate=get_immigration_rate,  # Function reference, not get_immigration_rate()
#     name="Immigration",
#     units="People/year"
# )

# Emigration = flow(
#     rate=get_emigration_rate,  # Function reference, not get_emigration_rate()
#     name="Emigration",
#     units="People/year"
# )

# Deaths = flow(
#     rate=get_death_rate,  # Function reference, not get_death_rate()
#     name="Deaths",
#     units="People/year"
# )

# # Housing flows  
# ConstructionRate = flow(
#     rate=get_construction_rate,  # Function reference, not get_construction_rate()
#     name="Construction",
#     units="Houses/year"
# )

# DemolitionRate = flow(
#     rate=get_demolition_rate,  # Function reference, not get_demolition_rate()
#     name="Demolition",
#     units="Houses/year"
# )

# # ============================================================================
# # Connect flows to stocks
# # ============================================================================

# # Population connections
# Population.add_inflow(Births, "birth_inflow")
# Population.add_inflow(Immigration, "immigration_inflow")
# Population.add_outflow(Emigration, "emigration_outflow")
# Population.add_outflow(Deaths, "death_outflow")

# # Housing connections
# Houses.add_inflow(ConstructionRate, "construction_inflow")
# Houses.add_outflow(DemolitionRate, "demolition_outflow")  # Note: This should be outflow, not inflow

# # Add to model
# sim_model.add_stock(Population)
# sim_model.add_stock(Houses)
# sim_model.add_flow(Births)
# sim_model.add_flow(Immigration)
# sim_model.add_flow(Emigration)
# sim_model.add_flow(Deaths)
# sim_model.add_flow(ConstructionRate)
# sim_model.add_flow(DemolitionRate)

# # ============================================================================
# # Add some debugging to see what's happening
# # ============================================================================

# def print_model_state(step_count):
#     """Print current model state every 10 years"""
#     if step_count % int(10 / sim_model.dt) == 0:  # Every 10 years
#         year = step_count * sim_model.dt
        
#         # Calculate current ratios
#         households_ratio = Population.values / (Houses.values * HouseholdSize)
#         land_fraction = (Houses.values * LandPerHouse) / Area
        
#         print(f"\nYear {year:.1f}:")
#         print(f"  Population: {Population.values:,.0f}")
#         print(f"  Houses: {Houses.values:,.0f}")
#         print(f"  Households/Houses Ratio: {households_ratio:.3f}")
#         print(f"  Land Fraction Occupied: {land_fraction:.3f}")
#         print(f"  Birth Rate: {get_birth_rate():.1f} people/year")
#         print(f"  Immigration Rate: {get_immigration_rate():.1f} people/year")
#         print(f"  Construction Rate: {get_construction_rate():.1f} houses/year")

# # Run simulation
# print("Starting dynamic population simulation...")
# print("This should show non-linear, cyclical behavior!")

# # Store step count for debugging
# original_step = sim_model.step
# step_counter = 0

# def debug_step():
#     global step_counter
#     step_counter += 1
#     if step_counter % int(10 / sim_model.dt) == 0:  # Every 10 years
#         print_model_state(step_counter)
#     return original_step()

# sim_model.step = debug_step

# # Run for 100 years
# sim_model.run(duration=100)

# # Plot results
# sim_model.plot()
# sim_model.print_summary()

# # ============================================================================
# # Additional analysis to show the cyclical behavior
# # ============================================================================

# print("\n" + "="*60)
# print("DYNAMIC BEHAVIOR ANALYSIS")
# print("="*60)

# # Calculate ratios over time
# population_history = [h for h in Population.history]
# houses_history = [h for h in Houses.history]

# households_ratio_history = []
# land_fraction_history = []

# for i in range(len(population_history)):
#     pop = population_history[i]
#     houses = houses_history[i]
    
#     ratio = pop / (houses * HouseholdSize)
#     land_frac = (houses * LandPerHouse) / Area
    
#     households_ratio_history.append(ratio)
#     land_fraction_history.append(land_frac)

# # Create detailed analysis plots
# plt.figure(figsize=(15, 12))

# # Plot 1: Population and Houses
# plt.subplot(3, 2, 1)
# plt.plot(sim_model.time_history, population_history, 'b-', label='Population', linewidth=2)
# plt.ylabel('Population')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.title('Population Over Time')

# plt.subplot(3, 2, 2)
# plt.plot(sim_model.time_history, houses_history, 'r-', label='Houses', linewidth=2)
# plt.ylabel('Houses')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.title('Houses Over Time')

# # Plot 2: Dynamic ratios
# plt.subplot(3, 2, 3)
# plt.plot(sim_model.time_history, households_ratio_history, 'g-', linewidth=2)
# plt.ylabel('Households/Houses Ratio')
# plt.grid(True, alpha=0.3)
# plt.title('Dynamic Households to Houses Ratio')

# plt.subplot(3, 2, 4)
# plt.plot(sim_model.time_history, land_fraction_history, 'orange', linewidth=2)
# plt.ylabel('Land Fraction Occupied')
# plt.grid(True, alpha=0.3)
# plt.title('Dynamic Land Occupancy')

# # Plot 3: Flow rates over time
# plt.subplot(3, 2, 5)
# if Immigration.history:
#     immigration_rates = [h for h in Immigration.history]
#     construction_rates = [h for h in ConstructionRate.history]
#     time_flow = sim_model.time_history[1:len(immigration_rates)+1]
    
#     plt.plot(time_flow, immigration_rates, 'purple', label='Immigration Rate', linewidth=2)
#     plt.ylabel('Immigration Rate')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.title('Dynamic Immigration Rate')

# plt.subplot(3, 2, 6)
# if ConstructionRate.history:
#     plt.plot(time_flow, construction_rates, 'brown', label='Construction Rate', linewidth=2)
#     plt.ylabel('Construction Rate')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.title('Dynamic Construction Rate')

# plt.xlabel('Years')
# plt.tight_layout()
# plt.show()

# print(f"\nFinal Population: {Population.values:,.0f}")
# print(f"Final Houses: {Houses.values:,.0f}")
# print(f"Final Households/Houses Ratio: {Population.values / (Houses.values * HouseholdSize):.3f}")
# print(f"Final Land Fraction: {(Houses.values * LandPerHouse) / Area:.3f}")

# print("\n✅ KEY FIX: Flow rates now recalculate every time step!")
# print("✅ This creates dynamic, non-linear behavior instead of linear growth!")
# print("✅ The system now shows feedback loops and cyclical patterns!")