# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 22:32:09 2025

@author: NagabhushanamTattaga
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from  Sim.sim_SD import  stock, flow, model

# Create simulation model
sim_model = model(
     time_unit="year",
     start_time="2024-01-01",
     name="Demographic Model 14x12"
 )
 
 
np.random.seed(42)  # For reproducible results
initial_pop = np.random.uniform(1000, 1000, (14, 12))


population = stock(
    dim=(14, 12),
    values=initial_pop,
    name="Population",
    units="thousands",
    min_value=0.0
)


growth_rates = np.random.uniform(0.005, 0.025, (14, 12))  # 0.5% to 2.5% annual growth

mortality_rates  = np.random.uniform(0.0025, 0.005, (14, 12))  # 0.5% to 2.5% annual growth

    
births = flow(rate=population.values*growth_rates, name="Births_Growth",units="thousands/year")
deaths = flow(rate=population.values*mortality_rates, name="Deaths", units="thousands/year")


population.add_inflow(births, "birth_inflow")
population.add_outflow(deaths, "death_outflow")
    
# Add to model
sim_model.add_stock(population)
sim_model.add_flow(births)
sim_model.add_flow(deaths)

# Run for 10 time units
sim_model.run(duration=10)


sim_model.plot()

sim_model.print_summary()

# print_progress()
# print_summary() 
# plot()     
