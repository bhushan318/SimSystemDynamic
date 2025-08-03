# 🌀 System Dynamics Simulation in Python

A lightweight, intuitive Python implementation of **System Dynamics (SD)** modeling. This tool allows users to define **stocks**, **flows**, and **flow rates**, and simulate the behavior of dynamic systems over time using `model.run()`.

---

## 📌 Features

* Define **stocks** with initial values
* Define **flows** with mathematical rate functions
* Automatically compute net inflows/outflows to each stock
* Run time-based simulations using simple Python syntax
* Visualize simulation results over time

---

## 🚀 Getting Started

### Installation

Clone this repository:

```bash
git clone https://github.com/bhushan318/SimSystemDynamic.git
cd sSimSystemDynamic
```

Install dependencies (if any):

```bash
pip install -r requirements.txt
```

### Example Usage

```python
from simulation import Model, Stock, Flow

# Define stocks
stock_population = Stock(name="Population", initial_value=1000)

# Define flow with a rate function
def birth_rate(t, state):
    return 0.02 * state["Population"]  # 2% growth per time step

birth_flow = Flow(name="Births", from_stock=None, to_stock=stock_population, rate_func=birth_rate)

# Build model
model = Model(start=0, end=50, dt=1)
model.add_stock(stock_population)
model.add_flow(birth_flow)

# Run simulation
results = model.run()

# Plot results
model.plot()
```

---

## 📈 Output

The simulation results will show how stock values (like population) evolve over time based on defined flow rules. A time series plot is generated for all stocks automatically.

---

## 📂 Folder Structure

```
system-dynamics-python/
├── simulation.py         # Core simulation engine
├── examples/             # Example models and usage
├── tests/                # Unit tests
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

---

## 🧠 Applications

* Population dynamics
* Supply chain models
* Epidemiological modeling
* Economic and ecological systems
* Feedback loops and control systems

---

## ✅ TODO

* [ ] Support for auxiliary/converter variables
* [ ] Support for graphical model building (GUI)
* [ ] Save/load model definitions
* [ ] Advanced integration methods (e.g., RK4)

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit pull requests.

---
