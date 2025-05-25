# Wireless Rechargeable Sensor Network Simulation

## Project Overview

This project implements a simulation framework for Wireless Rechargeable Sensor Networks (WRSNs) with an adaptive mobile charging strategy. The framework models sensor energy consumption, mobile charger movement, and various charging optimization algorithms to maximize network lifetime and charging efficiency.

## Features

- **Adaptive Position-Based Charging**: Optimizes mobile charger positioning to maximize charging coverage.
- **Deep Q-Learning Agent**: Reinforcement learning approach for charging decisions.
- **Zone-Based Charging Model**: Variable charging efficiency based on sensor distance.
- **Position Batching**: Extended charging at high-value positions.
- **Energy-Aware Path Planning**: Round-trip energy planning with efficient return-to-base strategies.
- **Comprehensive Metrics**: Detailed analysis of network performance and charging efficiency.

## Project Structure

- `main.py`: Entry point that orchestrates the simulation pipeline.
- `adaptive_charging.py`: Core implementation of position-based charging algorithm.
- `environment.py`: Environment setup, state representation, and reward calculation.
- `sensor_node.py`: Sensor node implementation with energy model.
- `mobile_charger.py`: Mobile charger implementation with movement and charging capabilities.
- `reinforcement_learning.py`: DQN implementation for learning-based charging.
- `config.py`: Centralized configuration parameters.
- `visualization.py` / `enhanced_visualization.py`: Visualization utilities for simulation results.
- `continuous_charging.py`: Implementation of charging while moving.

## Installation

```bash
# Create a virtual environment (recommended)
conda create -n WSNs python=3.8
conda activate WSNs

# Install required packages
pip install numpy matplotlib torch
```

## Configuration

Key parameters are centralized in `config.py`:

```python
# Environment dimensions
AREA_WIDTH = 600        # Width of the simulation area (m)
AREA_HEIGHT = 600       # Height of the simulation area (m)

# Sensor and Mobile Charger Parameters
NUM_SENSORS = 100               # Number of sensors in the network
SENSOR_CAPACITY = 10800         # Sensor battery capacity (J)
MC_CAPACITY = 1000000           # Mobile charger capacity (J)
CHARGING_RADIUS = 60            # Charging radius (m)
CHARGING_RATE = 3.6             # Charging rate (J/s)
MOVEMENT_COST_PER_M = 60        # Energy cost for movement (J/m)
```

## Usage

To run the simulation:

```bash
python main.py
```

This will execute:

1. Environment initialization with sensor placement
2. DQN agent training or loading (if previously saved)
3. Optimal position-based charging simulation
4. Results analysis and reporting

## Implementation Details

### Adaptive Charging Strategy

The system employs a multi-factor priority calculation for determining which sensors to charge:

1. **Energy Level**: Prioritizes sensors with lower energy levels
2. **Consumption Rate**: Higher weight for sensors consuming energy faster
3. **Distance**: Considers proximity to optimize movement
4. **Time-to-Death**: Urgency factor based on projected sensor lifetime

### Optimal Position Finding

The simulation uses a grid-based approach to find the best charging position:

```python
def find_optimal_charging_position(mc, sensors, candidate_sensors=None, grid_size=GRID_SEARCH_SIZE):
    # Search for position that maximizes charging efficiency for multiple sensors
    # Uses a grid-based approach with adjustable resolution
    # Returns (position, covered_sensors, efficiency_score)
```

### Zone-Based Charging Model

Charging efficiency varies based on distance from the mobile charger:

- **Inner Zone** (0-40% of radius): 75% efficiency
- **Middle Zone** (40-70% of radius): 55% efficiency
- **Outer Zone** (70-100% of radius): 35% efficiency

### Position Batching

For positions covering multiple sensors, the charger stays longer:

```python
if len(covered_sensors) >= 3:
    extra_charging_rounds = min(4, len(covered_sensors) // 2 + 1)
    # Extended charging at valuable positions
```

## Metrics and Evaluation

The simulation calculates multiple performance metrics:

1. **Life-Survival Ratio**: Network lifetime with charging / without charging
2. **Energy Efficiency**: Energy transferred to sensors / total energy used
3. **Survival Rate**: Percentage of charging requests fulfilled
4. **Charging Delay**: Time between request and charging
5. **Network Health**: Percentage of alive sensors and average energy levels

## Advanced Features

### Deep Q-Learning Agent

The DQN implementation features:

- Prioritized experience replay for efficient learning
- Dueling network architecture (separate value and advantage streams)
- Target network for stable learning
- Dynamic learning rate schedule

### Energy-Aware Path Planning

The mobile charger plans charging missions with energy awareness:

```python
# Calculate total energy needed for round trip
total_energy_needed = energy_to_target + est_charging_energy + energy_to_return_to_base

# Make decisions based on energy constraints
if mc.energy < total_energy_needed:
    # Partial mission or return to base logic
```

## Extensions and Future Work

The simulation framework is designed to be extensible. Possible enhancements include:

- Dynamic sensor deployment and failure models
- Multi-charger coordination strategies
- Energy harvesting sensor models
- Network topology-aware charging strategies
- Predictive consumption models for proactive charging
