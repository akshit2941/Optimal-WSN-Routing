import random
import numpy as np
from sensor_node import SensorNode
from mobile_charger import MobileCharger
from config import AREA_WIDTH, AREA_HEIGHT, NUM_SENSORS, MC_CAPACITY, MOVEMENT_COST_PER_M

def initialize_environment(num_sensors=NUM_SENSORS):
    sensors = []
    for i in range(num_sensors):
        x = random.uniform(0, AREA_WIDTH)
        y = random.uniform(0, AREA_HEIGHT)
        sensors.append(SensorNode(i, x, y))
    mc = MobileCharger()
    return sensors, mc

def get_state_vector(sensors, mc):
    state = []

    # 1. Global state: MC energy (normalized)
    mc_energy_ratio = mc.energy / MC_CAPACITY
    state.append(mc_energy_ratio)

    for s in sensors:
        # 2. Distance to the MC (normalized)
        distance = np.linalg.norm([mc.x - s.x, mc.y - s.y])
        max_dist = np.linalg.norm([AREA_WIDTH, AREA_HEIGHT])
        distance_norm = distance / max_dist

        # 3. Residual energy (normalized)
        energy_ratio = s.energy / s.capacity

        # 4. Time window status (1 if open)
        window_open = 1 if s.needs_charging() else 0

        # 5. Time left before death (normalized)
        if s.dead or s.consumption_rate == 0:
            time_left_ratio = 0
        else:
            time_left = s.energy / s.consumption_rate
            time_left_ratio = min(time_left / 1000, 1)  # cap to avoid large numbers

        # Append sensor state
        state += [distance_norm, energy_ratio, window_open, time_left_ratio]

    return np.array(state, dtype=np.float32)

def get_valid_actions(sensors, mc):
    actions = []
    for i, sensor in enumerate(sensors):
        if sensor.needs_charging() and not sensor.dead:
            actions.append(i)
    actions.append(NUM_SENSORS)  # action to return to base
    return actions

def explain_action(action):
    if action == NUM_SENSORS:
        return "Return to base station"
    else:
        return f"Charge sensor {action}"

def count_dead_sensors(sensors):
    return sum(1 for s in sensors if s.dead)

def calculate_reward(prev_state, next_state, action, mc, sensors):
    """
    Enhanced reward function that considers charging efficiency
    """
    # Extract meaningful metrics from states
    prev_sensor_energy = sum(s.energy for s in sensors if not s.dead)
    next_sensor_energy = sum(s.energy for s in sensors if not s.dead)
    
    # Energy transferred to sensors
    energy_transferred = max(0, next_sensor_energy - prev_sensor_energy)
    
    # Energy used for movement (distance × movement cost)
    movement_distance = np.linalg.norm([mc.previous_x - mc.x, mc.y - mc.previous_y])
    movement_energy = MOVEMENT_COST_PER_M * movement_distance
    
    # Calculate different reward components
    
    # 1. Basic charging reward (positive reinforcement)
    charging_reward = energy_transferred * 0.01  # Scale factor
    
    # 2. Efficiency reward (energy transferred / energy used for movement)
    efficiency_reward = 0
    if movement_energy > 0:
        efficiency_ratio = energy_transferred / movement_energy
        efficiency_reward = min(efficiency_ratio * 10, 50)  # Cap at 50
    
    # 3. Death penalty (heavily penalize sensor deaths)
    prev_dead = sum(1 for s in sensors if s.dead)
    curr_dead = sum(1 for s in sensors if s.dead)
    death_penalty = (curr_dead - prev_dead) * -100  # Severe penalty for deaths
    
    # 4. Critical rescue bonus (reward for saving sensors close to death)
    critical_rescue_bonus = 0
    for s in sensors:
        if not s.dead and s.energy < 0.05 * s.capacity:  # Very low energy
            # If it was charged during this step
            if hasattr(s, 'previous_energy') and s.energy > s.previous_energy:
                critical_rescue_bonus += 25  # Big bonus for saving critical sensors
    
    # 5. Return-to-base reward (when appropriate)
    base_reward = 0
    if action == NUM_SENSORS:  # Return to base action
        if mc.energy < 0.2 * MC_CAPACITY:  # Low energy
            base_reward = 15  # Good decision to return
        else:
            base_reward = -5  # Unnecessary return
    
    # Combine all reward components
    total_reward = charging_reward + efficiency_reward + death_penalty + critical_rescue_bonus + base_reward
    
    return total_reward