import random
import numpy as np
from sensor_node import SensorNode
from mobile_charger import MobileCharger
from config import AREA_WIDTH, AREA_HEIGHT, NUM_SENSORS, MC_CAPACITY

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

def calculate_reward(prev_dead_count, curr_dead_count, distance_moved, charged=False):
    move_penalty = - (distance_moved / 100)  # less harsh (dividing by 100 not 10)

    new_deaths = curr_dead_count - prev_dead_count
    death_penalty = new_deaths * -50  # keep this harsh

    charge_bonus = 0
    if charged:
        charge_bonus = 10  # strong positive signal when it charges

    return move_penalty + death_penalty + charge_bonus