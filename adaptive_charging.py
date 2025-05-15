import numpy as np
import random
from enhanced_visualization import visualize_optimal_position
from environment import initialize_environment
from sensor_node import SensorNode
from config import *

def calculate_adaptive_priority(sensor, mc, alpha=0.3, beta=0.2, gamma=0.2, delta=0.3):
    """
    Calculate adaptive priority score for a sensor based on multiple factors
    """
    # 1. Normalize energy level percentage (lower energy % = higher priority)
    energy_ratio = sensor.energy / sensor.capacity
    energy_percentage_factor = 1 - energy_ratio
    
    # 2. Residual energy factor (lower absolute energy = higher priority)
    # Use CRITICAL_ENERGY_RATIO from config instead of hardcoded 300J
    min_critical_energy = CRITICAL_ENERGY_RATIO * SENSOR_CAPACITY
    residual_energy_factor = 1 - min(1, max(0, sensor.energy - min_critical_energy) / 
                                    (sensor.capacity - min_critical_energy))
    
    # 3. Normalize consumption rate (use CHARGING_RATE as reference)
    # Max consumption should be related to the charging rate
    max_consumption_rate = CHARGING_RATE / 1000  # Was hardcoded as 0.0001
    consumption_factor = min(sensor.consumption_rate / max_consumption_rate, 1.0)
    
    # 4. Normalize distance
    distance = np.linalg.norm([mc.x - sensor.x, mc.y - sensor.y])
    max_distance = np.sqrt(AREA_WIDTH**2 + AREA_HEIGHT**2)
    distance_factor = 1 - (distance / max_distance)
    
    # Calculate urgency - scaled by MC_SPEED (movement capability)
    # Time units in seconds scaled by movement capability
    time_scale = 200 * (10 / MC_SPEED)  # Adjusts based on MC speed
    if sensor.consumption_rate > 0:
        time_to_death = sensor.energy / sensor.consumption_rate
        urgency_factor = max(0, min(1, 1 - (time_to_death - (time_scale/2)) / time_scale))
    else:
        urgency_factor = 0
    
    # Combine factors with weights
    priority = (alpha * energy_percentage_factor + 
                beta * consumption_factor + 
                gamma * distance_factor +
                delta * residual_energy_factor) * (1 + urgency_factor)
    
    return priority

def get_sensors_needing_charging(sensors, mc, adaptive_threshold=True, fixed_threshold_ratio=THRESHOLD_RATIO):
    """
    Get sensors needing charging, either using a fixed threshold or an adaptive approach.
    """
    if not adaptive_threshold:
        # Original approach: fixed threshold
        to_charge = [s for s in sensors if s.needs_charging()]
        # Sort by distance (closest first)
        to_charge.sort(key=lambda s: np.linalg.norm([mc.x - s.x, mc.y - s.y]))
        return to_charge
    
    # Adaptive approach
    sensor_priorities = []
    for sensor in sensors:
        if sensor.dead:
            continue
            
        # Calculate priority score
        priority = calculate_adaptive_priority(sensor, mc)
        
        # Add to list if priority exceeds minimum threshold
        if priority > MIN_PRIORITY_THRESHOLD:  # From config
            sensor_priorities.append((sensor, priority))
    
    # Sort by priority (highest first)
    sensor_priorities.sort(key=lambda x: x[1], reverse=True)
    
    # Return just the sensors in priority order
    return [s for s, _ in sensor_priorities]

def find_optimal_charging_position(mc, sensors, candidate_sensors=None, grid_size=GRID_SEARCH_SIZE):
    """
    Find the optimal position for the charger to maximize charging efficiency 
    for multiple sensors.
    """
    if candidate_sensors is None:
        # Use threshold from config
        candidate_sensors = [s for s in sensors if not s.dead and 
                            s.energy < (1-THRESHOLD_RATIO) * s.capacity]
    
    if not candidate_sensors:
        return (mc.x, mc.y), [], 0  # No sensors need charging
    
    # Define the search area using config parameters
    min_x = max(0, min(s.x for s in candidate_sensors) - CHARGING_RADIUS*CHARGING_SEARCH_EXPANSION)
    max_x = min(AREA_WIDTH, max(s.x for s in candidate_sensors) + CHARGING_RADIUS*CHARGING_SEARCH_EXPANSION)
    min_y = max(0, min(s.y for s in candidate_sensors) - CHARGING_RADIUS*CHARGING_SEARCH_EXPANSION)
    max_y = min(AREA_HEIGHT, max(s.y for s in candidate_sensors) + CHARGING_RADIUS*CHARGING_SEARCH_EXPANSION)
    
    # Increase grid resolution for better position finding
    step_x = (max_x - min_x) / grid_size
    step_y = (max_y - min_y) / grid_size
    
    best_pos = (mc.x, mc.y)  # Default to current position
    best_score = 0
    best_covered = []
    
    # Evaluate each grid position
    for i in range(grid_size + 1):
        for j in range(grid_size + 1):
            pos_x = min_x + i * step_x
            pos_y = min_y + j * step_y
            
            score, covered = evaluate_charging_position(pos_x, pos_y, candidate_sensors)
            
            # Movement energy cost from config
            distance_to_move = np.linalg.norm([mc.x - pos_x, mc.y - pos_y])
            energy_cost = distance_to_move * MOVEMENT_COST_PER_M
            
            # Scale based on MC_CAPACITY instead of hardcoded value
            movement_penalty = min(0.3, energy_cost / (MC_CAPACITY * 0.008))
            
            # Apply coverage bonus based on number of sensors covered
            coverage_bonus = 1.0 + min(1.5, len(covered) * 0.3)
            
            # Adjusted score with reduced movement penalty and coverage bonus
            adjusted_score = score * coverage_bonus * (1 - 0.1 * movement_penalty)
            
            # Require minimum sensors or significant improvement to change position
            if (len(covered) >= MIN_SENSORS_FOR_POSITION and adjusted_score > best_score) or \
               (adjusted_score > best_score * SCORE_IMPROVEMENT_FACTOR):
                best_score = adjusted_score
                best_pos = (pos_x, pos_y)
                best_covered = covered
    
    return best_pos, best_covered, best_score

def evaluate_charging_position(x, y, sensors):
    """
    Evaluate a potential charging position with improved priority calculations
    """
    total_score = 0
    covered_sensors = []
    
    for sensor in sensors:
        # Check if sensor is within charging radius
        distance = np.linalg.norm([x - sensor.x, y - sensor.y])
        
        if distance <= CHARGING_RADIUS:
            # Calculate zone-based efficiency using config parameters
            distance_ratio = distance / CHARGING_RADIUS
            
            if distance_ratio <= CHARGING_INNER_ZONE:
                efficiency = INNER_EFFICIENCY
                zone_weight = 1.0
            elif distance_ratio <= CHARGING_MIDDLE_ZONE:
                efficiency = MIDDLE_EFFICIENCY
                zone_weight = 0.8
            else:
                efficiency = OUTER_EFFICIENCY
                zone_weight = 0.5
            
            # Revised priority calculation using config-based values
            energy_deficit = 1 - (sensor.energy / sensor.capacity)
            consumption_factor = min(sensor.consumption_rate / (CHARGING_RATE/1000), 1.0)
            
            # Add urgency factor
            time_factor = MC_SPEED * 200  # Scale by MC_SPEED
            urgency = max(0, 1 - (sensor.energy / (sensor.consumption_rate * time_factor)))
            urgency_boost = 1.0 + min(2.0, urgency)
            
            # Enhanced sensor score calculation
            sensor_score = (
                efficiency * 
                zone_weight * 
                (energy_deficit * 0.6 + consumption_factor * 0.4) * 
                (sensor.capacity - sensor.energy) *  # Absolute energy need
                urgency_boost  # Prioritize based on time-to-death
            )
            
            total_score += sensor_score
            covered_sensors.append(sensor.id)
    
    # Apply a nonlinear scaling for multiple covered sensors
    coverage_multiplier = 1.0
    if len(covered_sensors) > 1:
        coverage_multiplier = len(covered_sensors) ** 1.5  # Superlinear scaling
    
    return total_score * coverage_multiplier, covered_sensors

def run_optimal_position_simulation(num_steps=10, time_step=3, num_sensors=NUM_SENSORS):
    """
    Run a simulation using the optimal position strategy for charging,
    with improved round-trip energy planning
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize environment
    sensors, mc = initialize_environment(num_sensors=num_sensors)
    
    # Define base station coordinates (consistent with visualization)
    base_station_x, base_station_y = 300, 300
    
    # Use config-based threshold for returning to base
    return_to_base_threshold = RETURN_THRESHOLD_RATIO * MC_CAPACITY
    
    # Initialize path history and evaluation metrics
    path_history = [(mc.x, mc.y)]
    total_energy_transferred = 0
    total_movement_energy = 0
    sensors_requested_charging = set()
    sensors_received_charging = set()
    
    # Initialize additional metrics for charging delay
    sensors_request_times = {}  # Stores when each sensor first requested charging {sensor_id: time}
    sensors_charged_times = {}  # Stores when each sensor was first charged {sensor_id: time}
    current_time = 0  # Track simulation time
    
    # Set varied energy levels using thresholds from config
    for i, s in enumerate(sensors):
        if i % 5 == 0:
            s.energy = 0.05 * SENSOR_CAPACITY
        elif i % 5 == 1:
            s.energy = 0.15 * SENSOR_CAPACITY
        elif i % 5 == 2:
            s.energy = 0.30 * SENSOR_CAPACITY
        elif i % 5 == 3:
            s.energy = 0.45 * SENSOR_CAPACITY
            
        # Vary consumption rates relative to CHARGING_RATE
        if i % 3 == 0:  # High consumption
            s.consumption_rate = random.uniform(CHARGING_RATE/60000, CHARGING_RATE/48000)
        elif i % 3 == 1:  # Medium consumption
            s.consumption_rate = random.uniform(CHARGING_RATE/120000, CHARGING_RATE/60000)
        else:  # Low consumption
            s.consumption_rate = random.uniform(CHARGING_RATE/480000, CHARGING_RATE/160000)
    
    # Run simulation
    for step in range(num_steps):
        print(f"--- Step {step+1}/{num_steps} --- MC Energy: {mc.energy:.1f}J")
        
        # Check remaining sensors and see if we need to continue
        alive_count = sum(1 for s in sensors if not s.dead)
        if alive_count == 0:
            print("All sensors have died! Ending simulation.")
            break
        
        # Get sensors that need charging
        to_charge = get_sensors_needing_charging(sensors, mc, adaptive_threshold=True)
        
        # Track sensors that requested charging, along with their request time
        for sensor in to_charge:
            if sensor.id not in sensors_requested_charging:
                sensors_requested_charging.add(sensor.id)
                sensors_request_times[sensor.id] = current_time
        
        if not to_charge:
            # Fast-forward time when no charging needed - use MC_SPEED to scale
            fast_forward = time_step * max(2, min(10, int(MC_SPEED)))
            for s in sensors:
                s.update_energy(fast_forward)
            print(f"No sensors need charging. Fast-forwarding {fast_forward}s of simulation time.")
            continue
        
        # Find the optimal charging position
        print(f"Finding optimal position to charge {len(to_charge)} candidate sensors...")
        optimal_pos, covered_sensors, efficiency_score = find_optimal_charging_position(mc, sensors, to_charge)
        
        # Visualize the optimal position before moving
        print(f"Found optimal position at {optimal_pos} with efficiency score {efficiency_score:.2f}")
        print(f"This position would cover {len(covered_sensors)} sensors: {covered_sensors}")
        visualize_optimal_position(sensors, mc, optimal_pos, covered_sensors, path_history)
        
        # NEW ROUND-TRIP ENERGY PLANNING with config values
        distance_to_target = np.linalg.norm([mc.x - optimal_pos[0], mc.y - optimal_pos[1]])
        distance_from_target_to_base = np.linalg.norm([optimal_pos[0] - base_station_x, optimal_pos[1] - base_station_y])
        
        energy_to_target = distance_to_target * MOVEMENT_COST_PER_M
        energy_to_return_to_base = distance_from_target_to_base * MOVEMENT_COST_PER_M
        
        # Estimate charging energy based on CHARGING_RATE and time_step
        est_charging_energy = len(covered_sensors) * CHARGING_RATE * time_step
        
        # Calculate total energy needed for round trip
        total_energy_needed = energy_to_target + est_charging_energy + energy_to_return_to_base
        
        # print(f"Energy analysis:")
        # print(f"  - Current energy: {mc.energy:.1f}J")
        # print(f"  - Energy to target: {energy_to_target:.1f}J")
        # print(f"  - Estimated charging energy: {est_charging_energy:.1f}J")
        # print(f"  - Energy to return to base: {energy_to_return_to_base:.1f}J")
        # print(f"  - Total energy needed: {total_energy_needed:.1f}J")
        
        # Modify the base station return logic:
        base_return_threshold = RETURN_THRESHOLD_RATIO * MC_CAPACITY
        if mc.energy < total_energy_needed:
            # Check if worth charging at current position before returning
            distance_to_base = np.linalg.norm([mc.x - base_station_x, mc.y - base_station_y])
            energy_needed_for_safe_return = distance_to_base * MOVEMENT_COST_PER_M * 1.2  # 20% safety margin
            
            # If we have enough energy for some charging and safe return, do a partial mission
            if mc.energy > energy_needed_for_safe_return + (est_charging_energy * 0.5):
                print(f"⚠️ Partially charging before return - enough energy for charging and safe return.")
                # Move to position, charge, then return
                mc.move_to(optimal_pos[0], optimal_pos[1])
                charged_nodes, energy_transferred = mc.charge_nodes_in_radius(sensors)
                total_energy_transferred += energy_transferred
                # Then ensure return to base
                distance_to_base = np.linalg.norm([mc.x - base_station_x, mc.y - base_station_y])
                energy_to_base = distance_to_base * MOVEMENT_COST_PER_M
                mc.move_to(base_station_x, base_station_y)
                total_movement_energy += energy_to_base
                path_history.append((base_station_x, base_station_y))
                previous_energy = mc.energy
                mc.energy = MC_CAPACITY
                print(f"MC recharged at base station: {previous_energy:.1f}J → {mc.energy:.1f}J")
            else:
                print(f"⚠️ Not enough energy for charging mission. Returning to base first...")
                distance_to_base = np.linalg.norm([mc.x - base_station_x, mc.y - base_station_y])
                energy_to_base = distance_to_base * MOVEMENT_COST_PER_M
                mc.move_to(base_station_x, base_station_y)
                total_movement_energy += energy_to_base
                path_history.append((base_station_x, base_station_y))
                previous_energy = mc.energy
                mc.energy = MC_CAPACITY
                print(f"MC recharged at base station: {previous_energy:.1f}J → {mc.energy:.1f}J")
        else:
            # Enough energy for round trip, proceed normally
            print(f"✓ Sufficient energy for complete mission. Moving to optimal position...")
            mc.move_to(optimal_pos[0], optimal_pos[1])
            total_movement_energy += energy_to_target

        # Add to path history
        path_history.append((mc.x, mc.y))

        # Charge all sensors within radius
        charged_nodes, energy_transferred = mc.charge_nodes_in_radius(sensors)
        total_energy_transferred += energy_transferred

        # Track which sensors received charging, along with charge time
        for charged_item in charged_nodes:
            if isinstance(charged_item, tuple):
                sensor_id = charged_item[0]  # First item is sensor ID
            else:
                sensor_id = charged_item
                
            sensors_received_charging.add(sensor_id)
            
            # Record the time when sensor was first charged (if it hasn't been charged before)
            if sensor_id not in sensors_charged_times:
                sensors_charged_times[sensor_id] = current_time

        # Update simulation time
        current_time += time_step

        # Update sensor energy levels for this time step
        for s in sensors:
            s.update_energy(time_step)
    
    # Calculate metrics but DON'T display them
    # Only calculate and return the data

    # 1. Survival Rate
    if len(sensors_requested_charging) > 0:
        survival_rate = len(sensors_received_charging) / len(sensors_requested_charging)
        sensors_requested = sorted(list(sensors_requested_charging))
        sensors_received = sorted(list(sensors_received_charging))
    else:
        survival_rate = 1.0
        sensors_requested = []
        sensors_received = []

    # 2. Energy Efficiency
    total_energy_used = total_movement_energy + total_energy_transferred
    if total_energy_used > 0:
        energy_efficiency = total_energy_transferred / total_energy_used
    else:
        energy_efficiency = 0

    # 3. Sensor Health Statistics
    alive_sensors = sum(1 for s in sensors if not s.dead)
    dead_sensors = sum(1 for s in sensors if s.dead)
    average_energy = sum(s.energy for s in sensors if not s.dead) / max(1, alive_sensors)
    average_energy_pct = average_energy / SENSOR_CAPACITY * 100

    # Calculate average charging delay
    charging_delays = []
    for sensor_id in sensors_received_charging:
        if sensor_id in sensors_request_times and sensor_id in sensors_charged_times:
            delay = sensors_charged_times[sensor_id] - sensors_request_times[sensor_id]
            charging_delays.append(delay)
    
    avg_charging_delay = 0
    if charging_delays:
        avg_charging_delay = sum(charging_delays) / len(charging_delays)
    
    # Return metrics including the new delay metric
    metrics = {
        "survival_rate": survival_rate,
        "energy_efficiency": energy_efficiency,
        "alive_ratio": alive_sensors/len(sensors),
        "average_energy": average_energy,
        "total_energy_transferred": total_energy_transferred,
        "total_movement_energy": total_movement_energy,
        "sensors_requested": sensors_requested,
        "sensors_received": sensors_received,
        "avg_charging_delay": avg_charging_delay,
        "charging_delays": charging_delays  # Include individual delays for detailed analysis
    }

    return sensors, mc, path_history, metrics