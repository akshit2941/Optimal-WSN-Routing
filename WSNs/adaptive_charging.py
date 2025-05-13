import numpy as np
import random
from enhanced_visualization import visualize_optimal_position
from environment import initialize_environment
from sensor_node import SensorNode
from config import AREA_WIDTH, AREA_HEIGHT, CHARGING_RADIUS, MC_CAPACITY, NUM_SENSORS, SENSOR_CAPACITY, THRESHOLD_RATIO, MOVEMENT_COST_PER_M

def calculate_adaptive_priority(sensor, mc, alpha=0.3, beta=0.2, gamma=0.2, delta=0.3):
    """
    Calculate adaptive priority score for a sensor based on multiple factors:
    - Energy level percentage (lower % = higher priority)
    - Residual energy (lower absolute energy = higher priority)
    - Consumption rate (higher rate = higher priority)
    - Distance from MC (shorter distance = higher priority)
    
    Parameters:
    - alpha: weight for energy level percentage (0.0-1.0)
    - beta: weight for consumption rate factor (0.0-1.0)
    - gamma: weight for distance factor (0.0-1.0)
    - delta: weight for residual energy factor (0.0-1.0)
    
    Returns a priority score where higher values indicate higher priority
    """
    # 1. Normalize energy level percentage (lower energy % = higher priority)
    energy_ratio = sensor.energy / sensor.capacity
    energy_percentage_factor = 1 - energy_ratio  # invert so lower energy % = higher value
    
    # 2. Residual energy factor (lower absolute energy = higher priority)
    # Normalize against a reasonable minimum energy (e.g., 300J)
    min_critical_energy = 300  # Consider this as a critical minimum energy level in Joules
    residual_energy_factor = 1 - min(1, max(0, sensor.energy - min_critical_energy) / 
                                    (sensor.capacity - min_critical_energy))
    
    # 3. Normalize consumption rate (higher rate = higher priority)
    max_consumption_rate = 0.0001  # Updated to match new consumption rate range
    consumption_factor = min(sensor.consumption_rate / max_consumption_rate, 1.0)
    
    # 4. Normalize distance (shorter distance = higher priority)
    distance = np.linalg.norm([mc.x - sensor.x, mc.y - sensor.y])
    max_distance = np.sqrt(AREA_WIDTH**2 + AREA_HEIGHT**2)
    distance_factor = 1 - (distance / max_distance)  # invert so closer = higher value
    
    # Calculate urgency based on remaining time until death
    if sensor.consumption_rate > 0:
        time_to_death = sensor.energy / sensor.consumption_rate
        # Normalize: anything under 100 seconds is urgent, over 1000 seconds is not urgent
        urgency_factor = max(0, min(1, 1 - (time_to_death - 100) / 900))
    else:
        urgency_factor = 0
    
    # Combine factors with weights
    # Note that alpha + beta + gamma + delta should ideally sum to 1.0
    priority = (alpha * energy_percentage_factor + 
                beta * consumption_factor + 
                gamma * distance_factor +
                delta * residual_energy_factor) * (1 + urgency_factor)
    
    return priority

def get_sensors_needing_charging(sensors, mc, adaptive_threshold=True, fixed_threshold_ratio=THRESHOLD_RATIO):
    """
    Get sensors needing charging, either using a fixed threshold or an adaptive approach.
    
    Parameters:
    - sensors: list of SensorNode objects
    - mc: MobileCharger object
    - adaptive_threshold: whether to use adaptive thresholds (True) or fixed (False)
    - fixed_threshold_ratio: ratio to use if using fixed threshold
    
    Returns list of sensors that need charging, sorted by priority
    """
    if not adaptive_threshold:
        # Original approach: fixed threshold
        to_charge = [s for s in sensors if s.needs_charging()]
        # Sort by distance (closest first)
        to_charge.sort(key=lambda s: np.linalg.norm([mc.x - s.x, mc.y - s.y]))
        return to_charge
    
    # Adaptive approach:
    # Calculate priority for all sensors
    sensor_priorities = []
    for sensor in sensors:
        if sensor.dead:
            continue
            
        # Calculate priority score
        priority = calculate_adaptive_priority(sensor, mc)
        
        # Add to list if priority exceeds minimum threshold
        # Higher energy sensors might still be charged if they're very close and 
        # have high consumption rate
        min_priority_threshold = 0.15  # Only consider sensors with some level of need
        if priority > min_priority_threshold:
            sensor_priorities.append((sensor, priority))
    
    # Sort by priority (highest first)
    sensor_priorities.sort(key=lambda x: x[1], reverse=True)
    
    # Return just the sensors in priority order
    return [s for s, _ in sensor_priorities]

def simulate_step_adaptive(sensors, mc, time_step=10):
    """Simulation step using adaptive charging threshold"""
    # 1. Update energy of all sensors
    for s in sensors:
        s.update_energy(time_step)

    # 2. Get sensors needing charging using adaptive threshold
    to_charge = get_sensors_needing_charging(sensors, mc, adaptive_threshold=True)
    
    if not to_charge:
        print("No sensor needs charging right now.")
        return False

    # 3. Select the highest priority sensor (first in list)
    target = to_charge[0]
    
    # 4. Move to sensor
    if not mc.move_to(target.x, target.y):
        print("MC does not have enough energy to move.")
        return False

    # 5. Charge all sensors within radius
    charged_nodes, total_energy = mc.charge_nodes_in_radius(sensors)
    
    if charged_nodes:
        print(f"Charged {len(charged_nodes)} sensors with total {total_energy:.2f} J")
        print(f"Sensor IDs charged: {charged_nodes}")
    else:
        print("No sensors were charged in this step.")

    return True

def run_adaptive_continuous_charging(num_steps=15, time_step=3, num_sensors=50, path_segments=8):
    """
    Run a simulation with adaptive charging threshold and continuous charging while moving.
    """
    from environment import initialize_environment
    from continuous_charging import move_and_charge_along_path
    from enhanced_visualization import visualize_sensor_priorities
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize environment
    sensors, mc = initialize_environment(num_sensors=num_sensors)
    
    # Set varied energy levels and consumption rates
    for i, s in enumerate(sensors):
        if i % 5 == 0:  # 20% very low
            s.energy = 0.05 * SENSOR_CAPACITY
        elif i % 5 == 1:  # 20% low
            s.energy = 0.2 * SENSOR_CAPACITY
        elif i % 5 == 2:  # 20% medium
            s.energy = 0.5 * SENSOR_CAPACITY
            
        # Vary consumption rates to create different priorities
        if i % 3 == 0:  # High consumption
            s.consumption_rate = random.uniform(0.00008, 0.0001)
        elif i % 3 == 1:  # Medium consumption
            s.consumption_rate = random.uniform(0.00004, 0.00007)
        else:  # Low consumption
            s.consumption_rate = random.uniform(0.00001, 0.00003)
    
    # Path tracking
    path_x = [mc.x]
    path_y = [mc.y]
    path_history = [(mc.x, mc.y)]  # Initialize path_history as list of tuples
    charging_events = []
    total_energy_transferred = 0
    
    # Run simulation
    for step in range(num_steps):
        print(f"--- Step {step+1}/{num_steps} --- MC Energy: {mc.energy:.1f}J")
        
        # Get sensors needing charging using adaptive threshold
        to_charge = get_sensors_needing_charging(sensors, mc, adaptive_threshold=True)
        
        if not to_charge:
            # Let energy deplete a bit
            for s in sensors:
                s.update_energy(time_step * 2)
            print("No sensors need charging right now.")
            continue
        
        # Show priorities of top 3 sensors
        print("Top priority sensors:")
        for i, sensor in enumerate(to_charge[:3]):
            priority = calculate_adaptive_priority(sensor, mc)
            energy_pct = sensor.energy / sensor.capacity * 100
            print(f"  {i+1}. Sensor {sensor.id}: Priority={priority:.2f}, Energy={energy_pct:.1f}%, " +
                  f"Rate={sensor.consumption_rate:.2f}J/s, " +
                  f"Distance={np.linalg.norm([mc.x - sensor.x, mc.y - sensor.y]):.1f}m")
        
        # Select highest priority sensor
        target = to_charge[0]
        
        # Save current position
        start_point = (mc.x, mc.y)
        end_point = (target.x, target.y)
        
        # Move and charge continuously
        energy_transferred, charged_node_ids, path_segment_x, path_segment_y = move_and_charge_along_path(
            mc, start_point, end_point, sensors, path_segments=path_segments
        )
        
        # Update tracking
        path_x.extend(path_segment_x[1:])
        path_y.extend(path_segment_y[1:])
        
        # Update path_history with new path segments
        for i in range(1, len(path_segment_x)):  # Skip first point to avoid duplication
            path_history.append((path_segment_x[i], path_segment_y[i]))
        
        total_energy_transferred += energy_transferred
        
        if charged_node_ids:
            print(f"Charged {len(charged_node_ids)} sensors with {energy_transferred:.1f}J while moving")
            charging_events.append((mc.x, mc.y, step+1))
        
        # Update energy of all sensors
        for s in sensors:
            s.update_energy(time_step)
        
        # Visualize priorities - NOW PASSING THE PATH HISTORY
        print("Visualizing sensor priorities...")
        visualize_sensor_priorities(sensors, mc, path_history)
    
    # Show final stats
    dead_count = sum(1 for s in sensors if s.dead)
    print(f"\nSimulation complete after {num_steps} steps:")
    print(f"Total energy transferred: {total_energy_transferred:.1f}J")
    print(f"Sensors dead: {dead_count} out of {num_sensors}")
    print(f"MC remaining energy: {mc.energy:.1f}J")


def run_adaptive_continuous_charging_with_zones(num_steps=10, time_step=3, num_sensors=40, path_segments=8):
    """
    Run a simulation with adaptive charging threshold, continuous charging while moving,
    and zone-based charging efficiency.
    """
    # Import necessary modules
    import random
    import numpy as np
    from environment import initialize_environment
    from continuous_charging import move_and_charge_along_path
    from enhanced_visualization import visualize_sensor_priorities_with_zones
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize environment
    sensors, mc = initialize_environment(num_sensors=num_sensors)
    
    # Initialize path history
    path_history = [(mc.x, mc.y)]
    
    # Set varied energy levels and consumption rates
    for i, s in enumerate(sensors):
        if i % 5 == 0:  # 20% very low
            s.energy = 0.05 * SENSOR_CAPACITY
        elif i % 5 == 1:  # 20% low
            s.energy = 0.2 * SENSOR_CAPACITY
        elif i % 5 == 2:  # 20% medium
            s.energy = 0.5 * SENSOR_CAPACITY
            
        # Vary consumption rates to create different priorities
        if i % 3 == 0:  # High consumption
            s.consumption_rate = random.uniform(0.00008, 0.0001)
        elif i % 3 == 1:  # Medium consumption
            s.consumption_rate = random.uniform(0.00004, 0.00007)
        else:  # Low consumption
            s.consumption_rate = random.uniform(0.00001, 0.00003)
    
    # Path tracking
    total_energy_transferred = 0
    
    # Visualize initial state
    print("Initial network state with charging zones:")
    visualize_sensor_priorities_with_zones(sensors, mc, path_history)
    
    # Run simulation
    for step in range(num_steps):
        print(f"--- Step {step+1}/{num_steps} --- MC Energy: {mc.energy:.1f}J")
        
        # Get sensors needing charging using adaptive threshold
        to_charge = get_sensors_needing_charging(sensors, mc, adaptive_threshold=True)
        
        if not to_charge:
            # Let energy deplete a bit
            for s in sensors:
                s.update_energy(time_step * 2)
            print("No sensors need charging right now.")
            continue
        
        # Show priorities of top 3 sensors
        print("Top priority sensors:")
        for i, sensor in enumerate(to_charge[:3]):
            priority = calculate_adaptive_priority(sensor, mc)
            energy_pct = sensor.energy / sensor.capacity * 100
            print(f"  {i+1}. Sensor {sensor.id}: Priority={priority:.2f}, Energy={energy_pct:.1f}%, " +
                  f"Rate={sensor.consumption_rate:.2f}J/s, " +
                  f"Distance={np.linalg.norm([mc.x - sensor.x, mc.y - sensor.y]):.1f}m")
        
        # Select highest priority sensor
        target = to_charge[0]
        
        # Save current position
        start_point = (mc.x, mc.y)
        end_point = (target.x, target.y)
        
        # Move and charge continuously
        energy_transferred, charged_node_ids, path_segment_x, path_segment_y = move_and_charge_along_path(
            mc, start_point, end_point, sensors, path_segments=path_segments
        )
        
        # Update path history with all segments
        for x, y in zip(path_segment_x, path_segment_y):
            path_history.append((x, y))
            
        total_energy_transferred += energy_transferred
        
        if charged_node_ids:
            print(f"Charged {len(charged_node_ids)} sensors with {energy_transferred:.1f}J while moving")
            print(f"Charged sensor IDs: {charged_node_ids}")
        
        # Update energy of all sensors
        for s in sensors:
            s.update_energy(time_step)
        
        # Visualize current state with path history and charging zones
        print(f"Visualizing network state with charging zones after step {step+1}...")
        visualize_sensor_priorities_with_zones(sensors, mc, path_history)
    
    # Show final stats
    dead_count = sum(1 for s in sensors if s.dead)
    print(f"\nSimulation complete after {num_steps} steps:")
    print(f"Total energy transferred: {total_energy_transferred:.1f}J")
    print(f"Sensors dead: {dead_count} out of {num_sensors}")
    print(f"MC remaining energy: {mc.energy:.1f}J")
    print(f"Total path points: {len(path_history)}")
    
    return sensors, mc, path_history

def find_optimal_charging_position(mc, sensors, candidate_sensors=None, grid_size=30):
    """
    Find the optimal position for the charger to maximize charging efficiency 
    for multiple sensors.
    """
    if candidate_sensors is None:
        # Consider sensors with lower energy threshold to focus on needier sensors
        candidate_sensors = [s for s in sensors if not s.dead and s.energy < 0.5 * s.capacity]
    
    if not candidate_sensors:
        return (mc.x, mc.y), [], 0  # No sensors need charging
    
    # Define the search area - expand it slightly to find better positions
    min_x = max(0, min(s.x for s in candidate_sensors) - CHARGING_RADIUS*1.2)
    max_x = min(AREA_WIDTH, max(s.x for s in candidate_sensors) + CHARGING_RADIUS*1.2)
    min_y = max(0, min(s.y for s in candidate_sensors) - CHARGING_RADIUS*1.2)
    max_y = min(AREA_HEIGHT, max(s.y for s in candidate_sensors) + CHARGING_RADIUS*1.2)
    
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
            
            # Reduce movement penalty to make movement less prohibitive
            distance_to_move = np.linalg.norm([mc.x - pos_x, mc.y - pos_y])
            energy_cost = distance_to_move * MOVEMENT_COST_PER_M
            movement_penalty = min(0.3, energy_cost / 8000)  # Much less severe penalty
            
            # Apply a multiplier for multi-sensor coverage to prefer positions that cover more sensors
            coverage_bonus = 1.0 + min(1.0, len(covered) * 0.2)  # +10% per sensor up to 50%
            
            # Adjusted score with reduced movement penalty and coverage bonus
            adjusted_score = score * coverage_bonus * (1 - 0.1 * movement_penalty)
            
            # Require a minimum of 2 sensors or significant score improvement to change position
            min_sensors_required = 2
            if (len(covered) >= min_sensors_required and adjusted_score > best_score) or \
               (adjusted_score > best_score * 1.5):  # Only move if significant improvement
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
            # Calculate zone-based efficiency
            distance_ratio = distance / CHARGING_RADIUS
            
            if distance_ratio <= 0.4:
                efficiency = 0.7  # Inner zone: 70% efficiency
                zone_weight = 1.0  # Full weight
            elif distance_ratio <= 0.7:
                efficiency = 0.5  # Middle zone: 50% efficiency
                zone_weight = 0.8  # Slightly reduced weight
            else:
                efficiency = 0.3  # Outer zone: 30% efficiency
                zone_weight = 0.5  # Half weight
            
            # Revised priority calculation - puts more emphasis on critical sensors
            energy_deficit = 1 - (sensor.energy / sensor.capacity)
            consumption_factor = min(sensor.consumption_rate / 0.0001, 1.0)
            
            # Add urgency factor
            urgency = max(0, 1 - (sensor.energy / (sensor.consumption_rate * 1000)))
            urgency_boost = 1.0 + min(2.0, urgency)  # Up to 3x multiplier for urgent sensors
            
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
        coverage_multiplier = len(covered_sensors) ** 1.2  # Superlinear scaling
    
    return total_score * coverage_multiplier, covered_sensors

def run_optimal_position_simulation(num_steps=10, time_step=3, num_sensors=40):
    """
    Run a simulation using the optimal position strategy for charging,
    with zone-based efficiency visualization and automatic return to base when energy is low
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize environment
    sensors, mc = initialize_environment(num_sensors=num_sensors)
    
    # Define base station coordinates (consistent with visualization)
    base_station_x, base_station_y = 300, 300
    
    # Define energy threshold for returning to base (15% of capacity)
    return_to_base_threshold = 0.15 * MC_CAPACITY
    
    # Initialize path history and evaluation metrics
    path_history = [(mc.x, mc.y)]
    total_energy_transferred = 0
    total_movement_energy = 0
    sensors_requested_charging = set()
    sensors_received_charging = set()
    
    # Set varied energy levels and consumption rates for interesting scenario
    for i, s in enumerate(sensors):
        if i % 5 == 0:  # 20% very low
            s.energy = 0.05 * SENSOR_CAPACITY
        elif i % 5 == 1:  # 20% low
            s.energy = 0.15 * SENSOR_CAPACITY
        elif i % 5 == 2:  # 20% medium-low
            s.energy = 0.30 * SENSOR_CAPACITY
        elif i % 5 == 3:  # 20% medium
            s.energy = 0.45 * SENSOR_CAPACITY
        # else: keep random levels for the rest
            
        # Vary consumption rates with more differential
        if i % 3 == 0:  # High consumption
            s.consumption_rate = random.uniform(0.00008, 0.0001)
        elif i % 3 == 1:  # Medium consumption
            s.consumption_rate = random.uniform(0.00004, 0.00008)
        else:  # Low consumption
            s.consumption_rate = random.uniform(0.00001, 0.00003)
    
    # Run simulation
    for step in range(num_steps):
        print(f"--- Step {step+1}/{num_steps} --- MC Energy: {mc.energy:.1f}J")
        
        # Check if MC needs to return to base station
        if mc.energy < return_to_base_threshold:
            print(f"MC energy low ({mc.energy:.1f}J)! Returning to base station...")
            
            # Calculate movement energy to base
            distance_to_base = np.linalg.norm([mc.x - base_station_x, mc.y - base_station_y])
            energy_to_base = distance_to_base * MOVEMENT_COST_PER_M
            
            if mc.energy < energy_to_base:
                print("WARNING: MC doesn't have enough energy to return to base!")
                # Move as far as possible toward base
                direction_x = base_station_x - mc.x
                direction_y = base_station_y - mc.y
                direction_norm = np.linalg.norm([direction_x, direction_y])
                max_distance = mc.energy / MOVEMENT_COST_PER_M
                
                # Calculate how far it can go
                ratio = max_distance / direction_norm
                new_x = mc.x + direction_x * ratio
                new_y = mc.y + direction_y * ratio
                
                # Move partially toward base
                mc.move_to(new_x, new_y)
                print(f"MC depleted all energy. Final position: ({mc.x:.1f}, {mc.y:.1f})")
                break
            
            # Move to base station
            mc.move_to(base_station_x, base_station_y)
            total_movement_energy += energy_to_base
            
            # Add base station visit to path history
            path_history.append((base_station_x, base_station_y))
            
            # Recharge instantly to full capacity
            previous_energy = mc.energy
            mc.energy = MC_CAPACITY
            print(f"MC recharged at base station: {previous_energy:.1f}J → {mc.energy:.1f}J")
        
        # Continue with regular charging logic
        to_charge = get_sensors_needing_charging(sensors, mc, adaptive_threshold=True)
        
        # Track sensors that requested charging
        for sensor in to_charge:
            sensors_requested_charging.add(sensor.id)
        
        if not to_charge:
            # Let energy deplete a bit to create need
            for s in sensors:
                s.update_energy(time_step * 2)
            print("No sensors need charging right now.")
            continue
        
        # Find the optimal charging position
        print(f"Finding optimal position to charge {len(to_charge)} candidate sensors...")
        optimal_pos, covered_sensors, efficiency_score = find_optimal_charging_position(mc, sensors, to_charge)
        
        # Visualize the optimal position before moving
        print(f"Found optimal position at {optimal_pos} with efficiency score {efficiency_score:.2f}")
        print(f"This position would cover {len(covered_sensors)} sensors: {covered_sensors}")
        visualize_optimal_position(sensors, mc, optimal_pos, covered_sensors, path_history)
        
        # Calculate movement energy
        distance_to_move = np.linalg.norm([mc.x - optimal_pos[0], mc.y - optimal_pos[1]])
        movement_energy = distance_to_move * MOVEMENT_COST_PER_M
        total_movement_energy += movement_energy
        
        # Check if movement would deplete energy below return threshold
        if mc.energy - movement_energy < return_to_base_threshold:
            print(f"Warning: Moving would deplete energy below return threshold.")
            print(f"Returning to base station first...")
            
            # Move to base and recharge
            distance_to_base = np.linalg.norm([mc.x - base_station_x, mc.y - base_station_y])
            energy_to_base = distance_to_base * MOVEMENT_COST_PER_M
            
            # Move to base
            mc.move_to(base_station_x, base_station_y)
            total_movement_energy += energy_to_base
            
            # Add base station visit to path history
            path_history.append((base_station_x, base_station_y))
            
            # Recharge instantly
            previous_energy = mc.energy
            mc.energy = MC_CAPACITY
            print(f"MC recharged at base station: {previous_energy:.1f}J → {mc.energy:.1f}J")
            
            # Now try moving to optimal position again
            mc.move_to(optimal_pos[0], optimal_pos[1])
            total_movement_energy += movement_energy
        else:
            # Move to the optimal position
            if not mc.move_to(optimal_pos[0], optimal_pos[1]):
                print("MC does not have enough energy to move to optimal position.")
                break
        
        # Add to path history
        path_history.append((mc.x, mc.y))
        
        # Charge all sensors within radius from the optimal position
        charged_nodes, total_energy = mc.charge_nodes_in_radius(sensors)
        total_energy_transferred += total_energy
        
        # Track which sensors received charging
        for sensor_id, _, _ in charged_nodes:
            sensors_received_charging.add(sensor_id)
        
        if charged_nodes:
            print(f"Charged {len(charged_nodes)} sensors with total {total_energy:.2f}J")
            print(f"Sensor IDs charged: {charged_nodes}")
        else:
            print("No sensors were charged in this step.")
        
        # Update energy of all sensors
        for s in sensors:
            s.update_energy(time_step)
        
        # Visualize after charging
        print(f"Visualizing network state with charging zones after step {step+1}...")
        visualize_optimal_position(sensors, mc, (mc.x, mc.y), [s_id for s_id, _, _ in charged_nodes], path_history)
    
    # Final stats
    dead_count = sum(1 for s in sensors if s.dead)
    print(f"\nSimulation complete after {num_steps} steps:")
    print(f"Total energy transferred: {total_energy_transferred:.1f}J")
    print(f"Sensors dead: {dead_count} out of {len(sensors)}")
    print(f"MC remaining energy: {mc.energy:.1f}J")
    
    # Calculate and display evaluation metrics
    print("\n--- Evaluation Metrics ---")
    
    # Survival Rate
    if len(sensors_requested_charging) > 0:
        survival_rate = len(sensors_received_charging) / len(sensors_requested_charging)
        print(f"Survival Rate: {survival_rate:.2f} ({len(sensors_received_charging)}/{len(sensors_requested_charging)})")
    else:
        print("Survival Rate: N/A (no sensors requested charging)")
    
    # Energy Efficiency
    total_energy_cost = total_movement_energy + total_energy_transferred
    if total_energy_cost > 0:
        energy_efficiency = total_energy_transferred / total_energy_cost
        print(f"Energy Efficiency: {energy_efficiency:.2f}")
        print(f"  - Energy received by sensors: {total_energy_transferred:.1f}J")
        print(f"  - Movement energy cost: {total_movement_energy:.1f}J")
        print(f"  - Total energy used: {total_energy_cost:.1f}J")
    else:
        print("Energy Efficiency: N/A (no energy was expended)")
    
    # Display unique sensors charged
    print(f"Unique sensors that requested charging: {sorted(list(sensors_requested_charging))}")
    print(f"Unique sensors that received charging: {sorted(list(sensors_received_charging))}")
    
    return sensors, mc, path_history