import numpy as np
import random
from config import AREA_WIDTH, AREA_HEIGHT, CHARGING_RADIUS, THRESHOLD_RATIO, MOVEMENT_COST_PER_M

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
    max_consumption_rate = 2.0
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
            s.energy = 0.05 * s.capacity
        elif i % 5 == 1:  # 20% low
            s.energy = 0.2 * s.capacity
        elif i % 5 == 2:  # 20% medium
            s.energy = 0.5 * s.capacity
            
        # Vary consumption rates to create different priorities
        if i % 3 == 0:  # High consumption
            s.consumption_rate = random.uniform(0.8, 1.5)
        elif i % 3 == 1:  # Medium consumption
            s.consumption_rate = random.uniform(0.4, 0.7)
        else:  # Low consumption
            s.consumption_rate = random.uniform(0.1, 0.3)
    
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
            s.energy = 0.05 * s.capacity
        elif i % 5 == 1:  # 20% low
            s.energy = 0.2 * s.capacity
        elif i % 5 == 2:  # 20% medium
            s.energy = 0.5 * s.capacity
            
        # Vary consumption rates to create different priorities
        if i % 3 == 0:  # High consumption
            s.consumption_rate = random.uniform(0.8, 1.5)
        elif i % 3 == 1:  # Medium consumption
            s.consumption_rate = random.uniform(0.4, 0.7)
        else:  # Low consumption
            s.consumption_rate = random.uniform(0.1, 0.3)
    
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

# Add these functions at the end of the file

def find_optimal_charging_position(mc, sensors, candidate_sensors=None, grid_size=20):
    """
    Find the optimal position for the charger to maximize charging efficiency 
    for multiple sensors.
    
    Parameters:
    - mc: Mobile charger object
    - sensors: List of all sensor nodes
    - candidate_sensors: List of sensors to consider for charging (defaults to all sensors below 80% capacity)
    - grid_size: Resolution of the search grid
    
    Returns:
    - best_pos: Tuple (x, y) representing the optimal position
    - covered_sensors: List of sensor IDs that would be covered from this position
    - efficiency_score: The calculated efficiency score of this position
    """
    if candidate_sensors is None:
        # Default: consider all sensors below 80% charge that aren't dead
        candidate_sensors = [s for s in sensors if not s.dead and s.energy < 0.8 * s.capacity]
    
    if not candidate_sensors:
        return (mc.x, mc.y), [], 0  # No sensors need charging
    
    # Define the search area - limit to where the sensors are
    min_x = max(0, min(s.x for s in candidate_sensors) - CHARGING_RADIUS)
    max_x = min(AREA_WIDTH, max(s.x for s in candidate_sensors) + CHARGING_RADIUS)
    min_y = max(0, min(s.y for s in candidate_sensors) - CHARGING_RADIUS)
    max_y = min(AREA_HEIGHT, max(s.y for s in candidate_sensors) + CHARGING_RADIUS)
    
    # Create a grid of candidate positions
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
            
            # Calculate movement cost penalty
            distance_to_move = np.linalg.norm([mc.x - pos_x, mc.y - pos_y])
            energy_cost = distance_to_move * MOVEMENT_COST_PER_M
            movement_penalty = min(1.0, energy_cost / 1000)  # Cap penalty at 1.0
            
            # Adjust score based on movement cost (less penalty for higher scores)
            adjusted_score = score * (1 - 0.2 * movement_penalty)
            
            if adjusted_score > best_score and len(covered) > 0:
                best_score = adjusted_score
                best_pos = (pos_x, pos_y)
                best_covered = covered
    
    return best_pos, best_covered, best_score

def evaluate_charging_position(x, y, sensors):
    """
    Evaluate a potential charging position based on:
    1. Number of sensors that can be charged
    2. Charging efficiency (based on distance)
    3. Priority of sensors (energy level and consumption rate)
    
    Returns:
    - score: Overall score of this position
    - covered_sensors: List of sensor IDs that would be covered
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
            
            # Calculate priority based on energy status
            energy_deficit = 1 - (sensor.energy / sensor.capacity)
            consumption_factor = min(sensor.consumption_rate / 2.0, 1.0)  # Normalize to 0-1
            
            # Combine factors for this sensor's contribution to the position score
            sensor_score = (
                efficiency * 
                zone_weight * 
                (energy_deficit * 0.7 + consumption_factor * 0.3) * 
                (sensor.capacity - sensor.energy)  # Absolute energy need
            )
            
            total_score += sensor_score
            covered_sensors.append(sensor.id)
    
    return total_score, covered_sensors

def run_optimal_position_simulation(num_steps=10, time_step=3, num_sensors=40):
    """
    Run a simulation using the optimal position strategy for charging
    """
    # Import necessary modules
    import random
    import numpy as np
    from environment import initialize_environment
    from sensor_node import SensorNode
    from enhanced_visualization import visualize_sensor_priorities, visualize_optimal_position
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize environment
    sensors, mc = initialize_environment(num_sensors=num_sensors)
    
    # Initialize path history
    path_history = [(mc.x, mc.y)]
    
    # Set varied energy levels and consumption rates for interesting scenario
    for i, s in enumerate(sensors):
        if i % 5 == 0:  # 20% very low
            s.energy = 0.05 * s.capacity
        elif i % 5 == 1:  # 20% low
            s.energy = 0.2 * s.capacity
        elif i % 5 == 2:  # 20% medium
            s.energy = 0.5 * s.capacity
            
        # Vary consumption rates to create different priorities
        if i % 3 == 0:  # High consumption
            s.consumption_rate = random.uniform(0.8, 1.5)
        elif i % 3 == 1:  # Medium consumption
            s.consumption_rate = random.uniform(0.4, 0.7)
        else:  # Low consumption
            s.consumption_rate = random.uniform(0.1, 0.3)
    
    # Create some clusters of sensors for interesting optimal positions
    # Add 10 more sensors in 2 clusters
    for i in range(5):
        cluster1_x = random.uniform(100, 150)
        cluster1_y = random.uniform(100, 150)
        sensors.append(SensorNode(num_sensors + i, cluster1_x, cluster1_y))
        
        cluster2_x = random.uniform(400, 450)
        cluster2_y = random.uniform(400, 450)
        sensors.append(SensorNode(num_sensors + i + 5, cluster2_x, cluster2_y))
    
    # Total energy transferred
    total_energy_transferred = 0
    
    # Visualize initial state
    print("Initial network state:")
    visualize_sensor_priorities(sensors, mc, path_history)
    
    # Run simulation
    for step in range(num_steps):
        print(f"--- Step {step+1}/{num_steps} --- MC Energy: {mc.energy:.1f}J")
        
        # Find sensors that need charging using adaptive threshold
        to_charge = get_sensors_needing_charging(sensors, mc, adaptive_threshold=True)
        
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
        
        # Move to the optimal position
        if not mc.move_to(optimal_pos[0], optimal_pos[1]):
            print("MC does not have enough energy to move to optimal position.")
            break
        
        # Add to path history
        path_history.append((mc.x, mc.y))
        
        # Charge all sensors within radius from the optimal position
        charged_nodes, total_energy = mc.charge_nodes_in_radius(sensors)
        total_energy_transferred += total_energy
        
        if charged_nodes:
            print(f"Charged {len(charged_nodes)} sensors with total {total_energy:.2f}J")
            print(f"Sensor IDs charged: {charged_nodes}")
        else:
            print("No sensors were charged in this step.")
        
        # Update energy of all sensors
        for s in sensors:
            s.update_energy(time_step)
        
        # Visualize after charging
        print(f"Visualizing network state after step {step+1}...")
        visualize_sensor_priorities(sensors, mc, path_history)
    
    # Final stats
    dead_count = sum(1 for s in sensors if s.dead)
    print(f"\nSimulation complete after {num_steps} steps:")
    print(f"Total energy transferred: {total_energy_transferred:.1f}J")
    print(f"Sensors dead: {dead_count} out of {len(sensors)}")
    print(f"MC remaining energy: {mc.energy:.1f}J")
    
    return sensors, mc, path_history