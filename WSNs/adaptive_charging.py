import numpy as np
import random
from config import AREA_WIDTH, AREA_HEIGHT, CHARGING_RADIUS, THRESHOLD_RATIO

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