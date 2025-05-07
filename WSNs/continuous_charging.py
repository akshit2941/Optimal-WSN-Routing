import numpy as np
import matplotlib.pyplot as plt
from config import CHARGING_RADIUS, MOVEMENT_COST_PER_M
from visualization import visualize_charging_heatmap

def move_and_charge_along_path(mc, start_point, end_point, sensors, path_segments=10):
    """
    Moves the mobile charger from start to end point in small increments,
    charging sensors within radius at each step along the path.
    
    Args:
        mc: The mobile charger object
        start_point: (x,y) starting coordinates
        end_point: (x,y) ending coordinates
        sensors: List of sensor nodes
        path_segments: Number of segments to divide the path into
    
    Returns:
        total_energy_transferred: Total energy transferred during movement
        all_charged_nodes: List of all node IDs that received charge
        path_x, path_y: Lists of coordinates visited
    """
    start_x, start_y = start_point
    end_x, end_y = end_point
    
    # Calculate total distance and movement cost
    total_distance = np.linalg.norm([end_x - start_x, end_y - start_y])
    movement_cost = total_distance * MOVEMENT_COST_PER_M
    
    if mc.energy < movement_cost:
        return 0, [], [mc.x], [mc.y]  # Not enough energy to complete move
    
    # Deduct movement cost upfront
    mc.energy -= movement_cost
    
    # Create path points (divide path into segments)
    path_x = []
    path_y = []
    charging_events = []
    
    # Statistics tracking
    total_energy_transferred = 0
    all_charged_nodes = []
    
    # Move along the path in increments, charging at each step
    for i in range(path_segments + 1):  # +1 to include end point
        # Calculate position at this segment
        progress = i / path_segments
        current_x = start_x + (end_x - start_x) * progress
        current_y = start_y + (end_y - start_y) * progress
        
        # Update charger position
        mc.x, mc.y = current_x, current_y
        path_x.append(current_x)
        path_y.append(current_y)
        
        # Charge sensors within radius at reduced efficiency while moving
        charged_nodes, energy_transferred = charge_while_moving(mc, sensors)
        
        if charged_nodes:
            total_energy_transferred += energy_transferred
            all_charged_nodes.extend([node_id for node_id in charged_nodes if node_id not in all_charged_nodes])
            
    return total_energy_transferred, all_charged_nodes, path_x, path_y


def charge_while_moving(mc, sensors):
    """Charge all sensors within radius while the charger is moving"""
    charged_nodes = []
    total_energy_transferred = 0
    
    if mc.energy <= 0:
        return charged_nodes, 0
    
    # Apply a movement efficiency factor (reduced efficiency while moving)
    # We can adjust this factor based on how effective we want charging-while-moving to be
    movement_efficiency_factor = 0.7  # 70% as effective as stationary charging
    
    # Check all sensors - this ensures we charge EVERY sensor in range
    for sensor in sensors:
        if sensor.dead:
            continue
            
        distance = np.linalg.norm([mc.x - sensor.x, mc.y - sensor.y])
        # Check if sensor is within charging radius, regardless of energy level
        if distance <= CHARGING_RADIUS:
            needed = sensor.capacity - sensor.energy
            if needed <= 0:
                continue  # Skip fully charged sensors
                
            # Calculate efficiency based on distance + movement penalty
            base_efficiency = max(0.3, 1 - (distance / CHARGING_RADIUS) * 0.7)
            moving_efficiency = base_efficiency * movement_efficiency_factor
            
            # Calculate energy to transfer (cap at needed amount or available energy)
            max_transfer = min(needed, mc.energy)
            energy_to_transfer = max_transfer * moving_efficiency
            
            if energy_to_transfer > 0:
                sensor.charge(energy_to_transfer)
                mc.energy -= energy_to_transfer
                total_energy_transferred += energy_to_transfer
                charged_nodes.append(sensor.id)
                
                if mc.energy <= 0:
                    break
    
    return charged_nodes, total_energy_transferred