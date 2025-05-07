import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import random
from config import AREA_WIDTH, AREA_HEIGHT, CHARGING_RADIUS, MC_CAPACITY
from environment import initialize_environment
from continuous_charging import move_and_charge_along_path
from visualization import visualize_charging_heatmap
from adaptive_charging import calculate_adaptive_priority

def visualize_path_step(sensors, mc, path_x, path_y, charging_events, step, final=False):
    """Helper function to visualize a single step in the charger's path"""
    plt.figure(figsize=(12, 10))
    
    # Plot environment basics
    alive_sensors = [s for s in sensors if not s.dead]
    dead_sensors = [s for s in sensors if s.dead]
    
    # Plot sensors with color intensity based on energy level
    for s in alive_sensors:
        energy_ratio = min(max(s.energy / s.capacity, 0.0), 1.0)  # Clamp between 0 and 1
        # Use a color scale from red (low energy) to green (full energy)
        color = [(1-energy_ratio), energy_ratio, 0]  # Make as a list, not tuple
        plt.scatter(s.x, s.y, c=[color], s=80, edgecolors='black', zorder=2)
        plt.annotate(f"{s.id}", (s.x, s.y), xytext=(-5, -5), 
                    textcoords="offset points", ha='center', fontsize=8)
    
    # Plot dead sensors
    if dead_sensors:
        plt.scatter([s.x for s in dead_sensors], [s.y for s in dead_sensors], 
                   c='red', marker='x', s=100, label='Dead Sensors', zorder=1)
    
    # Plot base station
    plt.scatter(300, 300, c='black', marker='s', s=100, label='Base Station')
    
    # Draw the current charging radius
    circle = plt.Circle((mc.x, mc.y), CHARGING_RADIUS, color='blue', 
                        fill=False, linestyle='--', alpha=0.5)
    plt.gcf().gca().add_artist(circle)
    
    # Plot the mobile charger's path with arrows to show direction
    if len(path_x) > 1:
        for i in range(len(path_x)-1):
            # Draw line segment with arrow
            dx = path_x[i+1] - path_x[i]
            dy = path_y[i+1] - path_y[i]
            plt.arrow(path_x[i], path_y[i], dx*0.8, dy*0.8, 
                     head_width=15, head_length=15, fc='blue', ec='blue', zorder=3, 
                     length_includes_head=True, alpha=0.7)
    
    # Highlight charging events along the path
    for cx, cy, event_step in charging_events:
        if event_step <= step:  # Only show events that have happened
            plt.scatter(cx, cy, c='yellow', edgecolors='orange', s=200, 
                       marker='*', zorder=4, alpha=0.8)
            plt.annotate(f"Charging\nStep {event_step}", (cx, cy), xytext=(10, 10),
                        textcoords="offset points", 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    
    # Plot current position of mobile charger
    plt.scatter(mc.x, mc.y, c='blue', marker='X', s=150, label='Mobile Charger')
    
    # Add a custom legend for energy levels
    legend_elements = [
        Patch(facecolor=(0,1,0), edgecolor='black', label='Full Energy'),
        Patch(facecolor=(0.5,0.5,0), edgecolor='black', label='Half Energy'),
        Patch(facecolor=(1,0,0), edgecolor='black', label='Low Energy'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='blue', markersize=10, label='Mobile Charger'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow', markersize=10, label='Charging Event'),
        Line2D([0], [0], marker='x', color='red', lw=0, label='Dead Sensor'),
        Line2D([0], [0], color='blue', lw=2, label='Charger Path')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Set title and grid
    title = "Mobile Charger Path and Charging Visualization"
    if final:
        title += f" (Complete Path After {step} Steps)"
    else:
        title += f" (Step {step} of {step})"
    plt.title(title)
    
    # Add MC energy status
    plt.annotate(f"MC Energy: {mc.energy:.0f}J", xy=(0.02, 0.98), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                va='top')
    
    # Add charging beams for sensors in range
    for s in alive_sensors:
        distance = np.linalg.norm([mc.x - s.x, mc.y - s.y])
        if distance <= CHARGING_RADIUS:
            # Calculate charging efficiency
            efficiency = max(0.3, 1 - (distance / CHARGING_RADIUS) * 0.7)
            # Draw charging beam with thickness/opacity based on efficiency
            plt.plot([mc.x, s.x], [mc.y, s.y], 'y-', 
                     alpha=efficiency*0.8, 
                     linewidth=2*efficiency,
                     zorder=1)
            
            # Add efficiency label
            plt.annotate(f"{efficiency*100:.0f}%", 
                        ((mc.x + s.x)/2, (mc.y + s.y)/2), 
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center',
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7),
                        fontsize=8)
    
    plt.grid(True)
    plt.xlim(0, AREA_WIDTH)
    plt.ylim(0, AREA_HEIGHT)
    plt.tight_layout()
    plt.show()

def visualize_charger_path(num_steps=15, time_step=10, show_step_by_step=True):
    """
    Visualizes the path of the mobile charger and how sensors get charged over time.
    
    Args:
        num_steps: Number of simulation steps to run
        time_step: Time between each step (impacts energy consumption)
        show_step_by_step: If True, shows each step; if False, only shows final state with full path
    """
    # Initialize environment with more sensors for better visualization
    sensors, mc = initialize_environment(num_sensors=30)
    
    # Create lists to store path and charging information
    path_x = [mc.x]
    path_y = [mc.y]
    charging_events = []  # List to store charging positions
    
    # Track initial energy states
    initial_energy = {s.id: s.energy for s in sensors}
    
    # Run simulation for specified steps
    for step in range(num_steps):
        print(f"--- Step {step+1} ---")
        
        # Filter sensors needing charging
        to_charge = [s for s in sensors if s.needs_charging()]
        if not to_charge:
            print("No sensor needs charging right now.")
            break
            
        # Select next sensor using greedy strategy (closest one)
        to_charge.sort(key=lambda s: np.linalg.norm([mc.x - s.x, mc.y - s.y]))
        target = to_charge[0]
        
        # Move to sensor
        if not mc.move_to(target.x, target.y):
            print("MC does not have enough energy to move.")
            break
            
        # Add new position to path
        path_x.append(mc.x)
        path_y.append(mc.y)
        
        # Charge all sensors within radius
        charged_nodes, total_energy = mc.charge_nodes_in_radius(sensors)
        
        if charged_nodes:
            print(f"Charged {len(charged_nodes)} sensors with total {total_energy:.2f} J")
            charging_events.append((mc.x, mc.y, step+1))
        else:
            print("No sensors were charged in this step.")
            
        # Update energy of all sensors
        for s in sensors:
            s.update_energy(time_step)
            
        # Visualize each step if requested
        if show_step_by_step:
            visualize_path_step(sensors, mc, path_x, path_y, charging_events, step+1)
    
    # Always show final state
    if not show_step_by_step:
        visualize_path_step(sensors, mc, path_x, path_y, charging_events, num_steps, final=True)
        
    # Show energy change summary
    final_energy = {s.id: s.energy for s in sensors}
    print("\nEnergy Change Summary:")
    for s_id in initial_energy:
        initial = initial_energy[s_id]
        final = final_energy[s_id]
        change = final - initial
        status = "CHARGED" if change > 0 else "DEPLETED" if change < 0 else "UNCHANGED"
        print(f"Sensor {s_id}: {initial:.1f}J → {final:.1f}J ({change:+.1f}J) - {status}")

def visualize_continuous_path_step(sensors, mc, all_paths_x, all_paths_y, 
                                  charging_events, step, all_charged_nodes_history):
    """Visualize path with continuous charging markers along the path"""
    plt.figure(figsize=(14, 10))
    
    # Plot environment basics
    alive_sensors = [s for s in sensors if not s.dead]
    dead_sensors = [s for s in sensors if s.dead]
    
    # Plot sensors with color intensity based on energy level
    for s in alive_sensors:
        energy_ratio = min(max(s.energy / s.capacity, 0.0), 1.0)
        color = [(1-energy_ratio), energy_ratio, 0]
        plt.scatter(s.x, s.y, c=[color], s=80, edgecolors='black', zorder=2)
        plt.annotate(f"{s.id}", (s.x, s.y), xytext=(-5, -5), 
                   textcoords="offset points", ha='center', fontsize=8)
    
    # Plot dead sensors
    if dead_sensors:
        plt.scatter([s.x for s in dead_sensors], [s.y for s in dead_sensors], 
                  c='red', marker='x', s=100, label='Dead Sensors', zorder=1)
    
    # Plot base station
    plt.scatter(300, 300, c='black', marker='s', s=100, label='Base Station')
    
    # Draw current charging radius
    circle = plt.Circle((mc.x, mc.y), CHARGING_RADIUS, color='blue', 
                      fill=False, linestyle='--', alpha=0.5)
    plt.gcf().gca().add_artist(circle)
    
    # Plot path segments with gradient coloring to show time progression
    for i, (path_x, path_y) in enumerate(zip(all_paths_x, all_paths_y)):
        # Use color gradient from light to dark blue to show chronology
        color_intensity = 0.3 + 0.7 * (i / len(all_paths_x))
        
        # Plot path with arrows
        for j in range(len(path_x)-1):
            dx = path_x[j+1] - path_x[j]
            dy = path_y[j+1] - path_y[j]
            plt.arrow(path_x[j], path_y[j], dx*0.8, dy*0.8,
                    head_width=8, head_length=8, 
                    fc=(0, 0, color_intensity), ec=(0, 0, color_intensity),
                    zorder=3, length_includes_head=True, alpha=0.7)
        
        # Show nodes charged in this segment
        charged_nodes = all_charged_nodes_history[i]
        if charged_nodes:
            for node_id in charged_nodes:
                # Find the sensor
                for s in sensors:
                    if s.id == node_id:
                        # Draw a connecting line showing charging happened
                        # Find the closest point on path to this sensor
                        distances = [np.linalg.norm([x-s.x, y-s.y]) for x, y in zip(path_x, path_y)]
                        min_dist_idx = distances.index(min(distances))
                        closest_x, closest_y = path_x[min_dist_idx], path_y[min_dist_idx]
                        
                        # Draw a curve showing charging happened
                        plt.plot([closest_x, s.x], [closest_y, s.y], 'y-', 
                               alpha=0.3, linewidth=1.5, zorder=1, linestyle=':')
            
    # Plot charging events
    for cx, cy, event_step in charging_events:
        if event_step <= step:
            plt.scatter(cx, cy, c='yellow', edgecolors='orange', s=200, 
                      marker='*', zorder=4, alpha=0.8)
    
    # Plot current MC position
    plt.scatter(mc.x, mc.y, c='blue', marker='X', s=150, label='Mobile Charger')
    
    # Add MC energy status
    plt.annotate(f"MC Energy: {mc.energy:.0f}J", xy=(0.02, 0.98), xycoords='axes fraction',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
               va='top')
    
    # Add custom legend
    legend_elements = [
        Patch(facecolor=(0,1,0), edgecolor='black', label='Full Energy'),
        Patch(facecolor=(0.5,0.5,0), edgecolor='black', label='Half Energy'),
        Patch(facecolor=(1,0,0), edgecolor='black', label='Low Energy'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='blue', markersize=10, label='Mobile Charger'),
        Line2D([0], [0], color='blue', lw=2, label='Charger Path'),
        Line2D([0], [0], color='yellow', linestyle=':', lw=2, label='Charging While Moving'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow', markersize=10, label='Stopping Point')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Set title and grid
    plt.title(f"Continuous Charging Visualization (Step {step})")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, AREA_WIDTH)
    plt.ylim(0, AREA_HEIGHT)
    plt.tight_layout()
    plt.show()

def visualize_continuous_charging_path(num_steps=20, time_step=3, num_sensors=60, path_segments=5):
    """
    Visualizes a mobile charger moving through the network and charging
    sensors continuously along its path.
    """
    # Set a fixed seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize environment with many sensors
    sensors, mc = initialize_environment(num_sensors=num_sensors)
    
    # Set varied energy levels
    for i, s in enumerate(sensors):
        # Create a mix of energy levels
        if i % 5 == 0:  # 20% very low
            s.energy = 0.05 * s.capacity
        elif i % 5 == 1:  # 20% low
            s.energy = 0.15 * s.capacity
        elif i % 5 == 2:  # 20% medium
            s.energy = 0.4 * s.capacity
        
        # Use very low consumption rates to keep simulation running longer
        s.consumption_rate = random.uniform(0.05, 0.3)
    
    # Path tracking and statistics
    path_x = [mc.x]
    path_y = [mc.y]
    charging_events = []
    total_energy_transferred = 0
    initial_energy = {s.id: s.energy for s in sensors}
    
    # Create animation data storage
    all_paths_x = []
    all_paths_y = []
    all_mc_positions = []
    all_charged_nodes_history = []
    step_energy_values = []
    
    # Main simulation loop
    for step in range(num_steps):
        print(f"--- Step {step+1}/{num_steps} --- MC Energy: {mc.energy:.1f}J")
        
        # Find sensors needing charging
        to_charge = [s for s in sensors if s.needs_charging()]
        if not to_charge:
            # Let energy deplete if no sensors need charging
            for s in sensors:
                s.update_energy(time_step * 2)
            print("No sensors need charging. Letting energy deplete...")
            continue
        
        # Prioritize based on energy level and distance
        def priority_score(sensor):
            distance = np.linalg.norm([mc.x - sensor.x, mc.y - sensor.y])
            energy_ratio = sensor.energy / sensor.capacity
            # Lower energy and shorter distance = higher priority
            return distance * 0.5 + energy_ratio * 100  # Weigh energy level more
        
        # Sort by priority
        to_charge.sort(key=priority_score)
        target = to_charge[0]
        print(f"Moving to sensor {target.id} ({target.energy/target.capacity:.1%} charge)")
        
        # Save current position
        start_point = (mc.x, mc.y)
        end_point = (target.x, target.y)
        
        # Move and charge continuously along path
        energy_transferred, charged_node_ids, path_segment_x, path_segment_y = move_and_charge_along_path(
            mc, start_point, end_point, sensors, path_segments=path_segments
        )
        
        # Update tracking variables
        path_x.extend(path_segment_x[1:])  # Skip first point (already in path_x)
        path_y.extend(path_segment_y[1:])
        total_energy_transferred += energy_transferred
        
        # Store animation data for this step
        all_paths_x.append(path_segment_x)
        all_paths_y.append(path_segment_y)
        all_mc_positions.append((mc.x, mc.y))
        all_charged_nodes_history.append(charged_node_ids)
        
        # Store energy values for all sensors at this step
        step_energy_values.append({s.id: s.energy for s in sensors})
        
        if charged_node_ids:
            print(f"Charged {len(charged_node_ids)} sensors with {energy_transferred:.1f}J while moving")
            charging_events.append((mc.x, mc.y, step+1))
        
        # Regular energy consumption for all sensors
        for s in sensors:
            s.update_energy(time_step)
            
        # Visualize current state
        visualize_continuous_path_step(
            sensors, mc, all_paths_x, all_paths_y, 
            charging_events, step+1, all_charged_nodes_history
        )
    
    # Show final energy summary
    final_energy = {s.id: s.energy for s in sensors}
    charged_count = sum(1 for s_id in initial_energy if final_energy[s_id] > initial_energy[s_id])
    depleted_count = sum(1 for s_id in initial_energy if final_energy[s_id] < initial_energy[s_id])
    
    print("\nEnergy Change Summary:")
    for s_id in initial_energy:
        initial = initial_energy[s_id]
        final = final_energy[s_id]
        change = final - initial
        status = "CHARGED" if change > 0 else "DEPLETED" if change < 0 else "UNCHANGED"
        print(f"Sensor {s_id}: {initial:.1f}J → {final:.1f}J ({change:+.1f}J) - {status}")
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"Total energy transferred: {total_energy_transferred:.1f}J")
    print(f"Final status: Charged: {charged_count}, Depleted: {depleted_count}")
    print(f"MC remaining energy: {mc.energy:.1f}J")
    
    # Show heatmap
    visualize_charging_heatmap(path_x, path_y, charging_events, sensors)

def visualize_enhanced_continuous_path_step(sensors, mc, all_paths_x, all_paths_y, 
                                           charging_events, step, all_charged_nodes_history):
    """Enhanced visualization of a single step of the continuous charging simulation"""
    plt.figure(figsize=(14, 10))
    
    # Plot environment basics
    alive_sensors = [s for s in sensors if not s.dead]
    dead_sensors = [s for s in sensors if s.dead]
    
    # Plot sensors with color intensity based on energy level
    for s in alive_sensors:
        energy_ratio = min(max(s.energy / s.capacity, 0.0), 1.0)  # Clamp between 0 and 1
        # Use a color scale from red (low energy) to green (full energy)
        color = [(1-energy_ratio), energy_ratio, 0]  # RGB
        
        # Highlight sensors that were charged in this step
        was_charged = s.id in all_charged_nodes_history[-1] if all_charged_nodes_history else False
        edge_color = 'cyan' if was_charged else 'black'
        edge_width = 2 if was_charged else 1
        
        # Make size based on energy level for enhanced visualization
        size = 60 + energy_ratio * 40  # Size from 60 to 100 based on energy
        
        plt.scatter(s.x, s.y, c=[color], s=size, edgecolors=edge_color, linewidth=edge_width, zorder=2)
        plt.annotate(f"{s.id}", (s.x, s.y), xytext=(-5, -5), 
                    textcoords="offset points", ha='center', fontsize=8)
    
    # Plot dead sensors
    if dead_sensors:
        plt.scatter([s.x for s in dead_sensors], [s.y for s in dead_sensors], 
                   c='red', marker='x', s=100, label='Dead Sensors', zorder=1)
    
    # Plot base station
    plt.scatter(300, 300, c='black', marker='s', s=100, label='Base Station')
    
    # Draw the current charging radius
    circle = plt.Circle((mc.x, mc.y), CHARGING_RADIUS, color='blue', 
                        fill=False, linestyle='--', alpha=0.5)
    plt.gcf().gca().add_artist(circle)
    
    # Plot all previous paths with fading effect
    for i in range(step-1):
        path_x = all_paths_x[i]
        path_y = all_paths_y[i]
        # Older paths are more transparent
        alpha = 0.3 * (i / (step-1)) + 0.2 if step > 1 else 0.5
        plt.plot(path_x, path_y, 'b-', linewidth=1.5, alpha=alpha, zorder=1)
    
    # Plot current path with more emphasis
    if step-1 < len(all_paths_x):
        path_x = all_paths_x[step-1]
        path_y = all_paths_y[step-1]
        plt.plot(path_x, path_y, 'b-', linewidth=2.5, zorder=3)
    
    # Highlight charging events
    for cx, cy, event_step in charging_events:
        if event_step <= step:  # Only show events that have happened
            plt.scatter(cx, cy, c='yellow', edgecolors='orange', s=200, 
                       marker='*', zorder=4, alpha=0.8)
    
    # Plot current position of mobile charger
    plt.scatter(mc.x, mc.y, c='blue', marker='X', s=150, label='Mobile Charger')
    
    # Add charging beams for recently charged sensors with enhanced visualization
    if step-1 < len(all_charged_nodes_history) and all_charged_nodes_history[step-1]:
        for sensor_id in all_charged_nodes_history[step-1]:
            sensor = next((s for s in sensors if s.id == sensor_id), None)
            if sensor:
                distance = np.linalg.norm([mc.x - sensor.x, mc.y - sensor.y])
                efficiency = max(0.3, 1 - (distance / CHARGING_RADIUS) * 0.7)
                
                # Enhanced visualization: gradient beam color based on efficiency
                # Higher efficiency = more yellow, lower = more orange
                beam_color = (1, efficiency, 0)  # RGB: Yellow to orange
                
                plt.plot([mc.x, sensor.x], [mc.y, sensor.y], color=beam_color, 
                         alpha=0.8, linewidth=3*efficiency, zorder=1)
                
                # Add small charging symbol at sensor position
                plt.scatter(sensor.x, sensor.y, marker='+', c='yellow', s=100, zorder=5, alpha=0.8)
    
    # Add a custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor=(0,1,0), edgecolor='black', label='Full Energy'),
        Patch(facecolor=(0.5,0.5,0), edgecolor='black', label='Half Energy'),
        Patch(facecolor=(1,0,0), edgecolor='black', label='Low Energy'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='blue', markersize=10, label='Mobile Charger'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow', markersize=10, label='Charging Event'),
        Line2D([0], [0], marker='x', color='red', lw=0, label='Dead Sensor'),
        Line2D([0], [0], color='blue', lw=2, label='Path'),
        Line2D([0], [0], color='yellow', lw=2, label='Charging Beam'),
        Line2D([0], [0], marker='+', color='yellow', markersize=10, label='Currently Charging')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Set title and grid
    plt.title(f"Enhanced Continuous Charging Visualization (Step {step})")
    
    # Add MC energy status and other statistics
    plt.annotate(f"MC Energy: {mc.energy:.0f}J", xy=(0.02, 0.98), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                va='top')
    
    # Add information about charged nodes
    if all_charged_nodes_history and all_charged_nodes_history[-1]:
        charged_text = f"Charging {len(all_charged_nodes_history[-1])} sensors"
        plt.annotate(charged_text, xy=(0.02, 0.93), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                    va='top')
    
    plt.grid(True)
    plt.xlim(0, AREA_WIDTH)
    plt.ylim(0, AREA_HEIGHT)
    plt.tight_layout()
    plt.show()
    
    
def visualize_enhanced_continuous_charging(num_steps=20, time_step=1, num_sensors=70, path_segments=12):
    """
    Enhanced visualization showing all sensors being charged as the mobile charger moves.
    Uses more path segments for smoother movement and charging.
    """
    # Set a fixed seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize environment with even more sensors for better visualization
    sensors, mc = initialize_environment(num_sensors=num_sensors)
    
    # Set varied energy levels - more sensors at low energy to show charging effect
    for i, s in enumerate(sensors):
        # Create a mix of energy levels with more low-energy sensors
        if i % 4 == 0:  # 25% very low
            s.energy = 0.05 * s.capacity
        elif i % 4 == 1:  # 25% low
            s.energy = 0.15 * s.capacity 
        elif i % 4 == 2:  # 25% medium
            s.energy = 0.4 * s.capacity
        else:
            s.energy = 0.7 * s.capacity
        
        # Use very low consumption rates to keep simulation running longer
        s.consumption_rate = random.uniform(0.02, 0.2)
    
    # Cluster some sensors to make charging while moving more visible
    # Place a group of 5 sensors in a line that the charger will pass through
    line_start_x, line_start_y = random.uniform(100, 200), random.uniform(100, 200)
    line_end_x, line_end_y = random.uniform(400, 500), random.uniform(400, 500)
    for i in range(5):
        progress = i / 4.0  # 0 to 1
        x = line_start_x + (line_end_x - line_start_x) * progress
        y = line_start_y + (line_end_y - line_start_y) * progress
        # Override 5 existing sensors with these positioned sensors
        sensors[i*3].x = x
        sensors[i*3].y = y
        sensors[i*3].energy = 0.1 * sensors[i*3].capacity  # Low energy to need charging
    
    # Path tracking and statistics
    path_x = [mc.x]
    path_y = [mc.y]
    charging_events = []
    total_energy_transferred = 0
    initial_energy = {s.id: s.energy for s in sensors}
    
    # Create animation data storage
    all_paths_x = []
    all_paths_y = []
    all_mc_positions = []
    all_charged_nodes_history = []
    
    # Main simulation loop
    for step in range(num_steps):
        print(f"--- Step {step+1}/{num_steps} --- MC Energy: {mc.energy:.1f}J")
        
        # Find sensors needing charging
        to_charge = [s for s in sensors if s.needs_charging()]
        if not to_charge:
            # Let energy deplete if no sensors need charging
            for s in sensors:
                s.update_energy(time_step * 2)
            print("No sensors need charging. Letting energy deplete...")
            continue
        
        # Prioritize based on energy level and distance
        # Modified to highly favor sensors in a line (to show charging while moving)
        def priority_score(sensor):
            distance = np.linalg.norm([mc.x - sensor.x, mc.y - sensor.y])
            energy_ratio = sensor.energy / sensor.capacity
            
            # Check if this is one of our lined-up sensors
            is_lined_up = any(np.linalg.norm([sensor.x - sensors[i*3].x, sensor.y - sensors[i*3].y]) < 0.1 
                             for i in range(5))
            
            # Heavily prioritize sensors in our line if they need charging
            if is_lined_up and energy_ratio < 0.3:
                return -1000  # Super high priority
            
            # Otherwise use normal priority formula
            return distance * 0.5 + energy_ratio * 100
        
        # Sort by priority
        to_charge.sort(key=priority_score)
        target = to_charge[0]
        print(f"Moving to sensor {target.id} ({target.energy/target.capacity:.1%} charge)")
        
        # Save current position
        start_point = (mc.x, mc.y)
        end_point = (target.x, target.y)
        
        # Move and charge continuously along path with more segments for smoother movement
        energy_transferred, charged_node_ids, path_segment_x, path_segment_y = move_and_charge_along_path(
            mc, start_point, end_point, sensors, path_segments=path_segments
        )
        
        # Update tracking variables
        path_x.extend(path_segment_x[1:])  # Skip first point (already in path_x)
        path_y.extend(path_segment_y[1:])
        total_energy_transferred += energy_transferred
        
        # Store animation data for this step
        all_paths_x.append(path_segment_x)
        all_paths_y.append(path_segment_y)
        all_mc_positions.append((mc.x, mc.y))
        all_charged_nodes_history.append(charged_node_ids)
        
        if charged_node_ids:
            print(f"Charged {len(charged_node_ids)} sensors with {energy_transferred:.1f}J while moving")
            charging_events.append((mc.x, mc.y, step+1))
        
        # Regular energy consumption for all sensors
        for s in sensors:
            s.update_energy(time_step)
            
        # Visualize current state with enhanced visualization
        visualize_enhanced_continuous_path_step(
            sensors, mc, all_paths_x, all_paths_y, 
            charging_events, step+1, all_charged_nodes_history
        )
    
    # Show final energy summary
    final_energy = {s.id: s.energy for s in sensors}
    charged_count = sum(1 for s_id in initial_energy if final_energy[s_id] > initial_energy[s_id])
    depleted_count = sum(1 for s_id in initial_energy if final_energy[s_id] < initial_energy[s_id])
    
    print("\nEnergy Change Summary:")
    for s_id in sorted(initial_energy.keys()):
        initial = initial_energy[s_id]
        final = final_energy[s_id]
        change = final - initial
        status = "CHARGED" if change > 0 else "DEPLETED" if change < 0 else "UNCHANGED"
        print(f"Sensor {s_id}: {initial:.1f}J → {final:.1f}J ({change:+.1f}J) - {status}")
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"Total energy transferred: {total_energy_transferred:.1f}J")
    print(f"Sensors charged: {charged_count}, Sensors depleted: {depleted_count}")
    print(f"MC remaining energy: {mc.energy:.1f}J")
    
    # Show heatmap of charging activity
    visualize_charging_heatmap(path_x, path_y, charging_events, sensors)

# Add this function to the existing file (at the end)

def visualize_sensor_priorities(sensors, mc, path_history=None):
    """Visualize sensors with their adaptive priority scores and path history"""
    plt.figure(figsize=(12, 10))
    
    # Import here to avoid circular imports
    from adaptive_charging import calculate_adaptive_priority
    
    # Calculate priorities for all alive sensors
    sensor_priorities = []
    for s in sensors:
        if not s.dead:
            priority = calculate_adaptive_priority(s, mc)
            sensor_priorities.append((s, priority))
    
    # Normalize priorities for visualization
    max_priority = max(p for _, p in sensor_priorities) if sensor_priorities else 1.0
    
    # Plot sensors with color based on priority
    for s, priority in sensor_priorities:
        # Normalize to 0-1 range
        norm_priority = priority / max_priority
        
        # Use color scale: red (high priority) to blue (low priority)
        color = [norm_priority, 0, 1-norm_priority]
        size = 50 + norm_priority * 100  # Size also reflects priority
        
        plt.scatter(s.x, s.y, c=[color], s=size, alpha=0.7, edgecolors='black')
        
        # Show priority score and sensor ID
        plt.annotate(f"ID:{s.id}\nP:{priority:.2f}", (s.x, s.y), 
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
        
        # Show energy level
        energy_pct = s.energy / s.capacity * 100
        plt.annotate(f"{energy_pct:.0f}%", (s.x, s.y), 
                    xytext=(5, -10), textcoords="offset points", fontsize=7,
                    color='green' if energy_pct > 50 else 'red')
    
    # Plot dead sensors
    dead_x = [s.x for s in sensors if s.dead]
    dead_y = [s.y for s in sensors if s.dead]
    if dead_x:
        plt.scatter(dead_x, dead_y, c='black', marker='x', s=100, label='Dead Sensors')
    
    # Plot MC
    plt.scatter(mc.x, mc.y, c='blue', marker='X', s=150, label='Mobile Charger')
    plt.scatter(300, 300, c='black', marker='s', label='Base Station')
    
    # Draw charging radius
    circle = plt.Circle((mc.x, mc.y), CHARGING_RADIUS, color='blue', 
                       fill=False, linestyle='--', alpha=0.5)
    plt.gcf().gca().add_artist(circle)
    
    # Plot path history if available
    if path_history and len(path_history) > 1:
        path_x, path_y = zip(*path_history)
        plt.plot(path_x, path_y, 'b-', alpha=0.4, linewidth=2, label='MC Path')
    
    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='High Priority'),
        Patch(facecolor='purple', edgecolor='black', label='Medium Priority'),
        Patch(facecolor='blue', edgecolor='black', label='Low Priority'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='blue', markersize=10, label='Mobile Charger')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Sensor Nodes with Adaptive Priority Visualization")
    plt.grid(True)
    plt.xlim(0, AREA_WIDTH)
    plt.ylim(0, AREA_HEIGHT)
    plt.show()

def visualize_sensor_priorities_with_zones(sensors, mc, path_history=None):
    """
    Visualize sensors with their adaptive priority scores, mobile charger path, and charging zones
    """
    plt.figure(figsize=(12, 10))
    
    from adaptive_charging import calculate_adaptive_priority
    
    # Draw charging zones
    outer_circle = plt.Circle((mc.x, mc.y), CHARGING_RADIUS, 
                             color='red', fill=True, alpha=0.1, label="Outer Zone (30%)")
    middle_circle = plt.Circle((mc.x, mc.y), CHARGING_RADIUS * 0.7, 
                              color='yellow', fill=True, alpha=0.15, label="Middle Zone (50%)")
    inner_circle = plt.Circle((mc.x, mc.y), CHARGING_RADIUS * 0.4, 
                             color='green', fill=True, alpha=0.2, label="Inner Zone (70%)")
    
    plt.gcf().gca().add_artist(outer_circle)
    plt.gcf().gca().add_artist(middle_circle)
    plt.gcf().gca().add_artist(inner_circle)
    
    # Calculate priorities for all alive sensors
    sensor_priorities = []
    for s in sensors:
        if not s.dead:
            priority = calculate_adaptive_priority(s, mc)
            sensor_priorities.append((s, priority))
    
    # Normalize priorities for visualization
    max_priority = max(p for _, p in sensor_priorities) if sensor_priorities else 1.0
    
    # Plot path history if provided
    if path_history and len(path_history) > 1:
        path_x = [p[0] for p in path_history]
        path_y = [p[1] for p in path_history]
        
        # Plot path with gradient to show direction/time
        for i in range(len(path_history) - 1):
            # Calculate gradient color (from light to darker blue)
            alpha = 0.3 + 0.7 * (i / max(1, len(path_history) - 2))
            color = (0, 0, 0.8, alpha)
            
            # Draw line segment with arrow to show direction
            plt.arrow(path_x[i], path_y[i], 
                     path_x[i+1] - path_x[i], 
                     path_y[i+1] - path_y[i],
                     head_width=10, head_length=10, 
                     fc=color, ec=color, 
                     length_includes_head=True,
                     zorder=2)
        
        # Mark visited points along the path
        plt.scatter(path_x[:-1], path_y[:-1], c='blue', alpha=0.5, 
                   s=30, marker='o', label='Previous Positions')
    
    # Plot sensors with color based on priority
    for s, priority in sensor_priorities:
        # Normalize to 0-1 range
        norm_priority = priority / max_priority
        
        # Use color scale: red (high priority) to blue (low priority)
        color = [norm_priority, 0, 1-norm_priority]
        size = 50 + norm_priority * 100  # Size also reflects priority
        
        plt.scatter(s.x, s.y, c=[color], s=size, alpha=0.7, edgecolors='black', zorder=3)
        
        # Show priority score and sensor ID
        plt.annotate(f"ID:{s.id}\nP:{priority:.2f}", (s.x, s.y), 
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
        
        # Show energy level and consumption rate
        energy_pct = s.energy / s.capacity * 100
        plt.annotate(f"{energy_pct:.0f}% | {s.consumption_rate:.1f}J/s", (s.x, s.y), 
                    xytext=(5, -10), textcoords="offset points", fontsize=7,
                    color='green' if energy_pct > 50 else 'red')
    
    # Determine charging zone for each sensor and annotate
    for s in sensors:
        if not s.dead:
            distance = np.linalg.norm([mc.x - s.x, mc.y - s.y])
            if distance <= CHARGING_RADIUS:
                distance_ratio = distance / CHARGING_RADIUS
                
                # Determine zone and efficiency
                if distance_ratio <= 0.4:
                    eff = "70%"
                    color = 'green'
                elif distance_ratio <= 0.7:
                    eff = "50%"
                    color = 'gold'
                else:
                    eff = "30%"
                    color = 'red'
                    
                # Add charging efficiency label
                plt.annotate(eff, (s.x, s.y), 
                            xytext=(0, 15), textcoords="offset points", 
                            ha='center', color=color, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.1", fc='white', alpha=0.7))
    
    # Plot dead sensors
    dead_x = [s.x for s in sensors if s.dead]
    dead_y = [s.y for s in sensors if s.dead]
    if dead_x:
        plt.scatter(dead_x, dead_y, c='black', marker='x', s=100, label='Dead Sensors')
    
    # Plot Mobile Charger and Base Station
    plt.scatter(mc.x, mc.y, c='blue', marker='X', s=150, label='Mobile Charger', zorder=5)
    plt.scatter(300, 300, c='black', marker='s', s=100, label='Base Station', zorder=5)
    
    # Add information box with MC energy
    info_text = f"MC Energy: {mc.energy:.0f}J ({mc.energy/MC_CAPACITY:.1%})\n"
    info_text += f"Alive Sensors: {sum(1 for s in sensors if not s.dead)}/{len(sensors)}\n"
    info_text += f"Charging Range: {CHARGING_RADIUS}m"
    
    plt.annotate(info_text, xy=(0.02, 0.98), xycoords='axes fraction',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
               va='top', fontsize=10)
    
    # Add charging zone legend
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='High Priority'),
        Patch(facecolor='purple', edgecolor='black', label='Medium Priority'),
        Patch(facecolor='blue', edgecolor='black', label='Low Priority'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='blue', markersize=10, label='Mobile Charger'),
        Line2D([0], [0], color='blue', alpha=0.7, lw=2, label='Charger Path'),
        Patch(facecolor='green', alpha=0.2, label='Inner Zone (70%)'),
        Patch(facecolor='yellow', alpha=0.15, label='Middle Zone (50%)'),
        Patch(facecolor='red', alpha=0.1, label='Outer Zone (30%)')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Sensor Nodes with Adaptive Priority and Charging Zones")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, AREA_WIDTH)
    plt.ylim(0, AREA_HEIGHT)
    plt.tight_layout()
    plt.show()


def visualize_optimal_position(sensors, mc, optimal_pos, covered_sensors, path_history=None):
    """
    Visualize the optimal charging position and the sensors that would be covered
    """
    # Use the existing visualization as a base
    visualize_sensor_priorities(sensors, mc, path_history)
    
    # Highlight the optimal position
    opt_x, opt_y = optimal_pos
    plt.scatter([opt_x], [opt_y], c='gold', marker='*', s=300, 
               label='Optimal Position', zorder=10, edgecolors='black')
    
    # Draw a line from current position to optimal position
    plt.plot([mc.x, opt_x], [mc.y, opt_y], 'g--', linewidth=2, 
            label='Planned Movement')
    
    # Draw the charging zones from the optimal position with colored fill
    outer_circle = plt.Circle(optimal_pos, CHARGING_RADIUS, 
                             color='red', fill=True, alpha=0.1, label="Outer Zone (30%)")
    middle_circle = plt.Circle(optimal_pos, CHARGING_RADIUS * 0.7, 
                              color='yellow', fill=True, alpha=0.15, label="Middle Zone (50%)")
    inner_circle = plt.Circle(optimal_pos, CHARGING_RADIUS * 0.4, 
                             color='green', fill=True, alpha=0.2, label="Inner Zone (70%)")
    
    plt.gcf().gca().add_artist(outer_circle)
    plt.gcf().gca().add_artist(middle_circle)
    plt.gcf().gca().add_artist(inner_circle)
    
    # Highlight covered sensors
    covered_x = []
    covered_y = []
    for s in sensors:
        if s.id in covered_sensors:
            covered_x.append(s.x)
            covered_y.append(s.y)
    
    if covered_x:
        plt.scatter(covered_x, covered_y, facecolors='none', edgecolors='gold', 
                   s=180, linewidth=2, label='Covered Sensors', zorder=6)
    
    # Add efficiency predictions for each covered sensor
    for s in sensors:
        if s.id in covered_sensors:
            distance = np.linalg.norm([opt_x - s.x, opt_y - s.y])
            distance_ratio = distance / CHARGING_RADIUS
            
            if distance_ratio <= 0.4:
                eff = "70%"
                color = 'green'
            elif distance_ratio <= 0.7:
                eff = "50%"
                color = 'gold'
            else:
                eff = "30%"
                color = 'red'
                
            # Add predicted efficiency label
            plt.annotate(f"Pred: {eff}", (s.x, s.y), 
                        xytext=(0, 25), textcoords="offset points", 
                        ha='center', color=color, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.1", fc='white', alpha=0.7))
    
    # Update legend with zone information
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Add zone patches to legend
    from matplotlib.patches import Patch
    zone_patches = [
        Patch(facecolor='green', alpha=0.2, label='Inner Zone (70%)'),
        Patch(facecolor='yellow', alpha=0.15, label='Middle Zone (50%)'),
        Patch(facecolor='red', alpha=0.1, label='Outer Zone (30%)')
    ]
    
    handles.extend(zone_patches)
    labels.extend(['Inner Zone (70%)', 'Middle Zone (50%)', 'Outer Zone (30%)'])
    
    plt.legend(handles=handles, labels=labels, loc='upper right')
    
    # Add information about the optimal position
    info_text = f"Optimal Position: ({opt_x:.1f}, {opt_y:.1f})\n"
    info_text += f"Covers {len(covered_sensors)} sensors\n"
    info_text += f"Movement Distance: {np.linalg.norm([mc.x - opt_x, mc.y - opt_y]):.1f}m"
    
    plt.annotate(info_text, xy=(0.02, 0.88), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                va='top', fontsize=10)
    
    plt.title("Optimal Charging Position Strategy")
    plt.tight_layout()
    plt.show()