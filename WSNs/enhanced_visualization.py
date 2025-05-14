import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
from config import AREA_WIDTH, AREA_HEIGHT, CHARGING_RADIUS, MC_CAPACITY

def visualize_optimal_position(sensors, mc, optimal_pos, covered_sensors, path_history=None):
    """
    Visualize the optimal charging position with charging zones (30%, 50%, 70% efficiency)
    
    Args:
        sensors: List of sensor nodes
        mc: Mobile charger object
        optimal_pos: Tuple (x,y) of optimal position
        covered_sensors: List of sensor IDs covered by the optimal position
        path_history: List of (x,y) positions representing MC path
    """
    plt.figure(figsize=(12, 10))
    
    # Import here to avoid circular imports
    from adaptive_charging import calculate_adaptive_priority
    
    # Draw charging zones for current position
    outer_circle = plt.Circle((mc.x, mc.y), CHARGING_RADIUS, 
                              color='red', fill=True, alpha=0.1)
    middle_circle = plt.Circle((mc.x, mc.y), CHARGING_RADIUS * 0.7, 
                               color='yellow', fill=True, alpha=0.15)
    inner_circle = plt.Circle((mc.x, mc.y), CHARGING_RADIUS * 0.4, 
                              color='green', fill=True, alpha=0.2)
    
    plt.gcf().gca().add_artist(outer_circle)
    plt.gcf().gca().add_artist(middle_circle)
    plt.gcf().gca().add_artist(inner_circle)
    
    # Draw charging zones for optimal position
    opt_x, opt_y = optimal_pos
    opt_outer_circle = plt.Circle((opt_x, opt_y), CHARGING_RADIUS, 
                                 color='gold', fill=False, linestyle='--', alpha=0.7)
    opt_middle_circle = plt.Circle((opt_x, opt_y), CHARGING_RADIUS * 0.7, 
                                  color='gold', fill=False, linestyle='--', alpha=0.7)
    opt_inner_circle = plt.Circle((opt_x, opt_y), CHARGING_RADIUS * 0.4, 
                                 color='gold', fill=False, linestyle='--', alpha=0.7)
    
    plt.gcf().gca().add_artist(opt_outer_circle)
    plt.gcf().gca().add_artist(opt_middle_circle)
    plt.gcf().gca().add_artist(opt_inner_circle)
    
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
    
    # Draw a line from current position to optimal position
    plt.plot([mc.x, opt_x], [mc.y, opt_y], 'g--', linewidth=2, 
             label='Planned Movement')
    
    # Plot sensors with color based on priority
    for s, priority in sensor_priorities:
        # Normalize to 0-1 range with clamping to prevent out-of-range values
        norm_priority = min(1.0, max(0.0, priority / max_priority))
        
        # Color scale: red (high priority) to blue (low priority)
        color = [norm_priority, 0, 1-norm_priority]
        size = 50 + norm_priority * 100  # Size also reflects priority
        
        plt.scatter(s.x, s.y, c=[color], s=size, alpha=0.7, edgecolors='black', zorder=3)
        
        # Show priority and sensor ID
        plt.annotate(f"ID:{s.id}\nP:{priority:.2f}", (s.x, s.y), 
                     xytext=(5, 5), textcoords="offset points", fontsize=8)
        
        # Show energy level and consumption rate
        energy_pct = s.energy / s.capacity * 100
        plt.annotate(f"{energy_pct:.0f}% | {s.consumption_rate:.4f}J/s", (s.x, s.y), 
                     xytext=(5, -10), textcoords="offset points", fontsize=7,
                     color='green' if energy_pct > 50 else 'red')
    
    # Highlight covered sensors from optimal position
    for s in sensors:
        if not s.dead and s.id in covered_sensors:
            # Calculate efficiency at optimal position
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
                
            # Mark as covered by optimal position
            plt.scatter(s.x, s.y, facecolors='none', edgecolors='gold', 
                       s=180, linewidth=2, zorder=6)
            
            # Add predicted efficiency
            plt.annotate(f"Opt: {eff}", (s.x, s.y), 
                        xytext=(0, 30), textcoords="offset points", 
                        ha='center', color=color, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.1", fc='lightyellow', alpha=0.7))
    
    # Highlight the optimal position
    plt.scatter([opt_x], [opt_y], c='gold', marker='*', s=300, 
               label='Optimal Position', zorder=10, edgecolors='black')
    
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
    info_text += f"Sensors in optimal range: {len(covered_sensors)}"
    
    plt.annotate(info_text, xy=(0.02, 0.98), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                va='top', fontsize=10)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='High Priority'),
        Patch(facecolor='purple', edgecolor='black', label='Medium Priority'),
        Patch(facecolor='blue', edgecolor='black', label='Low Priority'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='blue', markersize=10, label='Mobile Charger'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=15, label='Optimal Position'),
        Patch(facecolor='green', alpha=0.2, label='Inner Zone (70%)'),
        Patch(facecolor='yellow', alpha=0.15, label='Middle Zone (50%)'),
        Patch(facecolor='red', alpha=0.1, label='Outer Zone (30%)'),
        Line2D([0], [0], color='gold', linestyle='--', lw=2, label='Optimal Zones'),
        Line2D([0], [0], color='green', linestyle='--', lw=2, label='Planned Movement')
    ]
    
    if path_history and len(path_history) > 1:
        legend_elements.append(Line2D([0], [0], color='blue', alpha=0.7, lw=2, label='Charger Path'))
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Optimal Charging Position with Zone-Based Efficiency")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, AREA_WIDTH)
    plt.ylim(0, AREA_HEIGHT)
    plt.tight_layout()
    plt.show()