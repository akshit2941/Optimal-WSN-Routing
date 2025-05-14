import matplotlib.pyplot as plt
import numpy as np
from config import AREA_WIDTH, AREA_HEIGHT, CHARGING_RADIUS

def plot_environment(sensors, mc):
    alive_x = [s.x for s in sensors if not s.dead]
    alive_y = [s.y for s in sensors if not s.dead]
    dead_x = [s.x for s in sensors if s.dead]
    dead_y = [s.y for s in sensors if s.dead]

    plt.figure(figsize=(8, 8))
    plt.scatter(alive_x, alive_y, c='green', label='Alive Sensors')
    plt.scatter(dead_x, dead_y, c='red', label='Dead Sensors')
    plt.scatter(mc.x, mc.y, c='blue', label='Mobile Charger (MC)', marker='X', s=100)
    plt.scatter(300, 300, c='black', label='Base Station', marker='s')
    
    # Draw charging radius
    circle = plt.Circle((mc.x, mc.y), CHARGING_RADIUS, color='blue', fill=False, linestyle='--', alpha=0.5)
    plt.gcf().gca().add_artist(circle)
    
    # Highlight nodes in charging range
    in_range_x = []
    in_range_y = []
    for s in sensors:
        if not s.dead:
            distance = np.linalg.norm([mc.x - s.x, mc.y - s.y])
            if distance <= CHARGING_RADIUS:
                in_range_x.append(s.x)
                in_range_y.append(s.y)
    
    if in_range_x:
        plt.scatter(in_range_x, in_range_y, c='yellow', edgecolors='orange', 
                   label='In Charging Range', s=80, zorder=3)
    
    plt.legend()
    plt.title("WRSN Environment")
    plt.grid(True)
    plt.xlim(0, AREA_WIDTH)
    plt.ylim(0, AREA_HEIGHT)
    plt.show()

def visualize_charging_heatmap(path_x, path_y, charging_events, sensors):
    """Creates a heatmap showing where most charging activity occurred"""
    plt.figure(figsize=(10, 8))
    
    # Plot base environment
    plt.scatter([s.x for s in sensors if not s.dead], 
                [s.y for s in sensors if not s.dead], 
                c='green', alpha=0.3, s=30)
    plt.scatter([s.x for s in sensors if s.dead], 
                [s.y for s in sensors if s.dead], 
                c='red', marker='x', s=30)
    
    # Plot path with low opacity
    plt.plot(path_x, path_y, 'b-', alpha=0.3, linewidth=1)
    
    # Create charging event heatmap
    if charging_events:
        charge_x = [x for x, y, _ in charging_events]
        charge_y = [y for x, y, _ in charging_events]
        
        # Use histogram2d to create a heatmap
        heatmap, xedges, yedges = np.histogram2d(charge_x, charge_y, 
                                                bins=[20, 20], 
                                                range=[[0, AREA_WIDTH], [0, AREA_HEIGHT]])
        extent = [0, AREA_WIDTH, 0, AREA_HEIGHT]
        
        plt.imshow(heatmap.T, extent=extent, origin='lower', 
                  cmap='hot', alpha=0.5, interpolation='gaussian')
        plt.colorbar(label='Charging Frequency')
        
        # Plot the charging locations as stars
        plt.scatter(charge_x, charge_y, c='yellow', marker='*', 
                   s=100, edgecolors='orange', alpha=0.7)
    
    plt.title("WRSN Charging Activity Heatmap")
    plt.xlim(0, AREA_WIDTH)
    plt.ylim(0, AREA_HEIGHT)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()