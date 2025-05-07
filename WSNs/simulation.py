import numpy as np
from environment import count_dead_sensors
from config import NUM_SENSORS
from visualization import plot_environment

def simulate_step(sensors, mc, time_step=10):
    # 1. Update energy of all sensors
    for s in sensors:
        s.update_energy(time_step)

    # 2. Filter sensors needing charging and still alive
    to_charge = [s for s in sensors if s.needs_charging()]
    if not to_charge:
        print("No sensor needs charging right now.")
        return False

    # 3. Select next sensor using greedy strategy (closest one)
    to_charge.sort(key=lambda s: np.linalg.norm([mc.x - s.x, mc.y - s.y]))
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

def run_simulation(sensors, mc, max_steps=1000, time_step=10, visualize_every=50):
    for step in range(max_steps):
        print(f"--- Step {step} ---")
        result = simulate_step(sensors, mc, time_step)
        if not result:
            print("Simulation ended (MC depleted or no tasks).")
            break

        if step % visualize_every == 0:
            plot_environment(sensors, mc)

    # Final stats
    dead = sum(1 for s in sensors if s.dead)
    alive = NUM_SENSORS - dead
    print(f"Simulation Complete. Alive: {alive}, Dead: {dead}")
    print(f"MC remaining energy: {mc.energy:.2f} J")