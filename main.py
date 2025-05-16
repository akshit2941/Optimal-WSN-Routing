import numpy as np
import os
import random
import torch
from config import *
from environment import initialize_environment, get_state_vector, get_valid_actions
from sensor_node import SensorNode
from mobile_charger import MobileCharger
from simulation import run_simulation, simulate_step
from reinforcement_learning import DQNAgent, train_dqn, evaluate_agent
from adaptive_charging import run_optimal_position_simulation

# Disable matplotlib visualization for the entire application
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

def main():
    # Initial setup and demonstration
    print("Initializing WRSN environment...")
    sensors, mc = initialize_environment()
    
    print("\nBasic environment initialization complete.")
    
    # Set up DQN agent as requested
    print("\nSetting up DQN agent...")
    state_size = 1 + NUM_SENSORS * 4
    action_size = NUM_SENSORS + 1
    agent = DQNAgent(state_size, action_size)
    
    # Path for saved model
    model_path = "trained_wrsn_agent.pth"
    
    # Check if we want to train or load a pretrained model
    train_new_model = False  # Set to False to load pretrained model
    
    if train_new_model or not os.path.exists(model_path):
        # Train agent
        print("\nTraining DQN agent...")
        rewards = train_dqn(agent, episodes=50)
        
        # Save the trained model
        agent.save(model_path)
        
        # Since visualization is disabled, just print training stats
        print(f"Training completed. Final reward: {rewards[-1]:.2f}")
        print(f"Average reward over last 10 episodes: {np.mean(rewards[-10:]):.2f}")
    else:
        # Load the pretrained model
        print("\nLoading pretrained DQN agent...")
        agent.load(model_path)
    
    # Evaluate agent
    print("\nEvaluating trained agent...")
    results = evaluate_agent(agent, episodes=3)
    
    # Display evaluation summary
    rewards_eval, deaths_eval = zip(*results)
    print("\n--- Evaluation Summary ---")
    print(f"Average Evaluation Reward: {np.mean(rewards_eval):.2f}")
    print(f"Average Dead Sensors: {np.mean(deaths_eval):.2f}")
    
    # Now run the optimal position simulation with visualization suppressed
    print("\nRunning simulation with optimal position strategy...")
    
    # Add a custom function to suppress visualization
    def dummy_visualize(*args, **kwargs):
        # This function does nothing, effectively suppressing visualization
        pass
    
    # Import and patch the module
    import adaptive_charging
    import enhanced_visualization
    
    # Save the original function
    original_visualize = enhanced_visualization.visualize_optimal_position
    
    # Replace with dummy function
    enhanced_visualization.visualize_optimal_position = dummy_visualize
    
    # Now run the simulation with longer parameters (visualization calls will do nothing)
    sensors, mc, path_history, metrics = run_optimal_position_simulation(
        num_steps=50,       # Increased from 6 to 20 for longer simulation
        time_step=10,        # Increased from 3 to 5 for more time per step
        num_sensors=NUM_SENSORS      # Used from config
    )
    
    # Restore the original function
    enhanced_visualization.visualize_optimal_position = original_visualize

    print("\nSimulation complete!")

    # Display comprehensive simulation results in one place
    print("\n=== SIMULATION RESULTS ===")

    print("\n--- Network Health Statistics ---")
    print(f"Sensors alive: {metrics['alive_ratio']*len(sensors):.0f}/{len(sensors)} ({metrics['alive_ratio']*100:.1f}%)")
    print(f"Sensors dead: {len(sensors) - metrics['alive_ratio']*len(sensors):.0f}/{len(sensors)} ({(1-metrics['alive_ratio'])*100:.1f}%)")
    print(f"Average energy of alive sensors: {metrics['average_energy']:.1f}J ({metrics['average_energy']/SENSOR_CAPACITY*100:.1f}%)")
    print(f"Mobile charger final energy: {mc.energy:.1f}J ({mc.energy/MC_CAPACITY*100:.1f}%)")
    print(f"Total positions visited: {len(path_history)}")

    print("\n--- Charging Performance Metrics ---")
    print(f"Survival Rate: {metrics['survival_rate']:.2f} ({len(metrics['sensors_received'])}/{len(metrics['sensors_requested'])})")
    if metrics['sensors_requested']:
        print(f"  - Sensors that requested charging: {metrics['sensors_requested']}")
        print(f"  - Sensors that received charging: {metrics['sensors_received']}")
        
        # Add this section to show sensors that were neglected
        neglected_sensors = set(metrics['sensors_requested']) - set(metrics['sensors_received'])
        if neglected_sensors:
            print(f"  - Sensors that requested but DIDN'T receive charging: {sorted(list(neglected_sensors))}")
            print(f"  - Neglected sensors count: {len(neglected_sensors)} ({len(neglected_sensors)/len(metrics['sensors_requested'])*100:.1f}% of requests)")
        else:
            print("  - All requested sensors were successfully charged!")
    else:
        print("  - No sensors requested charging")

    print(f"Energy Efficiency: {metrics['energy_efficiency']:.2f}")
    print(f"  - Energy received by sensors: {metrics['total_energy_transferred']:.1f}J")
    print(f"  - Movement energy cost: {metrics['total_movement_energy']:.1f}J")
    print(f"  - Total energy used: {metrics['total_energy_transferred'] + metrics['total_movement_energy']:.1f}J")

    # Add the new charging delay metric
    print(f"Average Charging Delay: {metrics['avg_charging_delay']:.1f} seconds")
    if metrics['charging_delays']:
        print(f"  - Min delay: {min(metrics['charging_delays']):.1f}s")
        print(f"  - Max delay: {max(metrics['charging_delays']):.1f}s")
        if len(metrics['charging_delays']) > 1:
            sorted_delays = sorted(metrics['charging_delays'])
            median_delay = sorted_delays[len(sorted_delays)//2]
            print(f"  - Median delay: {median_delay:.1f}s")

if __name__ == "__main__":
    main()