# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import os
# import random
# from config import *
# from environment import initialize_environment, get_state_vector, get_valid_actions
# from sensor_node import SensorNode
# from mobile_charger import MobileCharger
# from visualization import plot_environment
# from simulation import run_simulation, simulate_step
# from reinforcement_learning import DQNAgent, train_dqn, evaluate_agent
# from enhanced_visualization import (
#     visualize_charger_path,
#     visualize_continuous_charging_path,
#     visualize_enhanced_continuous_charging,
#     visualize_optimal_position
# )
# from adaptive_charging import run_adaptive_continuous_charging, run_optimal_position_simulation, simulate_step_adaptive, run_adaptive_continuous_charging_with_zones
# from enhanced_visualization import visualize_sensor_priorities, visualize_sensor_priorities_with_zones

# def main():
#     # Initial setup and demonstration
#     print("Initializing WRSN environment...")
#     sensors, mc = initialize_environment()
    
#     # Basic visualization
#     print("\nVisualizing initial environment...")
#     plot_environment(sensors, mc)
    
#     # Run simple simulation
#     print("\nRunning basic simulation...")
#     # sensors, mc = initialize_environment()
#     # run_simulation(sensors, mc, max_steps=100, time_step=15, visualize_every=20)
    
#     # Set up DQN agent
#     print("\nSetting up DQN agent...")
#     state_size = 1 + NUM_SENSORS * 4
#     action_size = NUM_SENSORS + 1
#     agent = DQNAgent(state_size, action_size)
    
#     # Path for saved model
#     model_path = "trained_wrsn_agent.pth"
    
#     # Check if we want to train or load a pretrained model
#     train_new_model = False  # Set to False to load pretrained model
    
#     if train_new_model or not os.path.exists(model_path):
#         # Train agent
#         print("\nTraining DQN agent...")
#         rewards = train_dqn(agent, episodes=50)
        
#         # Save the trained model
#         agent.save(model_path)
        
#         # Plot training progress
#         plt.figure(figsize=(10, 6))
#         plt.plot(rewards)
#         plt.xlabel("Episode")
#         plt.ylabel("Total Reward")
#         plt.title("Reward Trend During DQN Training")
#         plt.grid(True)
#         plt.show()
#     else:
#         # Load the pretrained model
#         print("\nLoading pretrained DQN agent...")
#         agent.load(model_path)
    
#     # Evaluate agent
#     print("\nEvaluating trained agent...")
#     results = evaluate_agent(agent, episodes=3)
    
#     # Display evaluation summary
#     rewards_eval, deaths_eval = zip(*results)
#     print("\n--- Evaluation Summary ---")
#     print(f"Average Evaluation Reward: {np.mean(rewards_eval):.2f}")
#     print(f"Average Dead Sensors: {np.mean(deaths_eval):.2f}")
    
#     # Show advanced visualization
#     print("\nRunning charger path visualization...")
#     # visualize_charger_path(num_steps=10, time_step=15)
    
#     # Show continuous charging demonstration
#     print("\nRunning continuous charging visualization...")
#     # visualize_continuous_charging_path(num_steps=10, time_step=2, num_sensors=30, path_segments=5)
    
#     # Add the enhanced continuous charging visualization
#     print("\nRunning enhanced continuous charging visualization...")
#     # visualize_enhanced_continuous_charging(
#     #     num_steps=10,
#     #     time_step=1,
#     #     num_sensors=50,
#     #     path_segments=5
#     # )
    
#     # Create an empty path history for the initial visualization
#     # path_history = [(mc.x, mc.y)]  # Start with current position of MC
    
#     # # Set varied energy levels and consumption rates for visualization
#     # for i, s in enumerate(sensors):
#     #     if i % 5 == 0:  # 20% very low
#     #         s.energy = 0.05 * s.capacity
#     #     elif i % 5 == 1:  # 20% low
#     #         s.energy = 0.2 * s.capacity
#     #     elif i % 5 == 2:  # 20% medium
#     #         s.energy = 0.5 * s.capacity
            
#     #     # Vary consumption rates to create different priorities
#     #     if i % 3 == 0:  # High consumption
#     #         s.consumption_rate = random.uniform(0.8, 1.5)
#     #     elif i % 3 == 1:  # Medium consumption
#     #         s.consumption_rate = random.uniform(0.4, 0.7)
#     #     else:  # Low consumption
#     #         s.consumption_rate = random.uniform(0.1, 0.3)
    
#     # # Visualize initial priorities
#     # print("Initial sensor priorities:")
#     # visualize_sensor_priorities(sensors, mc, path_history)
    
    
#     # print("\nRunning adaptive continuous charging simulation...")
#     # run_adaptive_continuous_charging(
#     #     num_steps=10,
#     #     time_step=3,
#     #     num_sensors=40,
#     #     path_segments=8
#     # )
    
#     # print("\nRunning adaptive continuous charging with zone-based efficiency...")
#     # sensors, mc, path_history = run_adaptive_continuous_charging_with_zones(
#     #     num_steps=8,
#     #     time_step=3,
#     #     num_sensors=40,
#     #     path_segments=8
#     # )

#     # # Visualize final state with complete path and charging zones
#     # print("\nFinal network state with complete mobile charger path and charging zones:")
#     # visualize_sensor_priorities_with_zones(sensors, mc, path_history)
    
#     # Run simulation using the optimal position strategy
#     # print("\nRunning simulation with optimal position strategy...")
#     # sensors, mc, path_history = run_optimal_position_simulation(
#     #     num_steps=6,      # Fewer steps as this is more computationally intensive
#     #     time_step=3,
#     #     num_sensors=40
#     # )

#     # # Show final state
#     # print("\nFinal network state with complete mobile charger path and charging zones:")
#     # visualize_sensor_priorities_with_zones(sensors, mc, path_history)
    
    
#     # In main.py, replace the commented-out code with:

#     print("\nRunning simulation with optimal position strategy and zone visualization...")
#     sensors, mc, path_history = run_optimal_position_simulation(
#         num_steps=6,      # Fewer steps as this is more computationally intensive
#         time_step=3,
#         num_sensors=50
#     )

#     # Show final state with zone visualization
#     print("\nFinal network state with complete mobile charger path and charging zones:")
#     visualize_optimal_position(sensors, mc, (mc.x, mc.y), [], path_history)
    
    
#     print("\nSimulation complete!")

# if __name__ == "__main__":
#     main()



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
    sensors, mc, path_history = run_optimal_position_simulation(
        num_steps=50,       # Increased from 6 to 20 for longer simulation
        time_step=10,        # Increased from 3 to 5 for more time per step
        num_sensors=80      # Increased from 50 to 70 for more complex environment
    )
    
    # Restore the original function
    enhanced_visualization.visualize_optimal_position = original_visualize
    
    print("\nSimulation complete! Results displayed above.")
    
    # Display additional summary statistics
    print("\n--- Final Network Statistics ---")
    alive_sensors = sum(1 for s in sensors if not s.dead)
    dead_sensors = sum(1 for s in sensors if s.dead)
    average_energy = sum(s.energy for s in sensors if not s.dead) / max(1, alive_sensors)
    
    print(f"Sensors alive: {alive_sensors}/{len(sensors)} ({alive_sensors/len(sensors)*100:.1f}%)")
    print(f"Sensors dead: {dead_sensors}/{len(sensors)} ({dead_sensors/len(sensors)*100:.1f}%)")
    print(f"Average energy of alive sensors: {average_energy:.1f}J ({average_energy/SENSOR_CAPACITY*100:.1f}%)")
    print(f"Mobile charger final energy: {mc.energy:.1f}J ({mc.energy/MC_CAPACITY*100:.1f}%)")
    print(f"Total positions visited: {len(path_history)}")

if __name__ == "__main__":
    main()