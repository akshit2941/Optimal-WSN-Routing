import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import random
from config import *
from environment import initialize_environment, get_state_vector, get_valid_actions
from sensor_node import SensorNode
from mobile_charger import MobileCharger
from visualization import plot_environment
from simulation import run_simulation, simulate_step
from reinforcement_learning import DQNAgent, train_dqn, evaluate_agent
from enhanced_visualization import (
    visualize_charger_path,
    visualize_continuous_charging_path,
    visualize_enhanced_continuous_charging
)
from adaptive_charging import run_adaptive_continuous_charging, run_optimal_position_simulation, simulate_step_adaptive, run_adaptive_continuous_charging_with_zones
from enhanced_visualization import visualize_sensor_priorities, visualize_sensor_priorities_with_zones

def main():
    # Initial setup and demonstration
    print("Initializing WRSN environment...")
    sensors, mc = initialize_environment()
    
    # Basic visualization
    print("\nVisualizing initial environment...")
    plot_environment(sensors, mc)
    
    # Run simple simulation
    print("\nRunning basic simulation...")
    # sensors, mc = initialize_environment()
    # run_simulation(sensors, mc, max_steps=100, time_step=15, visualize_every=20)
    
    # Set up DQN agent
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
        
        # Plot training progress
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward Trend During DQN Training")
        plt.grid(True)
        plt.show()
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
    
    # Show advanced visualization
    print("\nRunning charger path visualization...")
    # visualize_charger_path(num_steps=10, time_step=15)
    
    # Show continuous charging demonstration
    print("\nRunning continuous charging visualization...")
    # visualize_continuous_charging_path(num_steps=10, time_step=2, num_sensors=30, path_segments=5)
    
    # Add the enhanced continuous charging visualization
    print("\nRunning enhanced continuous charging visualization...")
    # visualize_enhanced_continuous_charging(
    #     num_steps=10,
    #     time_step=1,
    #     num_sensors=50,
    #     path_segments=5
    # )
    
    # Create an empty path history for the initial visualization
    # path_history = [(mc.x, mc.y)]  # Start with current position of MC
    
    # # Set varied energy levels and consumption rates for visualization
    # for i, s in enumerate(sensors):
    #     if i % 5 == 0:  # 20% very low
    #         s.energy = 0.05 * s.capacity
    #     elif i % 5 == 1:  # 20% low
    #         s.energy = 0.2 * s.capacity
    #     elif i % 5 == 2:  # 20% medium
    #         s.energy = 0.5 * s.capacity
            
    #     # Vary consumption rates to create different priorities
    #     if i % 3 == 0:  # High consumption
    #         s.consumption_rate = random.uniform(0.8, 1.5)
    #     elif i % 3 == 1:  # Medium consumption
    #         s.consumption_rate = random.uniform(0.4, 0.7)
    #     else:  # Low consumption
    #         s.consumption_rate = random.uniform(0.1, 0.3)
    
    # # Visualize initial priorities
    # print("Initial sensor priorities:")
    # visualize_sensor_priorities(sensors, mc, path_history)
    
    
    # print("\nRunning adaptive continuous charging simulation...")
    # run_adaptive_continuous_charging(
    #     num_steps=10,
    #     time_step=3,
    #     num_sensors=40,
    #     path_segments=8
    # )
    
    print("\nRunning adaptive continuous charging with zone-based efficiency...")
    sensors, mc, path_history = run_adaptive_continuous_charging_with_zones(
        num_steps=8,
        time_step=3,
        num_sensors=40,
        path_segments=8
    )

    # Visualize final state with complete path and charging zones
    print("\nFinal network state with complete mobile charger path and charging zones:")
    visualize_sensor_priorities_with_zones(sensors, mc, path_history)
    
    # Run simulation using the optimal position strategy
    # print("\nRunning simulation with optimal position strategy...")
    # sensors, mc, path_history = run_optimal_position_simulation(
    #     num_steps=6,      # Fewer steps as this is more computationally intensive
    #     time_step=3,
    #     num_sensors=40
    # )

    # # Show final state
    # print("\nFinal network state with complete mobile charger path and charging zones:")
    # visualize_sensor_priorities_with_zones(sensors, mc, path_history)
    
    
    
    print("\nSimulation complete!")

if __name__ == "__main__":
    main()