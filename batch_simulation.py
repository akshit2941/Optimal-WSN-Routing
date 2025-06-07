import numpy as np
import os
import random
import torch
import time
import pandas as pd
import sys
from io import StringIO
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

def setup_global_agent():
    """Set up and train the DQN agent once for all simulations"""
    print("Setting up global DQN agent...")
    
    # Initialize environment for agent setup
    sensors, mc = initialize_environment()
    
    # Set up DQN agent
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
        
        # Print training stats
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
    
    return model_path

def run_simulation_for_charging_rate(charging_rate, model_path, num_runs=25, num_steps=50, time_step=30):
    """
    Run simulation for a specific charging rate using the shared agent model
    """
    print(f"Starting simulations for CHARGING_RATE = {charging_rate} J/s...")
    
    # Create a timestamp for the output file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"simulation_results_rate_{charging_rate}Js_{timestamp}.xlsx"
    
    print(f"Starting {num_runs} simulation runs with CHARGING_RATE = {charging_rate} J/s...")
    print(f"Results will be saved to: {output_file}")
    
    # Create a DataFrame to store results
    results_df = pd.DataFrame()
    
    # Generate a master seed for this batch run
    master_seed = int(time.time() * (1 + charging_rate/10)) % 10000
    print(f"Master seed for CHARGING_RATE = {charging_rate} J/s: {master_seed}")
    
    # Override CHARGING_RATE in the global scope for this process
    global CHARGING_RATE
    CHARGING_RATE = charging_rate
    
    # Initial setup and demonstration
    print(f"Initializing WRSN environment with CHARGING_RATE = {charging_rate} J/s...")
    
    # Use consistent seed for initial setup
    random.seed(master_seed)
    np.random.seed(master_seed)
    torch.manual_seed(master_seed)
    
    # Initialize environment with updated CHARGING_RATE
    sensors, mc = initialize_environment()
    
    print(f"\nBasic environment initialization complete for CHARGING_RATE = {charging_rate} J/s.")
    
    # Set up DQN agent without training (use shared model)
    print(f"\nLoading shared DQN agent for CHARGING_RATE = {charging_rate} J/s...")
    state_size = 1 + NUM_SENSORS * 4
    action_size = NUM_SENSORS + 1
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    
    # Suppress visualization
    import enhanced_visualization
    original_visualize = enhanced_visualization.visualize_optimal_position
    enhanced_visualization.visualize_optimal_position = lambda *args, **kwargs: None
    
    # Generate a list of random seeds for each run
    run_seeds = []
    for i in range(num_runs):
        # Create seed that's unique for this run
        seed = (master_seed * (i + 1) * 31 + int(charging_rate * 100)) % 100000
        run_seeds.append(seed)
    
    # Print the seeds we'll be using
    print(f"\nUsing the following seeds for CHARGING_RATE = {charging_rate} J/s:")
    for i, seed in enumerate(run_seeds[:5]):
        print(f"  Run {i+1}: Seed {seed}")
    if len(run_seeds) > 5:
        print(f"  ... and {len(run_seeds)-5} more seeds")
    
    # Run the actual simulations
    for run, seed in enumerate(run_seeds, 1):
        print(f"\nRun {run}/{num_runs} - Using seed {seed} for CHARGING_RATE = {charging_rate} J/s")
        
        # Set the random seed for this run
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Suppress detailed output during simulation
        original_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # Run the simulation with optimal position strategy
            sensors, mc, path_history, metrics = run_optimal_position_simulation(
                num_steps=num_steps,
                time_step=time_step,
                num_sensors=NUM_SENSORS,
                random_seed=seed  # Pass the seed
            )
            
            # Restore stdout
            sys.stdout = original_stdout
            print(f"  Run {run} completed successfully for CHARGING_RATE = {charging_rate} J/s")
            
            # Create a dictionary for this run's results
            run_results = {
                "Charging Rate (J/s)": charging_rate,
                "Run": run,
                "Seed": seed,
                "Alive Sensors": int(metrics['alive_ratio']*NUM_SENSORS),
                "Alive Percentage": metrics['alive_ratio']*100,
                "Dead Sensors": int(NUM_SENSORS - metrics['alive_ratio']*NUM_SENSORS),
                "Dead Percentage": (1-metrics['alive_ratio'])*100,
                "Life-Survival Ratio": metrics['life_survival_ratio'],
                "Average Energy (J)": metrics['average_energy'],
                "Average Energy (%)": metrics['average_energy']/SENSOR_CAPACITY*100,
                "MC Final Energy (J)": mc.energy,
                "MC Final Energy (%)": mc.energy/MC_CAPACITY*100,
                "Total Positions Visited": len(path_history),
                "Survival Rate": metrics['survival_rate'],
                "Requested Sensors Count": len(metrics['sensors_requested']),
                "Charged Sensors Count": len(metrics['sensors_received']),
                "Neglected Sensors Count": len(metrics['sensors_requested']) - len(metrics['sensors_received']),
                "Neglected Percentage": ((len(metrics['sensors_requested']) - len(metrics['sensors_received']))/max(1, len(metrics['sensors_requested'])))*100,
                "Energy Efficiency": metrics['energy_efficiency'],
                "Energy Transferred (J)": metrics['total_energy_transferred'],
                "Movement Energy Cost (J)": metrics['total_movement_energy'],
                "Total Energy Used (J)": metrics['total_energy_transferred'] + metrics['total_movement_energy'],
                "Average Charging Delay (s)": metrics['avg_charging_delay'],
            }
            
            # Add additional metrics if available
            if metrics['charging_delays']:
                run_results["Min Delay (s)"] = min(metrics['charging_delays'])
                run_results["Max Delay (s)"] = max(metrics['charging_delays'])
                sorted_delays = sorted(metrics['charging_delays'])
                run_results["Median Delay (s)"] = sorted_delays[len(sorted_delays)//2]
            else:
                run_results["Min Delay (s)"] = 0
                run_results["Max Delay (s)"] = 0
                run_results["Median Delay (s)"] = 0
                
            if "life_survival_ratio_lifetime" in metrics:
                run_results["Life-Survival Ratio (Lifetime)"] = metrics['life_survival_ratio_lifetime']
                run_results["Network Lifetime with Charging (s)"] = metrics.get('current_time', 0.0)
                run_results["Est. Lifetime without Charging (s)"] = metrics.get('baseline_lifetime', 0.0)
            
            # Append to DataFrame
            results_df = pd.concat([results_df, pd.DataFrame([run_results])], ignore_index=True)
            
        except Exception as e:
            # Restore stdout and handle errors
            sys.stdout = original_stdout
            print(f"  Error in run {run} for CHARGING_RATE = {charging_rate} J/s: {str(e)}")
            continue
    
    # Restore original visualization function
    enhanced_visualization.visualize_optimal_position = original_visualize
    
    # Calculate summary statistics
    if not results_df.empty:
        # Create summary DataFrame
        summary_df = pd.DataFrame()
        
        # Select numeric columns
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
        if "Seed" in numeric_cols:
            numeric_cols.remove("Seed")
        if "Run" in numeric_cols:
            numeric_cols.remove("Run")
        if "Charging Rate (J/s)" in numeric_cols:
            numeric_cols.remove("Charging Rate (J/s)")
        
        # Calculate statistics
        summary_stats = {
            "Mean": results_df[numeric_cols].mean(),
            "Median": results_df[numeric_cols].median(),
            "Std Dev": results_df[numeric_cols].std(),
            "Min": results_df[numeric_cols].min(),
            "Max": results_df[numeric_cols].max()
        }
        
        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_stats)
        
        try:
            # Try newer pandas API
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name=f'Rate_{charging_rate}Js', index=False)
                summary_df.to_excel(writer, sheet_name=f'Summary_{charging_rate}Js')
            print(f"\nBatch simulation complete for CHARGING_RATE = {charging_rate} J/s. Results saved to {output_file}")
            
        except Exception as e:
            print(f"Excel writing error for CHARGING_RATE = {charging_rate} J/s: {e}")
            # Fall back to CSV if Excel fails
            results_df.to_csv(output_file.replace(".xlsx", ".csv"), index=False)
            summary_df.to_csv(output_file.replace(".xlsx", f"_summary_rate_{charging_rate}Js.csv"))
            print(f"Results saved as CSV files instead for CHARGING_RATE = {charging_rate} J/s.")
    else:
        print(f"No results to save for CHARGING_RATE = {charging_rate} J/s")
    
    print(f"Process completed for CHARGING_RATE = {charging_rate} J/s")
    return charging_rate, results_df

def run_sequential_charging_rate_simulations(charging_rates=[14.4], num_runs=25, num_steps=50, time_step=30):
    """
    Run simulations for different charging rates sequentially (one by one)
    """
    print(f"Starting sequential simulations for charging rates: {charging_rates} J/s")
    print(f"Using constant sensor count: {NUM_SENSORS}")
    
    # First, set up the global DQN agent that will be used by all simulations
    model_path = setup_global_agent()
    
    # Run simulations for each charging rate sequentially
    all_results = {}
    for rate in charging_rates:
        print(f"\n{'='*80}")
        print(f"  STARTING SIMULATIONS FOR CHARGING_RATE = {rate} J/s")
        print(f"{'='*80}\n")
        
        try:
            # Run the simulation for this charging rate
            charging_rate, results_df = run_simulation_for_charging_rate(rate, model_path, num_runs, num_steps, time_step)
            all_results[charging_rate] = results_df
            print(f"Completed simulations for CHARGING_RATE = {charging_rate} J/s")
        except Exception as e:
            print(f"Error running simulations for CHARGING_RATE = {rate} J/s: {str(e)}")
    
    print("All sequential simulations completed!")
    
    # Create combined results file
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        combined_file = f"combined_rate_results_{timestamp}.xlsx"
        
        with pd.ExcelWriter(combined_file, engine='openpyxl') as writer:
            # Write individual sheets for each charging rate
            for rate, df in all_results.items():
                if not df.empty:
                    df.to_excel(writer, sheet_name=f'Rate_{rate}Js', index=False)
            
            # Create comparison sheet with key metrics
            comparison_data = []
            for rate, df in all_results.items():
                if not df.empty:
                    avg_data = df.mean()
                    comparison_data.append({
                        "Charging Rate (J/s)": rate,
                        "Alive Percentage": avg_data.get("Alive Percentage", 0),
                        "Energy Efficiency": avg_data.get("Energy Efficiency", 0),
                        "Survival Rate": avg_data.get("Survival Rate", 0),
                        "Average Charging Delay (s)": avg_data.get("Average Charging Delay (s)", 0),
                        "Life-Survival Ratio": avg_data.get("Life-Survival Ratio", 0),
                        "Movement Energy Cost (J)": avg_data.get("Movement Energy Cost (J)", 0),
                        "Energy Transferred (J)": avg_data.get("Energy Transferred (J)", 0),
                        "Total Positions Visited": avg_data.get("Total Positions Visited", 0)
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df = comparison_df.sort_values(by="Charging Rate (J/s)")
                comparison_df.to_excel(writer, sheet_name='Rate_Comparison', index=False)
        
        print(f"Combined results saved to {combined_file}")
    except Exception as e:
        print(f"Error creating combined results: {e}")

if __name__ == "__main__":
    # Run sequential simulations with different charging rates
    run_sequential_charging_rate_simulations(
        charging_rates=[14.4],  # Charging rates to test
        num_runs=25,                           # 25 runs each as requested
        num_steps=50,                          # 50 simulation steps
        time_step=30                           # 30 seconds per step
    )