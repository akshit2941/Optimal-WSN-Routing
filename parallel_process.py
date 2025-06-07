import numpy as np
import os
import random
import torch
import time
import pandas as pd
import multiprocessing
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

def run_simulation_for_sensor_count(sensor_count, num_runs=5, num_steps=50, time_step=30):
    """
    Run simulation for a specific sensor count
    """
    print(f"Starting process for {sensor_count} sensors...")
    
    # Create a timestamp for the output file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"simulation_results_{sensor_count}_sensors_{timestamp}.xlsx"
    
    print(f"Starting {num_runs} simulation runs with {sensor_count} sensors...")
    print(f"Results will be saved to: {output_file}")
    
    # Create a DataFrame to store results
    results_df = pd.DataFrame()
    
    # Generate a master seed for this batch run
    master_seed = int(time.time() * (1 + sensor_count/1000)) % 10000
    print(f"Master seed for {sensor_count} sensors: {master_seed}")
    
    # Initial setup and demonstration
    print(f"Initializing WRSN environment with {sensor_count} sensors...")
    
    # Override NUM_SENSORS in the global scope for this process
    global NUM_SENSORS
    NUM_SENSORS = sensor_count
    
    # Use consistent seed for initial setup
    random.seed(master_seed)
    np.random.seed(master_seed)
    torch.manual_seed(master_seed)
    sensors, mc = initialize_environment(num_sensors=sensor_count)
    
    print(f"\nBasic environment initialization complete for {sensor_count} sensors.")
    
    # Set up DQN agent with sensor-count-specific naming
    print(f"\nSetting up DQN agent for {sensor_count} sensors...")
    state_size = 1 + sensor_count * 4
    action_size = sensor_count + 1
    agent = DQNAgent(state_size, action_size)
    
    # Path for saved model with sensor count in name
    model_path = f"trained_wrsn_agent_{sensor_count}_sensors.pth"
    
    # Check if we want to train or load a pretrained model
    train_new_model = False  # Set to False to load pretrained model
    
    if train_new_model or not os.path.exists(model_path):
        # Train agent
        print(f"\nTraining DQN agent for {sensor_count} sensors...")
        rewards = train_dqn(agent, episodes=50)
        
        # Save the trained model
        agent.save(model_path)
        
        # Since visualization is disabled, just print training stats
        print(f"Training completed for {sensor_count} sensors. Final reward: {rewards[-1]:.2f}")
        print(f"Average reward over last 10 episodes: {np.mean(rewards[-10:]):.2f}")
    else:
        # Load the pretrained model
        print(f"\nLoading pretrained DQN agent for {sensor_count} sensors...")
        agent.load(model_path)
    
    # Evaluate agent
    print(f"\nEvaluating trained agent for {sensor_count} sensors...")
    results = evaluate_agent(agent, episodes=3)
    
    # Display evaluation summary
    rewards_eval, deaths_eval = zip(*results)
    print(f"\n--- Evaluation Summary for {sensor_count} sensors ---")
    print(f"Average Evaluation Reward: {np.mean(rewards_eval):.2f}")
    print(f"Average Dead Sensors: {np.mean(deaths_eval):.2f}")
    
    # Suppress visualization
    import enhanced_visualization
    original_visualize = enhanced_visualization.visualize_optimal_position
    enhanced_visualization.visualize_optimal_position = lambda *args, **kwargs: None
    
    # Generate a list of random seeds for each run
    run_seeds = []
    for i in range(num_runs):
        # Create seed that's unique for this sensor count
        seed = (master_seed * (i + 1) * sensor_count) % 100000
        run_seeds.append(seed)
    
    # Run the simulation multiple times with different seeds
    print(f"\nRunning {num_runs} simulations for {sensor_count} sensors...")
    
    # Run the actual simulations
    for run, seed in enumerate(run_seeds, 1):
        print(f"\nRun {run}/{num_runs} - Using seed {seed} for {sensor_count} sensors")
        
        # Set the random seed for this run
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Suppress detailed output during simulation
        import sys
        from io import StringIO
        original_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # Run the simulation with optimal position strategy
            sensors, mc, path_history, metrics = run_optimal_position_simulation(
                num_steps=num_steps,
                time_step=time_step,
                num_sensors=sensor_count,
                random_seed=seed  # Pass the seed from each run
            )
            
            # Restore stdout
            sys.stdout = original_stdout
            print(f"  Run {run} completed successfully for {sensor_count} sensors.")
            
            # Create a dictionary for this run's results
            run_results = {
                "Sensor Count": sensor_count,
                "Run": run,
                "Seed": seed,
                "Alive Sensors": int(metrics['alive_ratio']*sensor_count),
                "Alive Percentage": metrics['alive_ratio']*100,
                "Dead Sensors": int(sensor_count - metrics['alive_ratio']*sensor_count),
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
            print(f"  Error in run {run} for {sensor_count} sensors: {str(e)}")
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
            numeric_cols.remove("Seed")  # Remove Seed from statistics
        if "Run" in numeric_cols:
            numeric_cols.remove("Run")   # Remove Run from statistics
        if "Sensor Count" in numeric_cols:
            numeric_cols.remove("Sensor Count")  # Remove Sensor Count from statistics
        
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
                results_df.to_excel(writer, sheet_name='Simulation_Results', index=False)
                summary_df.to_excel(writer, sheet_name='Summary_Statistics')
            print(f"\nBatch simulation complete for {sensor_count} sensors. Results saved to {output_file}")
            
        except Exception as e:
            print(f"Excel writing error for {sensor_count} sensors: {e}")
            # Fall back to CSV if Excel fails
            results_df.to_csv(output_file.replace(".xlsx", ".csv"), index=False)
            summary_df.to_csv(output_file.replace(".xlsx", f"_summary_{sensor_count}_sensors.csv"))
            print(f"Results saved as CSV files instead for {sensor_count} sensors.")
    else:
        print(f"No results to save for {sensor_count} sensors")
    
    print(f"Process completed for {sensor_count} sensors")
    return sensor_count, results_df

def run_parallel_simulations(sensor_counts=[100, 200, 300, 400, 500, 600, 800], num_runs=5, num_steps=50, time_step=30):
    """
    Run simulations for different sensor counts in parallel
    """
    print(f"Starting parallel simulations for sensor counts: {sensor_counts}")
    
    # Determine number of processes to use (limited by CPU cores)
    max_processes = min(len(sensor_counts), multiprocessing.cpu_count() - 1)
    print(f"Using {max_processes} parallel processes")
    
    # Create a pool of processes
    pool = multiprocessing.Pool(processes=max_processes)
    
    # Create tasks for each sensor count
    tasks = []
    for count in sensor_counts:
        tasks.append(pool.apply_async(run_simulation_for_sensor_count, 
                                     args=(count, num_runs, num_steps, time_step)))
    
    # Close the pool to new tasks
    pool.close()
    
    # Wait for all processes to complete and collect results
    all_results = {}
    for task in tasks:
        try:
            sensor_count, results_df = task.get()
            all_results[sensor_count] = results_df
            print(f"Collected results for {sensor_count} sensors")
        except Exception as e:
            print(f"Error collecting results: {e}")
    
    # Join the pool (wait for all processes to finish)
    pool.join()
    
    print("All parallel simulations completed!")
    
    # Create combined results file
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        combined_file = f"combined_results_{timestamp}.xlsx"
        
        with pd.ExcelWriter(combined_file, engine='openpyxl') as writer:
            # Write individual sheets for each sensor count
            for count, df in all_results.items():
                if not df.empty:
                    df.to_excel(writer, sheet_name=f'{count}_Sensors', index=False)
            
            # Create comparison sheet with key metrics
            comparison_data = []
            for count, df in all_results.items():
                if not df.empty:
                    avg_data = df.mean()
                    comparison_data.append({
                        "Sensor Count": count,
                        "Alive Percentage": avg_data.get("Alive Percentage", 0),
                        "Energy Efficiency": avg_data.get("Energy Efficiency", 0),
                        "Survival Rate": avg_data.get("Survival Rate", 0),
                        "Average Charging Delay (s)": avg_data.get("Average Charging Delay (s)", 0),
                        "Life-Survival Ratio": avg_data.get("Life-Survival Ratio", 0),
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
        
        print(f"Combined results saved to {combined_file}")
    except Exception as e:
        print(f"Error creating combined results: {e}")

if __name__ == "__main__":
    # Run parallel simulations with different sensor counts
    # Adjust num_runs to reduce simulation time if needed
    run_parallel_simulations(
        sensor_counts=[100, 200, 300, 400, 500, 600, 800],
        num_runs=1,  # Reduced from 25 for faster overall execution
        num_steps=50,
        time_step=30
    )