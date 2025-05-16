import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from environment import (
    get_valid_actions, 
    explain_action, 
    count_dead_sensors, 
    get_state_vector,
    initialize_environment,  # Added this import
    calculate_reward  # Added this import
)
from config import NUM_SENSORS, MC_CAPACITY

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=256):  # Increased from 128 to 256
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Add batch normalization
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim//2)  # Additional layer
        self.bn3 = nn.BatchNorm1d(hidden_dim//2)
        self.advantage = nn.Linear(hidden_dim//2, action_size)  # Dueling architecture
        self.value = nn.Linear(hidden_dim//2, 1)  # Dueling architecture

    def forward(self, x):
        # Apply batch norm only during training (when batch size > 1)
        if x.shape[0] > 1:
            x = F.leaky_relu(self.bn1(self.fc1(x)))  # Leaky ReLU instead of ReLU
            x = F.leaky_relu(self.bn2(self.fc2(x)))
            x = F.leaky_relu(self.bn3(self.fc3(x)))
        else:
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
            x = F.leaky_relu(self.fc3(x))
            
        advantage = self.advantage(x)
        value = self.value(x)
        # Combine value and advantage (dueling architecture)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.buffer = []
        self.capacity = capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling weight
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, done):
        max_priority = self.max_priority if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.size += 1
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # New experiences get max priority to ensure they're sampled
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        if self.size < batch_size:
            indices = np.random.choice(self.size, batch_size, replace=True)
        else:
            # Prioritized sampling
            priorities = self.priorities[:self.size] ** self.alpha
            probabilities = priorities / priorities.sum()
            indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        self.beta = min(1.0, self.beta + self.beta_increment)  # Increase beta
        
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
            indices
        )
    
    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = min(error + 1e-5, 1000)  # Small epsilon to prevent 0 priority
            self.max_priority = max(self.max_priority, self.priorities[idx])
            
    def __len__(self):
        return self.size

class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())  # sync weights

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        self.loss_fn = nn.MSELoss()
        self.replay_buffer = PrioritizedReplayBuffer()

    def act(self, state, valid_actions):
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)  # Explore
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            q_values = q_values.squeeze().detach().numpy()

            # Mask invalid actions
            masked_q = np.full(self.action_size, -np.inf)
            masked_q[valid_actions] = q_values[valid_actions]

            return int(np.argmax(masked_q))  # Exploit best valid action

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        # Get batch with importance sampling weights
        states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(batch_size)

        # Compute target Q-values with double DQN
        with torch.no_grad():
            # Get actions from policy network
            next_q_policy = self.q_network(next_states)
            next_actions = next_q_policy.max(1)[1].unsqueeze(1)
            
            # Get Q-values from target network for those actions
            next_q_target = self.target_network(next_states)
            next_q_vals = next_q_target.gather(1, next_actions).squeeze()
            
            targets = rewards + (1 - dones) * self.gamma * next_q_vals

        # Compute current Q-values
        current_q_vals = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute TD errors for prioritized replay
        td_errors = torch.abs(targets - current_q_vals).detach().cpu().numpy()
        
        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Compute weighted loss
        losses = F.smooth_l1_loss(current_q_vals, targets, reduction='none')
        loss = (losses * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()

        # Step the scheduler
        self.scheduler.step(loss)

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save the trained model to disk"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load a trained model from disk"""
        if not os.path.exists(path):
            print(f"No saved model found at {path}")
            return False
        
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {path}")
        return True

def train_dqn(agent, episodes=500, max_steps=200, batch_size=64, target_update=10):
    episode_rewards = []

    for ep in range(episodes):
        sensors, mc = initialize_environment()
        state = get_state_vector(sensors, mc)
        total_reward = 0
        done = False
        dead_before = count_dead_sensors(sensors)

        for step in range(max_steps):
            valid_actions = get_valid_actions(sensors, mc)
            if not valid_actions:
                print(f"  Step {step}: No valid actions available.")
                break

            action = agent.act(state, valid_actions)
            action_desc = explain_action(action)
            print(f"  Step {step}: Action {action} → {action_desc}")

            # --- Execute action ---
            if action == NUM_SENSORS:
                mc.move_to(300, 300)
                mc.energy = MC_CAPACITY  # recharge fully
                reward = -1.0
                next_state = get_state_vector(sensors, mc)
                done = False
            else:
                target = sensors[action]
                distance = np.linalg.norm([mc.x - target.x, mc.y - target.y])
                move_ok = mc.move_to(target.x, target.y)

                if not move_ok:
                    reward = -5
                    done = True
                else:
                    # Charge all nodes in radius instead of just one
                    charged_nodes, total_energy = mc.charge_nodes_in_radius(sensors)
                    
                    # Update reward based on how many nodes were charged
                    num_charged = len(charged_nodes)

                    # Simulate energy drain
                    for s in sensors:
                        s.update_energy(30)  # larger step to apply Fix 2

                    dead_after = count_dead_sensors(sensors)
                    
                    # Calculate next_state BEFORE using it in calculate_reward
                    next_state = get_state_vector(sensors, mc)
                    
                    # Now we can use next_state in calculate_reward
                    base_reward = calculate_reward(state, next_state, action, mc, sensors)
                    
                    # Bonus for charging multiple nodes
                    multi_charge_bonus = 0
                    if num_charged > 0:
                        multi_charge_bonus = 15.0 + (num_charged - 1) * 8.0  # Extra bonus for each additional node
                    
                    reward = base_reward + multi_charge_bonus
                    
                    # Update the dead count
                    dead_before = dead_after
                    done = mc.energy <= 0 or dead_after == NUM_SENSORS

            # Store and train
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train(batch_size)

            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()

        # Periodically update target network
        if ep % target_update == 0:
            agent.update_target_network()

        # Final episode-level bonus
        if not done and dead_before == 0:
            total_reward += 25  # bonus for keeping all sensors alive

        # Log results
        dead_now = count_dead_sensors(sensors)
        print(f"Episode {ep+1}/{episodes} - Reward: {total_reward:.2f}, Dead: {dead_now}, Epsilon: {agent.epsilon:.3f}")

        episode_rewards.append(total_reward)

    return episode_rewards

def evaluate_agent(agent, episodes=5, max_steps=200, time_step=30):
    print("\n--- Evaluation Mode (Epsilon = 0.0) ---")
    agent.epsilon = 0.0  # greedy policy
    results = []

    for ep in range(episodes):
        sensors, mc = initialize_environment()
        state = get_state_vector(sensors, mc)
        total_reward = 0
        dead_before = count_dead_sensors(sensors)

        print(f"\nEpisode {ep+1}:")

        for step in range(max_steps):
            valid_actions = get_valid_actions(sensors, mc)
            if not valid_actions:
                print(f"  Step {step}: No valid actions left.")
                break

            action = agent.act(state, valid_actions)
            action_desc = explain_action(action)
            print(f"  Step {step}: Action {action} → {action_desc}")

            if action == NUM_SENSORS:
                mc.move_to(300, 300)
                mc.energy = MC_CAPACITY
                reward = -0.1
                next_state = get_state_vector(sensors, mc)
            else:
                target = sensors[action]
                distance = np.linalg.norm([mc.x - target.x, mc.y - target.y])
                move_ok = mc.move_to(target.x, target.y)

                if not move_ok:
                    reward = -5
                    break
                else:
                    charged = mc.charge_node(target)

                    for s in sensors:
                        s.update_energy(time_step)

                    dead_after = count_dead_sensors(sensors)
                    next_state = get_state_vector(sensors, mc)

                    # Use the updated calculate_reward function
                    reward = calculate_reward(state, next_state, action, mc, sensors)
                    if charged > 0:
                        reward += 15.0

            state = next_state
            total_reward += reward

        dead_final = count_dead_sensors(sensors)
        print(f"Episode {ep+1} Complete → Total Reward: {total_reward:.2f}, Dead Sensors: {dead_final}")
        charged_count = sum(1 for s in sensors if s.energy > 0.95 * s.capacity)
        print(f"  Sensors fully charged: {charged_count}")

        results.append((total_reward, dead_final))

    return results