from gymnasium import spaces
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import P3Environment
import pandas as pd
import matplotlib.pyplot as plt

# Define a simple Policy Network for REINFORCE
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)

# REINFORCE Algorithm
class REINFORCE:
    def __init__(self, input_dim, output_dim, learning_rate=0.001, gamma=0.99):
        self.policy_net = PolicyNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy_net(state)
        action_dist = Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy().item()  # Compute entropy for optional logging
        return action.item(), log_prob, entropy
    
    def update_policy(self, rewards, log_probs):
        G = 0
        returns = []
        for r in rewards[::-1]:
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        policy_loss = []
        for log_prob, ret in zip(log_probs, returns):
            policy_loss.append(-log_prob * ret)
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        return loss.item()  # Return loss for logging

# Custom callback to log reward, episode length, and policy loss
class TrainingLoggerCallback:
    def __init__(self, check_freq: int, log_file: str):
        self.check_freq = check_freq
        self.log_file = log_file
        self.rewards = []
        self.episode_lengths = []
        self.losses = []  # To store policy loss
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.timestep = 0
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        df = pd.DataFrame(columns=['timestep', 'mean_reward_last_100', 'mean_episode_length_last_100', 'mean_loss_last_100'])
        df.to_csv(self.log_file, mode='w', header=True, index=False)
        print(f"Initialized CSV at {self.log_file} with headers")

    def on_step(self, reward, done, loss=0):
        self.current_episode_reward += reward
        self.current_episode_length += 1
        self.timestep += 1
        if done:
            self.rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            print(f"Episode done: Reward = {self.current_episode_reward:.2f}, Length = {self.current_episode_length}")
            self.current_episode_reward = 0
            self.current_episode_length = 0
        if loss != 0:
            self.losses.append(loss)
        if self.timestep % self.check_freq == 0:
            mean_reward = np.mean(self.rewards[-100:]) if self.rewards else 0
            mean_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
            mean_loss = np.mean(self.losses[-100:]) if self.losses else 0
            df = pd.DataFrame({
                'timestep': [self.timestep],
                'mean_reward_last_100': [mean_reward],
                'mean_episode_length_last_100': [mean_length],
                'mean_loss_last_100': [mean_loss]
            })
            df.to_csv(self.log_file, mode='a', header=False, index=False)
            print(f"Logged at timestep {self.timestep}: Mean Reward = {mean_reward:.2f}, Mean Length = {mean_length:.2f}, Mean Loss = {mean_loss:.2f}")

if __name__ == "__main__":
    # Set up the environment
    env = P3Environment(max_steps=1000, render_mode=None)

    # Adjust hyperparameters for REINFORCE
    hyperparams = {
        'learning_rate': 0.0001,
        'gamma': 1.0
    }

    # Initialize REINFORCE
    if hasattr(env.observation_space, 'spaces'):
        input_dim = sum(1 if isinstance(space, spaces.Discrete) else np.prod(space.shape) for space in env.observation_space.spaces.values())
    else:
        input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = REINFORCE(input_dim, output_dim, learning_rate=hyperparams['learning_rate'], gamma=hyperparams['gamma'])

    # Function to flatten observation
    def flatten_observation(obs):
        return np.concatenate([
            obs['num_students'],
            obs['textbooks_kiny'],
            obs['textbooks_eng'],
            obs['textbooks_math'],
            obs['guides_kiny'],
            obs['guides_eng'],
            obs['guides_math'],
            obs['quality_kiny'],
            obs['quality_eng'],
            obs['quality_math'],
            obs['grant_usage'],
            obs['time_since_last_delivery'],
            np.array([obs['delivery_success_history']]),
            np.array([obs['urgency_level']]),
            obs['infrastructure_rating'],
            np.array([obs['location_type']])
        ])

    # Train the model
    experiment_value = input("Enter an experiment name (anything integers or strings): ")
    log_file = f"logs/training_log_reinforce_{experiment_value}.csv"
    callback = TrainingLoggerCallback(check_freq=1000, log_file=log_file)
    max_timesteps = 100000
    timestep = 0
    log_probs = []
    rewards = []

    while timestep < max_timesteps:
        state = env.reset()[0]
        state_flat = flatten_observation(state)
        done = False
        episode_log_probs = []
        episode_rewards = []
        
        while not done:
            action, log_prob, _ = agent.select_action(state_flat)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state_flat = flatten_observation(next_state)
            episode_log_probs.append(log_prob)
            episode_rewards.append(reward)
            state_flat = next_state_flat
            callback.on_step(reward, done)
            timestep += 1
            if timestep >= max_timesteps:
                break
        
        log_probs.extend(episode_log_probs)
        rewards.extend(episode_rewards)
        if len(rewards) >= 50:
            loss = agent.update_policy(rewards, log_probs)
            callback.on_step(0, False, loss=loss)  # Log loss after policy update
            log_probs = []
            rewards = []

    # Plotting
    try:
        data = pd.read_csv(log_file)
        expected_columns = ['timestep', 'mean_reward_last_100', 'mean_episode_length_last_100', 'mean_loss_last_100']
        if not all(col in data.columns for col in expected_columns):
            raise KeyError(f"CSV missing expected columns: {expected_columns}")
        if data.empty:
            raise ValueError("CSV file is empty")
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(data['timestep'], data['mean_reward_last_100'], label='Mean Reward')
        plt.xlabel('Timestep')
        plt.ylabel('Mean Reward')
        plt.title('Mean Reward Over Time')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(data['timestep'], data['mean_loss_last_100'], label='Mean Policy Loss', color='r')
        plt.xlabel('Timestep')
        plt.ylabel('Policy Loss')
        plt.title('Objective Function (Policy Loss) Over Time')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"graph/reinforce_metrics_{experiment_value}.png")
        plt.close()
        print(f"Plots saved as graph/reinforce_metrics_{experiment_value}.png")
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error plotting metrics: {e}. Skipping plotting.")

    # Save the model
    torch.save(agent.policy_net.state_dict(), f"model/pg/reinforce_model_{experiment_value}.pth")
    env.close()
    print(f"REINFORCE training completed and model saved as reinforce_model_{experiment_value}.pth")