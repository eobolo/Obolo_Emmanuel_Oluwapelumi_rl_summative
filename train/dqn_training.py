import gymnasium as gym
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from environment.custom_env import P3Environment
import pandas as pd
import matplotlib.pyplot as plt

# Custom callback to log reward, episode length, and DQN loss
class TrainingLoggerCallback(BaseCallback):
    def __init__(self, check_freq: int, log_file: str, verbose=1):
        super(TrainingLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_file = log_file
        self.rewards = []
        self.episode_lengths = []
        self.losses = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        # Ensure log file directory exists and initialize CSV with headers
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        df = pd.DataFrame(columns=['timestep', 'mean_reward_last_100', 'mean_episode_length_last_100', 'mean_loss_last_100'])
        df.to_csv(self.log_file, mode='w', header=True, index=False)
        if self.verbose:
            print(f"Initialized CSV at {self.log_file} with headers")

    def _on_step(self) -> bool:
        # Accumulate rewards and episode length
        self.current_episode_reward += self.locals['rewards'][0] if self.locals['rewards'] else 0
        self.current_episode_length += 1
        # Log episode data when done
        if self.locals.get('dones', [False])[0]:
            self.rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            if self.verbose:
                print(f"Episode done: Reward = {self.current_episode_reward:.2f}, Length = {self.current_episode_length}")
            self.current_episode_reward = 0
            self.current_episode_length = 0
        # Log DQN loss from logger (train/loss)
        loss = self.logger.name_to_value.get('train/loss', 0) if hasattr(self.logger, 'name_to_value') else 0
        if self.verbose and loss != 0:
            print(f"Captured loss: {loss:.2f}")
        self.losses.append(loss)
        # Log metrics at check frequency
        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.rewards[-100:]) if self.rewards else 0
            mean_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
            mean_loss = np.mean(self.losses[-100:]) if self.losses else 0
            df = pd.DataFrame({
                'timestep': [self.num_timesteps],
                'mean_reward_last_100': [mean_reward],
                'mean_episode_length_last_100': [mean_length],
                'mean_loss_last_100': [mean_loss]
            })
            df.to_csv(self.log_file, mode='a', header=False, index=False)
            if self.verbose:
                print(f"Logged at timestep {self.num_timesteps}: Mean Reward = {mean_reward:.2f}, Mean Length = {mean_length:.2f}, Mean Loss = {mean_loss:.2f}")
        return True

if __name__ == "__main__":
    # Set up the environment
    env = P3Environment(max_steps=1000, render_mode=None)

    # Adjust hyperparameters (DQN #1 as default)
    hyperparams = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'batch_size': 32,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.02,
        'exploration_fraction': 0.1,
        'buffer_size': 10000,
        'learning_starts': 1000,
        'target_update_interval': 1000,
        'train_freq': 4
    }

    # Initialize DQN with MultiInputPolicy
    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=hyperparams['learning_rate'],
        gamma=hyperparams['gamma'],
        batch_size=hyperparams['batch_size'],
        exploration_initial_eps=hyperparams['exploration_initial_eps'],
        exploration_final_eps=hyperparams['exploration_final_eps'],
        exploration_fraction=hyperparams['exploration_fraction'],
        buffer_size=hyperparams['buffer_size'],
        learning_starts=hyperparams['learning_starts'],
        target_update_interval=hyperparams['target_update_interval'],
        train_freq=hyperparams['train_freq'],
        verbose=1,
        tensorboard_log="./dqn_tensorboard/"
    )

    # Train the model
    experiment_value = input("Enter an experiment name (anything integers or strings): ")
    log_file = f"logs/training_log_dqn_{experiment_value}.csv"
    callback = TrainingLoggerCallback(check_freq=1000, log_file=log_file)
    model.learn(total_timesteps=200000, callback=callback)

    # Save the model
    model.save(f"model/dqn/dqn_model_{experiment_value}")

    # Plotting
    try:
        data = pd.read_csv(log_file)
        # Verify CSV has expected columns
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
        plt.plot(data['timestep'], data['mean_loss_last_100'], label='Mean Loss', color='r')
        plt.xlabel('Timestep')
        plt.ylabel('Mean Loss')
        plt.title('Objective Function (Q-Loss) Over Time')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"graph/dqn_metrics_{experiment_value}.png")
        plt.close()
        print(f"Plots saved as dqn_metrics_{experiment_value}.png")
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error plotting metrics: {e}. Skipping plotting, but training completed.")
        # Save an empty plot to indicate failure
        plt.figure(figsize=(12, 4))
        plt.text(0.5, 0.5, f"Plotting failed: {e}", ha='center', va='center')
        plt.savefig(f"graph/dqn_metrics_{experiment_value}_failed.png")
        plt.close()

    # Close the environment
    env.close()

    print(f"DQN training completed and model saved as dqn_model_{experiment_value}.zip")