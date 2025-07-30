import os
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env_continuous import P3EnvironmentContinuous

# Define hyperparameters for each experiment
hyperparams_dict = {
    1: {
        'learning_rate': 0.0001,
        'gamma': 1.0,
        'n_steps': 4096,
        'batch_size': 128,
        'ent_coef': 0.0,
        'clip_range': 0.1,
        'n_epochs': 10,
        'gae_lambda': 0.95,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': dict(net_arch=[64, 64])
    },
    2: {
        'learning_rate': 0.0003,
        'gamma': 1.0,
        'n_steps': 2048,
        'batch_size': 128,
        'ent_coef': 0.01,
        'clip_range': 0.2,
        'n_epochs': 10,
        'gae_lambda': 0.95,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': dict(net_arch=[64, 64])
    },
    3: {
        'learning_rate': 0.0002,
        'gamma': 1.0,
        'n_steps': 4096,
        'batch_size': 128,
        'ent_coef': 0.02,
        'clip_range': 0.25,
        'n_epochs': 10,
        'gae_lambda': 0.95,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': dict(net_arch=[64, 64])
    },
}

# Custom callback for logging
class TrainingLoggerCallback(BaseCallback):
    def __init__(self, check_freq, log_file, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_file = log_file
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []  # To store total loss (or policy_gradient_loss/value_loss)
        self.entropies = []  # Optional: to store positive entropy
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        df = pd.DataFrame(columns=['timestep', 'mean_reward_last_100', 'mean_episode_length_last_100', 'mean_loss_last_100', 'mean_entropy_last_100'])
        df.to_csv(self.log_file, mode='w', header=True, index=False)
        if verbose:
            print(f"Initialized CSV at {self.log_file} with headers")

    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None:
            for done, reward, info in zip(self.locals['dones'], self.locals['rewards'], self.locals['infos']):
                if done:
                    episode_reward = info.get('episode', {}).get('r', reward)
                    episode_length = info.get('episode', {}).get('l', 0)
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    if self.verbose:
                        print(f"Episode done: Reward = {episode_reward:.2f}, Length = {episode_length}")
        # Log total loss (or choose 'train/policy_gradient_loss' or 'train/value_loss')
        loss_key = 'train/loss'  # Options: 'train/policy_gradient_loss', 'train/value_loss'
        loss = self.logger.name_to_value.get(loss_key, 0) if hasattr(self.logger, 'name_to_value') else 0
        entropy = abs(self.logger.name_to_value.get('rollout/entropy_loss', 0)) if hasattr(self.logger, 'name_to_value') else 0
        if self.verbose and loss != 0:
            print(f"Captured {loss_key}: {loss:.2f}")
        if self.verbose and entropy != 0:
            print(f"Captured entropy: {entropy:.2f}")
        self.losses.append(loss)
        self.entropies.append(entropy)
        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            mean_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
            mean_loss = np.mean(self.losses[-100:]) if self.losses else 0
            mean_entropy = np.mean(self.entropies[-100:]) if self.entropies else 0
            df = pd.DataFrame({
                'timestep': [self.num_timesteps],
                'mean_reward_last_100': [mean_reward],
                'mean_episode_length_last_100': [mean_length],
                'mean_loss_last_100': [mean_loss],
                'mean_entropy_last_100': [mean_entropy]
            })
            df.to_csv(self.log_file, mode='a', header=False, index=False)
            if self.verbose:
                print(f"Logged at timestep {self.num_timesteps}: Mean Reward = {mean_reward:.2f}, Mean Length = {mean_length:.2f}, Mean Loss = {mean_loss:.2f}, Mean Entropy = {mean_entropy:.2f}")
        return True

if __name__ == "__main__":
    experiment_value = 1
    if len(sys.argv) > 1:
        try:
            experiment_value = int(sys.argv[1])
        except Exception:
            print("Invalid experiment number, defaulting to 1.")
            experiment_value = 1
    print(f"Running PPO Experiment {experiment_value}")
    hyperparams = hyperparams_dict.get(experiment_value, hyperparams_dict[1])
    print("Using hyperparameters:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")
    env = P3EnvironmentContinuous()
    model = PPO('MultiInputPolicy', env, verbose=1, **hyperparams)
    log_file = f"logs/training_log_ppo_{experiment_value}.csv"
    callback = TrainingLoggerCallback(check_freq=1000, log_file=log_file, verbose=1)
    model.learn(total_timesteps=200000, callback=callback)

    # Plotting
    try:
        data = pd.read_csv(log_file)
        expected_columns = ['timestep', 'mean_reward_last_100', 'mean_episode_length_last_100', 'mean_loss_last_100', 'mean_entropy_last_100']
        if not all(col in data.columns for col in expected_columns):
            raise KeyError(f"CSV missing expected columns: {expected_columns}")
        if data.empty:
            raise ValueError("CSV file is empty")
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(data['timestep'], data['mean_reward_last_100'], label='Mean Reward')
        plt.xlabel('Timestep')
        plt.ylabel('Mean Reward')
        plt.title('Mean Reward Over Time')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(data['timestep'], data['mean_loss_last_100'], label='Mean Loss', color='r')
        plt.xlabel('Timestep')
        plt.ylabel('Loss')
        plt.title('Objective Function (Total Loss) Over Time')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"graph/ppo_metrics_{experiment_value}.png")
        plt.close()
        print(f"Plots saved as graph/ppo_metrics_{experiment_value}.png")
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error plotting metrics: {e}. Skipping plotting, but training completed.")
        plt.figure(figsize=(12, 4))
        plt.text(0.5, 0.5, f"Plotting failed: {e}", ha='center', va='center')
        plt.savefig(f"graph/ppo_metrics_{experiment_value}_failed.png")
        plt.close()

    model.save(f"model/pg/ppo_model_{experiment_value}")
    print(f"PPO training completed and model saved as ppo_model_{experiment_value}.zip")