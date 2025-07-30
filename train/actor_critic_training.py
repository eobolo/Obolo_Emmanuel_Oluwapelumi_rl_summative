import os
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env_continuous import P3EnvironmentContinuous

# Define hyperparameters for each experiment
hyperparams_dict = {
    1: {
        'learning_rate': 0.0001,
        'gamma': 1.0,
        'n_steps': 4096,
        'gae_lambda': 0.95,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': dict(net_arch=[64, 64])
    },
    2: {
        'learning_rate': 0.0003,
        'gamma': 1.0,
        'n_steps': 2048,
        'gae_lambda': 0.95,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': dict(net_arch=[64, 64])
    },
    3: {
        'learning_rate': 0.0002,
        'gamma': 1.0,
        'n_steps': 4096,
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
        self.timesteps = []
        self.log_data = []
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None:
            for done, reward in zip(self.locals['dones'], self.locals['rewards']):
                if done:
                    self.episode_rewards.append(self.locals['infos'][0].get('episode', {}).get('r', reward))
                    self.episode_lengths.append(self.locals['infos'][0].get('episode', {}).get('l', 0))
        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            mean_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
            self.timesteps.append(self.num_timesteps)
            self.log_data.append({
                'timestep': self.num_timesteps,
                'mean_reward_last_100': mean_reward,
                'mean_episode_length_last_100': mean_length
            })
            pd.DataFrame(self.log_data).to_csv(self.log_file, index=False)
        return True

if __name__ == "__main__":
    # Parse experiment number from command line
    experiment_value = 1
    if len(sys.argv) > 1:
        try:
            experiment_value = int(sys.argv[1])
        except Exception:
            print("Invalid experiment number, defaulting to 1.")
            experiment_value = 1
    print(f"Running Actor-Critic Experiment {experiment_value}")
    hyperparams = hyperparams_dict.get(experiment_value, hyperparams_dict[1])
    print("Using hyperparameters:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")
    env = P3EnvironmentContinuous()
    model = A2C('MultiInputPolicy', env, verbose=1, **hyperparams)
    log_file = f"logs/training_log_actor_critic_{experiment_value}.csv"
    callback = TrainingLoggerCallback(check_freq=1000, log_file=log_file)
    model.learn(total_timesteps=200000, callback=callback)
    model.save(f"model/pg/actor_critic_model_{experiment_value}")
    print("Training complete. Model saved.")