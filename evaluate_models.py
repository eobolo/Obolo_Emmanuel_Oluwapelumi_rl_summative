import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO, A2C
import torch
from environment.custom_env import P3Environment
from environment.custom_env_continuous import P3EnvironmentContinuous
from state.num_of_students_state import numStudentState
from state.textbook_available_state import sample_textbooks_available
from state.urgency_level_state import sample_urgency_level
from state.location_type_state import sample_location_type
from gymnasium import spaces
from train.reinforce_training import PolicyNetwork

# Test environment for discrete actions
class TestP3Environment(P3Environment):
    def __init__(self, max_steps=1000, render_mode=None, test_mode=False):
        super().__init__(max_steps, render_mode)
        self.test_mode = test_mode
        if test_mode:
            self._original_sample_num_students = numStudentState().sample_num_students
            self._original_sample_textbooks_available = sample_textbooks_available
            self._original_sample_urgency_level = sample_urgency_level
            self._original_sample_location_type = sample_location_type
            
            def test_sample_textbooks_available(self, num_students, location_type):
                t_kiny, t_eng, t_math = self._original_sample_textbooks_available(num_students, location_type)
                return tuple(int(x * 0.25) for x in (t_kiny, t_eng, t_math))
            self.test_sample_textbooks_available = test_sample_textbooks_available.__get__(self, TestP3Environment)
            
            def test_sample_urgency_level(self):
                categories = ['Low', 'Medium', 'High']
                probabilities = [0.1, 0.2, 0.7]
                urgency_str = np.random.choice(categories, p=probabilities)
                return {'Low': 0, 'Medium': 1, 'High': 2}[urgency_str]
            self.test_sample_urgency_level = test_sample_urgency_level.__get__(self, TestP3Environment)
            
            def test_sample_location_type(self):
                return 0 if np.random.random() < 0.9 else 1
            self.test_sample_location_type = test_sample_location_type.__get__(self, TestP3Environment)
            
            numStudentState.sample_num_students = lambda self: int(np.clip(
                np.random.normal(200 if np.random.random() < 0.7 else 300, 50 if np.random.random() < 0.7 else 60),
                50, 1500
            ))

    def reset(self, seed=None):
        state, _ = super().reset(seed=seed)
        if self.test_mode:
            numStudentState.sample_num_students = self._original_sample_num_students
        return state, {}

# Test environment for continuous actions
class TestP3EnvironmentContinuous(P3EnvironmentContinuous):
    def __init__(self, max_steps=1000, render_mode=None, test_mode=False):
        super().__init__(max_steps, render_mode)
        self.test_mode = test_mode
        if test_mode:
            self._original_sample_num_students = numStudentState().sample_num_students
            self._original_sample_textbooks_available = sample_textbooks_available
            self._original_sample_urgency_level = sample_urgency_level
            self._original_sample_location_type = sample_location_type
            
            def test_sample_textbooks_available(self, num_students, location_type):
                t_kiny, t_eng, t_math = self._original_sample_textbooks_available(num_students, location_type)
                return tuple(int(x * 0.25) for x in (t_kiny, t_eng, t_math))
            self.test_sample_textbooks_available = test_sample_textbooks_available.__get__(self, TestP3EnvironmentContinuous)
            
            def test_sample_urgency_level(self):
                categories = ['Low', 'Medium', 'High']
                probabilities = [0.1, 0.2, 0.7]
                urgency_str = np.random.choice(categories, p=probabilities)
                return {'Low': 0, 'Medium': 1, 'High': 2}[urgency_str]
            self.test_sample_urgency_level = test_sample_urgency_level.__get__(self, TestP3EnvironmentContinuous)
            
            def test_sample_location_type(self):
                return 0 if np.random.random() < 0.9 else 1
            self.test_sample_location_type = test_sample_location_type.__get__(self, TestP3EnvironmentContinuous)
            
            numStudentState.sample_num_students = lambda self: int(np.clip(
                np.random.normal(200 if np.random.random() < 0.7 else 300, 50 if np.random.random() < 0.7 else 60),
                50, 1500
            ))

    def reset(self, seed=None):
        state, _ = super().reset(seed=seed)
        if self.test_mode:
            numStudentState.sample_num_students = self._original_sample_num_students
        return state, {}

# Load REINFORCE model
def load_reinforce_model(model_path):
    state_dict = torch.load(model_path)
    input_dim = sum(1 if isinstance(space, spaces.Discrete) else np.prod(space.shape) for space in P3Environment().observation_space.spaces.values())
    output_dim = P3Environment().action_space.n
    policy_net = PolicyNetwork(input_dim, output_dim)
    policy_net.load_state_dict(state_dict)
    return policy_net

# Testing function
def test_model(model_type, model_path, env_class, log_file, n_episodes=100):
    env = env_class(max_steps=1000, render_mode=None, test_mode=True)
    if model_type == 'dqn':
        model = DQN.load(model_path)
    elif model_type == 'ppo':
        model = PPO.load(model_path)
    elif model_type == 'reinforce':
        model = load_reinforce_model(model_path)  # Load PolicyNetwork
    elif model_type == 'actor_critic':
        model = A2C.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    rewards = []
    lengths = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        while not done:
            if model_type in ['dqn', 'reinforce']:  # Discrete models
                if model_type == 'dqn':
                    action, _ = model.predict(state)
                    action = P3Environment.decode_action(action) # Decode to 6-tuple
                elif model_type == 'reinforce':
                    state_flat = np.concatenate([
                        state['num_students'], state['textbooks_kiny'], state['textbooks_eng'], state['textbooks_math'],
                        state['guides_kiny'], state['guides_eng'], state['guides_math'], state['quality_kiny'],
                        state['quality_eng'], state['quality_math'], state['grant_usage'], state['time_since_last_delivery'],
                        np.array([state['delivery_success_history']]), np.array([state['urgency_level']]),
                        state['infrastructure_rating'], np.array([state['location_type']])
                    ])
                    state_tensor = torch.FloatTensor(state_flat)
                    probs = model(state_tensor)
                    action_dist = torch.distributions.Categorical(probs)
                    action_idx = action_dist.sample().item()
                    action = P3Environment.decode_action(action_idx)  # Decode to 6-tuple
            elif model_type in ['ppo', 'actor_critic']:  # Continuous models
                action, _ = model.predict(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            state = next_state
            done = terminated or truncated
        rewards.append(episode_reward)
        lengths.append(episode_length)

    mean_reward = np.mean(rewards)
    mean_length = np.mean(lengths)
    baseline = {'dqn': 38000, 'ppo': 16852.97, 'reinforce': 7335.98, 'actor_critic': 13735.67}[model_type]
    retention = (mean_reward / baseline) * 100
    df = pd.DataFrame({'reward': rewards, 'length': lengths})
    df.to_csv(log_file, index=False)
    env.close()
    return mean_reward, mean_length, retention

# Model configurations
model_configs = [
    ('dqn', 'model/dqn/dqn_model_1.zip', TestP3Environment),
    ('ppo', 'model/pg/ppo_model_3.zip', TestP3EnvironmentContinuous),
    ('reinforce', 'model/pg/reinforce_model_1.pth', TestP3Environment),
    ('actor_critic', 'model/pg/actor_critic_model_3.zip', TestP3EnvironmentContinuous)
]

# Run tests
results = {}
for model_type, model_path, env_class in model_configs:
    log_file = f'logs/test_results_{model_type}.csv'
    mean_reward, mean_length, retention = test_model(model_type, model_path, env_class, log_file)
    results[model_type] = {'mean_reward': mean_reward, 'mean_length': mean_length, 'retention': retention}
    print(f"{model_type}: Mean Reward = {mean_reward}, Mean Length = {mean_length}, Retention = {retention:.2f}%")

# Save summary
summary_df = pd.DataFrame(results).T
summary_df.to_csv('logs/test_summary.csv')