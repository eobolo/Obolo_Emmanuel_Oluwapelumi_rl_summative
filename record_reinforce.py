import asyncio
import platform
import time
import numpy as np
import pygame
import imageio
import cv2
import torch
import torch.nn as nn
from environment.custom_env import P3Environment
from gymnasium import spaces

# Define the Policy Network (same as in training)
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

# Function to flatten observation dictionary into 1D array (same as in training)
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

# Main demonstration function
FPS = 30
DEMO_DURATION = 180  # 3 minutes in seconds

async def demonstrate_agent(env, model, save_gif=False, gif_filename="reinforce_agent_demo.gif"):
    frames = [] if save_gif else None
    model.eval()  # Set model to evaluation mode
    start_time = time.time()  # Track start time
    episode = 0
    print("Starting demonstration for at least 3 minutes...")
    
    while (time.time() - start_time) < DEMO_DURATION:
        episode += 1
        obs, _ = env.reset()
        state_flat = flatten_observation(obs)
        done = False
        total_reward = 0
        step_count = 0
        print(f"\nEpisode {episode} Initial State: {obs}")
        
        while not done and (time.time() - start_time) < DEMO_DURATION:
            if not env.is_paused():  # Skip step if paused
                # Select action using the REINFORCE policy
                state_tensor = torch.FloatTensor(state_flat)
                probs = model(state_tensor)
                action_dist = torch.distributions.Categorical(probs)
                action_idx = action_dist.sample().item()
                action = env.decode_action(action_idx)  # Decode to tuple
                print(f"Step {step_count + 1}: Action Index = {action_idx}, Decoded Action = {action}")
                # Step the environment
                obs, reward, terminated, truncated, _ = env.step(action)
                state_flat = flatten_observation(obs)
                print(f"State: {obs}, Reward: {reward:.2f}")
                total_reward += reward
                step_count += 1
                done = terminated or truncated
            # Render the environment
            env.render()
            # Capture frame for GIF if enabled
            if save_gif and hasattr(env, 'screen') and env.screen is not None:
                frame = np.flipud(np.array(pygame.surfarray.array3d(env.screen)))  # Flip vertically
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # Rotate 90 degrees
                frames.append(frame)
            await asyncio.sleep(1.0 / FPS)  # Control frame rate
        
        print(f"Episode {episode}: Total Reward = {total_reward:.2f}, Steps = {step_count}")
    
    # Print total runtime
    total_time = time.time() - start_time
    print(f"\nDemonstration completed. Total runtime: {total_time:.2f} seconds")
    
    # Save GIF if enabled
    if save_gif and frames:
        imageio.mimsave(gif_filename, frames, fps=30)
        print(f"GIF saved as {gif_filename}")
        # Save first frame for debugging
        cv2.imwrite("debug_frame.png", frames[0])
        print("Debug frame saved as debug_frame.png")
    
    # Clean up
    env.close()

async def main():
    # Initialize environment
    env = P3Environment(max_steps=2000, render_mode="human")  # Match training max_steps
    
    # Initialize PolicyNetwork
    input_dim = sum(1 if isinstance(space, spaces.Discrete) else np.prod(space.shape) 
                    for space in env.observation_space.spaces.values())  # 16 dimensions
    output_dim = env.action_space.n  # 8000 actions
    model = PolicyNetwork(input_dim, output_dim)
    
    # Load the trained model
    experiment_value = "1"  # Replace with the actual experiment name used in training
    model_path = f"model/pg/reinforce_model_{experiment_value}.pth"
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return
    
    # Run demonstration
    await demonstrate_agent(env, model, save_gif=True)

# Run the demonstration
if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())