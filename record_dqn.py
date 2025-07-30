import asyncio
import platform
import numpy as np
import pygame
import imageio
import cv2
from stable_baselines3 import DQN
from environment.custom_env import P3Environment# Import environment and utility functions

# Main demonstration function
FPS = 30

async def demonstrate_agent(env, model, num_episodes=3, save_gif=False, gif_filename="dqn_agent_demo.gif"):
    frames = [] if save_gif else None
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            if not env.is_paused():  # Skip step if paused
                # Get action from the trained DQN model
                action, _ = model.predict(obs, deterministic=True)
                print(action)
                action = P3Environment.decode_action(action)
                print(action)
                # Step the environment
                obs, reward, terminated, truncated, _ = env.step(action)
                print(obs)
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
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Steps = {step_count}")
    
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
    env = P3Environment(max_steps=100, render_mode="human")
    
    # Load the trained DQN model
    try:
        model = DQN.load("model/dqn/dqn_model_4.zip")
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return
    
    # Run demonstration
    await demonstrate_agent(env, model, num_episodes=3, save_gif=True)

# Run the demonstration
if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())