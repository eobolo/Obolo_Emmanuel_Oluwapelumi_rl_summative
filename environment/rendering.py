import gymnasium as gym
import numpy as np
import imageio
import pygame
import cv2  # Add OpenCV for better color handling

# Assuming P3Environment is in environment/custom_env.py
from environment.custom_env import P3Environment

def generate_gif(env, filename="random_action.gif", steps=100):
    frames = []
    state, _ = env.reset()
    for _ in range(steps):
        if not env.is_paused():  # Skip step and render if paused
            action = env.sample()
            state, _, _, _, _ = env.step(action)
            env.render()
            # Capture frame and convert RGB to BGR using OpenCV, then rotate
            if hasattr(env, 'screen'):
                frame = np.flipud(np.array(pygame.surfarray.array3d(env.screen)))  # Flip vertically and convert to array
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # Rotate 90 degrees clockwise
                frames.append(frame)
        else:
            # Render the current frame without stepping if paused
            env.render()
            if hasattr(env, 'screen'):
                frame = np.flipud(np.array(pygame.surfarray.array3d(env.screen)))  # Flip vertically and convert to array
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # Rotate 90 degrees clockwise
                frames.append(frame)
            pygame.time.wait(100)  # Small delay to reduce CPU usage while paused
    if frames:
        imageio.mimsave(filename, frames, fps=30)
        print(f"GIF saved as {filename}")
    else:
        print("No frames captured. Ensure rendering is working.")
    # Debug: Save first frame as image to verify colors and orientation
    if frames:
        cv2.imwrite("debug_frame.png", frames[0])
        print("Debug frame saved as debug_frame.png")

if __name__ == "__main__":
    env = P3Environment(max_steps=100, render_mode="human")
    try:
        generate_gif(env)
    finally:
        env.close()