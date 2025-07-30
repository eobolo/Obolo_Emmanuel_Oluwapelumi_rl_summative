import asyncio
import platform
import numpy as np
import pygame
from stable_baselines3 import PPO
from environment.custom_env_continuous import P3EnvironmentContinuous  # Import your environment class

# Main demonstration function
FPS = 30

async def main():
    # Initialize environment with human render mode
    env = P3EnvironmentContinuous(max_steps=1000, render_mode="human")
    
    # Load the trained PPO model
    # Note: In Pyodide, ensure the model file is preloaded or accessible
    try:
        model = PPO.load("model/pg/actor_critic_model_3")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Enhance render method to include urgency-based coloring and state display
    def enhanced_render(self):
        if self.render_mode != "human" or self.screen is None:
            return
        # Set background color based on urgency level
        urgency_colors = {
            0: (0, 255, 0),   # Low: Green
            1: (255, 255, 0), # Medium: Yellow
            2: (255, 0, 0)    # High: Red
        }
        self.sprites['background'].fill(urgency_colors.get(self.state['urgency_level'], (200, 150, 200)))
        self.screen.blit(self.sprites['background'], (0, 0))
        # Display state values
        font = pygame.font.Font(None, 24)
        y_offset = 10
        for key, value in self.state.items():
            text = font.render(f"{key}: {value}", True, (0, 0, 0))
            self.screen.blit(text, (10, y_offset))
            y_offset += 25
        # Display last action and reward
        action_text = f"Action: {self.last_action.get('action', 'None')}"
        reward_text = f"Reward: {self.last_action.get('reward', 0):.2f}"
        self.screen.blit(font.render(action_text, True, (0, 0, 0)), (10, y_offset))
        self.screen.blit(font.render(reward_text, True, (0, 0, 0)), (10, y_offset + 25))
        pygame.display.flip()
    
    # Override the environment's render method
    env.render = enhanced_render.__get__(env, P3EnvironmentContinuous)
    
    # Run 3 episodes
    for episode in range(3):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            # Get action from the trained model
            action, _ = model.predict(obs, deterministic=True)
            print(action)
            # Step the environment
            obs, reward, terminated, truncated, _ = env.step(action)
            print(obs)
            # Render the environment
            env.render()
            total_reward += reward
            step_count += 1
            done = terminated or truncated
            # Control frame rate
            await asyncio.sleep(1.0 / FPS)
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Steps = {step_count}")
    
    # Clean up
    env.close()

# Run the demonstration
if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())