"""
Start SmartDelivery 3D Visualization

This script starts the 3D visualization with the environment.
"""

import sys
import os
import time

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import P3Environment
from renderer.threejs_renderer import ThreeJSRenderer

def convert_numpy_state(state):
    """Convert NumPy types in state to Python native types."""
    converted_state = {}
    for key, value in state.items():
        if hasattr(value, 'dtype'):  # NumPy type
            if hasattr(value, 'item'):
                converted_state[key] = value.item()
            else:
                converted_state[key] = value
        else:
            converted_state[key] = value
    return converted_state

def convert_action(action):
    """Convert action to Python native types if it contains NumPy types."""
    if action is None:
        return None
    if isinstance(action, (list, tuple)):
        converted_action = []
        for item in action:
            if hasattr(item, 'dtype'):  # NumPy type
                if hasattr(item, 'item'):
                    converted_action.append(item.item())
                else:
                    converted_action.append(item)
            else:
                converted_action.append(item)
        return converted_action
    else:
        return action

def main():
    """Main function to start the 3D visualization."""
    print("Starting SmartDelivery 3D Visualization...")
    
    # Initialize environment and renderer
    env = P3Environment()
    renderer = ThreeJSRenderer()
    
    try:
        # Start the web server
        print("Starting web server...")
        renderer.start()
        
        # Wait for server to be ready
        time.sleep(2)
        
        # Health check
        try:
            import requests
            response = requests.get('http://127.0.0.1:5000', timeout=5)
            if response.status_code == 200:
                print("Web server is running at http://127.0.0.1:5000")
            else:
                print("Web server responded with status:", response.status_code)
        except Exception as e:
            print(f"Could not verify web server: {e}")
        
        # Reset environment
        state, _ = env.reset()
        state = convert_numpy_state(state)
        
        # Send initial state
        renderer.update_state(state)
        print(f"Initial state sent - Students: {state['num_students']}")
        
        # Main loop with hardcoded limits
        step_count = 0
        episode_count = 0
        max_steps = 1000  # Limit for GIF generation
        
        print(f"Starting simulation (max {max_steps} steps per episode)")
        
        while step_count < max_steps:
            # Sample action
            action = env.sample()
            action = convert_action(action)
            
            # Take step
            state, reward, terminated, truncated, info = env.step(action)
            state = convert_numpy_state(state)
            
            # Update visualization
            renderer.update_state(state, action)
            
            # Print progress every 10 steps
            step_count += 1
            if step_count % 10 == 0:
                print(f"Step {step_count}/{max_steps}: Students={state['num_students']}, "
                      f"Textbooks={state['textbooks_kiny'] + state['textbooks_eng'] + state['textbooks_math']}, "
                      f"Reward={reward:.2f}")
            
            # Check if episode is done
            if terminated or truncated:
                episode_count += 1
                state, _ = env.reset()
                state = convert_numpy_state(state)
                renderer.update_state(state)
                print(f"Episode {episode_count} reset (Step {step_count})")
            
            # Small delay for visualization
            time.sleep(0.5)
        
        print(f"Simulation completed: {step_count} steps, {episode_count} episodes")
        print("Ready for GIF recording! The visualization will continue running.")
        print("Press Ctrl+C to stop when you're done recording.")
        
        # Keep running for manual GIF recording
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping visualization...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("Cleaning up...")
        renderer.stop()
        env.close()
        print("Cleanup complete!")

if __name__ == "__main__":
    main() 