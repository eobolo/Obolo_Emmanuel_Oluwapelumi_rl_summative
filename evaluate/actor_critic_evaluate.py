import pandas as pd
import matplotlib.pyplot as plt
import os

def evaluate_actor_critic_model(experiment_number):
    """
    Evaluate Actor-Critic model and generate graphs.
    Args:
        experiment_number (int): The experiment number (1, 2, ...)
    """
    training_log_file = f"training_log_actor_critic_{experiment_number}.csv"
    log_path = f"logs/{training_log_file}"
    if not os.path.exists(log_path):
        print(f"ERROR: Log file {log_path} not found!")
        print("Make sure you have trained the Actor-Critic model first.")
        return
    print(f"Reading training log: {training_log_file}")
    log = pd.read_csv(log_path)
    os.makedirs("graph", exist_ok=True)
    # Graph 1: Training Reward Trend
    plt.figure(figsize=(12, 6))
    plt.plot(log['timestep'], log['mean_reward_last_100'],
             label=f'Actor-Critic Experiment {experiment_number} - Mean Reward (Last 100 Episodes)',
             linewidth=2, color='blue')
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Mean Reward', fontsize=12)
    plt.title(f'Actor-Critic Training Reward Trend - Experiment {experiment_number}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    reward_graph_path = f"graph/actor_critic_reward_experiment_{experiment_number}.png"
    plt.savefig(reward_graph_path, dpi=300, bbox_inches='tight')
    print(f"Reward graph saved: {reward_graph_path}")
    plt.show()
    # Graph 2: Training Episode Length Trend
    plt.figure(figsize=(12, 6))
    plt.plot(log['timestep'], log['mean_episode_length_last_100'],
             label=f'Actor-Critic Experiment {experiment_number} - Mean Episode Length (Last 100 Episodes)',
             linewidth=2, color='green')
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Mean Episode Length', fontsize=12)
    plt.title(f'Actor-Critic Training Episode Length Trend - Experiment {experiment_number}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    episode_graph_path = f"graph/actor_critic_episode_length_experiment_{experiment_number}.png"
    plt.savefig(episode_graph_path, dpi=300, bbox_inches='tight')
    print(f"Episode length graph saved: {episode_graph_path}")
    plt.show()
    # Print summary statistics
    print(f"\n=== Actor-Critic Experiment {experiment_number} Summary ===")
    print(f"Total timesteps: {log['timestep'].max():,}")
    print(f"Final mean reward: {log['mean_reward_last_100'].iloc[-1]:.2f}")
    print(f"Final mean episode length: {log['mean_episode_length_last_100'].iloc[-1]:.2f}")
    print(f"Best mean reward: {log['mean_reward_last_100'].max():.2f}")
    print(f"Best mean episode length: {log['mean_episode_length_last_100'].max():.2f}")

def evaluate_all_actor_critic_experiments():
    print("=== Actor-Critic Model Evaluation ===")
    for experiment in range(1, 5):
        print(f"\n--- Evaluating Actor-Critic Experiment {experiment} ---")
        try:
            evaluate_actor_critic_model(experiment)
        except Exception as e:
            print(f"Error evaluating experiment {experiment}: {e}")
            print("Skipping to next experiment...")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        experiment_num = int(sys.argv[1])
        if 1 <= experiment_num <= 5:
            evaluate_actor_critic_model(experiment_num)
        else:
            print("Experiment number must be between 1 and 4")
    else:
        evaluate_all_actor_critic_experiments()