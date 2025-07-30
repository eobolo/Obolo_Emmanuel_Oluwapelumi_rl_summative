#!/usr/bin/env python3
"""
Main entry point for running RL experiments on the Rwanda Textbook Distribution System.

This script provides a unified interface to train different RL algorithms:
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization) 
- REINFORCE
- Actor-Critic (A2C)

Usage (commands allowed):
    python main.py --algorithm <algorithm> --experiment <experiment_number>
    python main.py --help
    python main.py --list-experiments 

Examples:
    python main.py --algorithm dqn --experiment 1
    python main.py --algorithm ppo --experiment 2
    python main.py --algorithm reinforce --experiment 1
    python main.py --algorithm actor_critic --experiment 3
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_dqn_experiment(experiment_name):
    """Run DQN training experiment."""
    print(f"Starting DQN Experiment {experiment_name}")
    print("=" * 50)
    
    # DQN uses input() for experiment selection, so we need to handle this differently
    script_path = Path("train/dqn_training.py")
    if not script_path.exists():
        print(f"Error: {script_path} not found!")
        return False
    
    try:
        # Run the script and provide input
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.getcwd()
        )
        
        # Send the experiment name as input
        stdout, _ = process.communicate(input=str(experiment_name))
        print(stdout)
        
        if process.returncode == 0:
            print(f"DQN Experiment {experiment_name} completed successfully!")
            return True
        else:
            print(f"DQN Experiment {experiment_name} failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"Error running DQN experiment: {e}")
        return False

def run_ppo_experiment(experiment_name):
    """Run PPO training experiment."""
    print(f"Starting PPO Experiment {experiment_name}")
    print("=" * 50)
    
    script_path = Path("train/ppo_training.py")
    if not script_path.exists():
        print(f"Error: {script_path} not found!")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), str(experiment_name)],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        print(result.stdout)
        if result.stderr:
            print("Errors/Warnings:", result.stderr)
        
        if result.returncode == 0:
            print(f"PPO Experiment {experiment_name} completed successfully!")
            return True
        else:
            print(f"PPO Experiment {experiment_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"Error running PPO experiment: {e}")
        return False

def run_reinforce_experiment(experiment_name):
    """Run REINFORCE training experiment."""
    print(f"Starting REINFORCE Experiment {experiment_name}")
    print("=" * 50)
    
    script_path = Path("train/reinforce_training.py")
    if not script_path.exists():
        print(f"Error: {script_path} not found!")
        return False
    
    try:
        # REINFORCE uses input() for experiment selection
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.getcwd()
        )
        
        # Send the experiment name as input
        stdout, _ = process.communicate(input=str(experiment_name))
        print(stdout)
        
        if process.returncode == 0:
            print(f"REINFORCE Experiment {experiment_name} completed successfully!")
            return True
        else:
            print(f"REINFORCE Experiment {experiment_name} failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"Error running REINFORCE experiment: {e}")
        return False

def run_actor_critic_experiment(experiment_name):
    """Run Actor-Critic training experiment."""
    print(f"Starting Actor-Critic Experiment {experiment_name}")
    print("=" * 50)
    
    script_path = Path("train/actor_critic_training.py")
    if not script_path.exists():
        print(f"Error: {script_path} not found!")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), str(experiment_name)],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        print(result.stdout)
        if result.stderr:
            print("Errors/Warnings:", result.stderr)
        
        if result.returncode == 0:
            print(f"Actor-Critic Experiment {experiment_name} completed successfully!")
            return True
        else:
            print(f"Actor-Critic Experiment {experiment_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"Error running Actor-Critic experiment: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run RL experiments on Rwanda Textbook Distribution System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --algorithm dqn --experiment 1
  python main.py --algorithm ppo --experiment 2
  python main.py --algorithm reinforce --experiment 1
  python main.py --algorithm actor_critic --experiment 3
  
Available Algorithms:
  dqn           - Deep Q-Network (value-based method)
  ppo           - Proximal Policy Optimization (policy gradient)
  reinforce     - REINFORCE algorithm (policy gradient)
  actor_critic  - Actor-Critic (A2C) algorithm (policy gradient)
  
Experiment Numbers:
  DQN: 1-4 (different hyperparameter configurations)
  PPO: 1-3 (different hyperparameter configurations)
  REINFORCE: 1-3 (different hyperparameter configurations)
  Actor-Critic: 1-3 (different hyperparameter configurations)
        """
    )
    
    parser.add_argument(
        '--algorithm', '-a',
        choices=['dqn', 'ppo', 'reinforce', 'actor_critic'],
        help='RL algorithm to run'
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        help='Experiment number/name to run'
    )
    
    parser.add_argument(
        '--list-experiments',
        action='store_true',
        help='List available experiments for each algorithm'
    )
    
    args = parser.parse_args()
    
    if args.list_experiments:
        print("Available Experiments:")
        print("=" * 50)
        print("DQN Experiments:")
        print("  1: lr=0.001, γ=0.99, batch=32, buffer=10k")
        print("  2: lr=0.0005, γ=0.99, batch=32, buffer=50k")
        print("  3: lr=0.0005, γ=0.99, batch=64, buffer=50k")
        print("  4: lr=0.00025, γ=0.95, batch=64, buffer=100k")
        print()
        print("PPO Experiments:")
        print("  1: lr=0.0001, γ=1.0, n_steps=4096, ent_coef=0.0")
        print("  2: lr=0.0003, γ=1.0, n_steps=2048, ent_coef=0.01")
        print("  3: lr=0.0002, γ=1.0, n_steps=4096, ent_coef=0.02")
        print()
        print("REINFORCE Experiments:")
        print("  1: lr=0.001, γ=0.99, update_threshold=100")
        print("  2: lr=0.0005, γ=1.0, update_threshold=1000")
        print("  3: lr=0.005, γ=1.0, update_threshold=1000")
        print()
        print("Actor-Critic Experiments:")
        print("  1: lr=0.0001, γ=1.0, n_steps=4096")
        print("  2: lr=0.0003, γ=1.0, n_steps=2048")
        print("  3: lr=0.0002, γ=1.0, n_steps=4096")
        return
    
    # Validate required arguments when not listing experiments
    if not args.algorithm:
        parser.error("--algorithm/-a is required when not using --list-experiments")
    if not args.experiment:
        parser.error("--experiment/-e is required when not using --list-experiments")
    
    # Ensure required directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("model/dqn", exist_ok=True)
    os.makedirs("model/pg", exist_ok=True)
    
    # Run the selected algorithm
    success = False
    
    if args.algorithm == 'dqn':
        success = run_dqn_experiment(args.experiment)
    elif args.algorithm == 'ppo':
        success = run_ppo_experiment(args.experiment)
    elif args.algorithm == 'reinforce':
        success = run_reinforce_experiment(args.experiment)
    elif args.algorithm == 'actor_critic':
        success = run_actor_critic_experiment(args.experiment)
    
    if success:
        print(f"\n{args.algorithm.upper()} Experiment {args.experiment} completed successfully!")
        print(f"Check logs/training_log_{args.algorithm}_{args.experiment}.csv for training metrics")
        print(f"Model saved in model/ directory")
    else:
        print(f"\n{args.algorithm.upper()} Experiment {args.experiment} failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()