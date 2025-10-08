#!/usr/bin/env python3
"""
Script to analyze training logs and visualize learning progress
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_training_logs(log_dir="logs"):
    """Analyze training logs and create visualizations"""
    print("📊 Analyzing Training Logs")
    print("=" * 50)
    
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"❌ Log directory '{log_dir}' not found!")
        print("Train some models first using:")
        print("python main.py --agent reinforce --env ant --episodes 100")
        return
    
    # Find all training runs
    runs = []
    for agent_dir in log_path.iterdir():
        if agent_dir.is_dir():
            for env_dir in agent_dir.iterdir():
                if env_dir.is_dir():
                    metrics_file = env_dir / "metrics.json"
                    if metrics_file.exists():
                        runs.append({
                            'agent': agent_dir.name,
                            'env': env_dir.name,
                            'path': env_dir,
                            'metrics_file': metrics_file
                        })
    
    if not runs:
        print("❌ No training logs found!")
        return
    
    print(f"Found {len(runs)} training runs:")
    for i, run in enumerate(runs):
        print(f"  {i+1}. {run['agent']} on {run['env']}")
    
    # Analyze each run
    for run in runs:
        print(f"\n🔍 Analyzing {run['agent']} on {run['env']}")
        
        try:
            # Load metrics
            with open(run['metrics_file'], 'r') as f:
                metrics = json.load(f)
            
            # Extract data
            episodes = metrics.get('episodes', [])
            rewards = metrics.get('rewards', [])
            lengths = metrics.get('lengths', [])
            losses = metrics.get('losses', {})
            
            if not episodes:
                print("  ❌ No episode data found")
                continue
            
            # Create visualizations
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{run["agent"].upper()} on {run["env"]} - Learning Progress', fontsize=16)
            
            # Reward over time
            axes[0, 0].plot(episodes, rewards, alpha=0.7)
            axes[0, 0].set_title('Reward Over Time')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Moving average of rewards
            if len(rewards) > 10:
                window = min(50, len(rewards) // 5)
                moving_avg = pd.Series(rewards).rolling(window=window).mean()
                axes[0, 1].plot(episodes, rewards, alpha=0.3, label='Raw')
                axes[0, 1].plot(episodes, moving_avg, label=f'Moving Avg ({window})', linewidth=2)
                axes[0, 1].set_title('Reward with Moving Average')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Total Reward')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Episode length over time
            axes[1, 0].plot(episodes, lengths, alpha=0.7, color='orange')
            axes[1, 0].set_title('Episode Length Over Time')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Episode Length')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Loss over time (if available)
            if losses:
                loss_keys = list(losses.keys())
                for loss_key in loss_keys:
                    if loss_key in losses and losses[loss_key]:
                        axes[1, 1].plot(episodes, losses[loss_key], label=loss_key, alpha=0.7)
                axes[1, 1].set_title('Loss Over Time')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No loss data available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Loss Over Time')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = run['path'] / 'learning_analysis.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  📊 Analysis saved to: {plot_path}")
            
            # Print summary statistics
            print(f"  📈 Summary:")
            print(f"    - Total episodes: {len(episodes)}")
            print(f"    - Initial reward: {rewards[0]:.2f}")
            print(f"    - Final reward: {rewards[-1]:.2f}")
            print(f"    - Best reward: {max(rewards):.2f}")
            print(f"    - Average reward: {np.mean(rewards):.2f}")
            print(f"    - Improvement: {rewards[-1] - rewards[0]:.2f}")
            
            plt.show()
            
        except Exception as e:
            print(f"  ❌ Error analyzing {run['agent']} on {run['env']}: {e}")

def compare_agents():
    """Compare different agents on the same environment"""
    print("\n🏆 Agent Comparison")
    print("=" * 50)
    
    log_path = Path("logs")
    if not log_path.exists():
        print("❌ No logs found for comparison")
        return
    
    # Group runs by environment
    env_runs = {}
    for agent_dir in log_path.iterdir():
        if agent_dir.is_dir():
            for env_dir in agent_dir.iterdir():
                if env_dir.is_dir():
                    metrics_file = env_dir / "metrics.json"
                    if metrics_file.exists():
                        env_name = env_dir.name
                        if env_name not in env_runs:
                            env_runs[env_name] = []
                        env_runs[env_name].append({
                            'agent': agent_dir.name,
                            'metrics_file': metrics_file
                        })
    
    # Create comparison plots for each environment
    for env_name, runs in env_runs.items():
        if len(runs) < 2:
            continue
            
        print(f"\n📊 Comparing agents on {env_name}")
        
        plt.figure(figsize=(12, 8))
        
        for i, run in enumerate(runs):
            try:
                with open(run['metrics_file'], 'r') as f:
                    metrics = json.load(f)
                
                episodes = metrics.get('episodes', [])
                rewards = metrics.get('rewards', [])
                
                if episodes and rewards:
                    # Plot moving average
                    window = min(50, len(rewards) // 5)
                    moving_avg = pd.Series(rewards).rolling(window=window).mean()
                    plt.plot(episodes, moving_avg, label=f"{run['agent'].upper()}", linewidth=2)
                    
            except Exception as e:
                print(f"  ❌ Error loading {run['agent']}: {e}")
        
        plt.title(f'Agent Comparison on {env_name}', fontsize=16)
        plt.xlabel('Episode')
        plt.ylabel('Reward (Moving Average)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save comparison plot
        comparison_path = log_path / f'comparison_{env_name}.png'
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Comparison saved to: {comparison_path}")
        
        plt.show()

if __name__ == "__main__":
    print("🔍 Training Analysis Tool")
    print("=" * 60)
    
    choice = input("Choose analysis type:\n1. Analyze individual training runs\n2. Compare agents\n3. Both\nEnter choice (1/2/3): ")
    
    if choice == "1":
        analyze_training_logs()
    elif choice == "2":
        compare_agents()
    elif choice == "3":
        analyze_training_logs()
        compare_agents()
    else:
        print("Invalid choice. Running individual analysis...")
        analyze_training_logs()
