#!/usr/bin/env python3
"""
Quick visualization script for trained RL agents.
This script allows you to quickly visualize a trained model without complex setup.
"""

import os
import torch
import numpy as np
from agents import ReinforceAgent, A2CAgent, DDPGAgent
from environments import get_env_dimensions
import gymnasium as gym

def quick_visualize(agent_name, env_name, model_path, episodes=3):
    """
    Quick visualization of a trained agent
    
    Args:
        agent_name (str): 'reinforce', 'a2c', or 'ddpg'
        env_name (str): 'ant', 'breakout', or 'seaquest'
        model_path (str): Path to the trained model
        episodes (int): Number of episodes to visualize
    """
    
    # Map environment names to IDs
    env_map = {
        'ant': 'Ant-v4',
        'breakout': 'BreakoutNoFrameskip-v4',
        'seaquest': 'SeaquestNoFrameskip-v4'
    }
    
    env_id = env_map[env_name]
    
    # Create environment with human rendering and same preprocessing as training
    print(f"Creating environment: {env_id}")
    from environments import make_env
    env = make_env(env_id, seed=42, render_mode='human')
    
    # Get environment dimensions
    state_dim, action_dim, is_continuous = get_env_dimensions(env_id)
    
    # Create agent
    print(f"Creating {agent_name.upper()} agent...")
    if agent_name == 'reinforce':
        agent = ReinforceAgent(state_dim, action_dim, is_continuous, lr=0.001, gamma=0.99, device='cpu')
    elif agent_name == 'a2c':
        agent = A2CAgent(state_dim, action_dim, is_continuous, actor_lr=0.001, critic_lr=0.001, gamma=0.99, device='cpu')
    elif agent_name == 'ddpg':
        agent = DDPGAgent(state_dim, action_dim, actor_lr=0.001, critic_lr=0.001, gamma=0.99, device='cpu')
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    agent.load(model_path)
    
    # Run episodes
    print(f"Starting visualization of {episodes} episodes...")
    print("Close the visualization window to stop early.")
    
    for episode in range(episodes):
        print(f"\n=== Episode {episode + 1} ===")
        
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        while not (done or truncated):
            # Select action (deterministic for visualization)
            action = agent.select_action(state, stochastic=False)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Render (this shows the visualization)
            env.render()
            
            # Add a small delay to make it more watchable
            import time
            time.sleep(0.01)  # 10ms delay
        
        print(f"Episode {episode + 1} completed!")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Total steps: {step_count}")
    
    env.close()
    print("Visualization completed!")

if __name__ == "__main__":
    # Example usage - modify these paths to your trained models
    print("Quick RL Agent Visualization")
    print("=" * 40)
    
    # You can modify these parameters
    AGENT = 'reinforce'  # 'reinforce', 'a2c', or 'ddpg'
    ENV = 'breakout'          # 'ant', 'breakout', or 'seaquest'
    MODEL_PATH = 'logs/20250928_190414/reinforce_breakout/final_model.pt'  # Path to your trained model
    EPISODES = 3
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Please train a model first or update the MODEL_PATH variable.")
        print("\nTo train a model, run:")
        print(f"python main.py --agent {AGENT} --env {ENV} --episodes 100")
        exit(1)
    
    try:
        quick_visualize(AGENT, ENV, MODEL_PATH, EPISODES)
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user.")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
