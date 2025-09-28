#!/usr/bin/env python3
"""
Fast visualization script for trained RL agents.
This version renders every few frames to make it faster and more watchable.
"""

import os
import torch
import numpy as np
from agents import ReinforceAgent, A2CAgent, DDPGAgent
from environments import get_env_dimensions, make_env

def fast_visualize(agent_name, env_name, model_path, episodes=3, render_every=5):
    """
    Fast visualization of a trained agent
    
    Args:
        agent_name (str): 'reinforce', 'a2c', or 'ddpg'
        env_name (str): 'ant', 'breakout', or 'seaquest'
        model_path (str): Path to the trained model
        episodes (int): Number of episodes to visualize
        render_every (int): Render every N frames (higher = faster)
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
    print(f"Starting fast visualization of {episodes} episodes...")
    print(f"Rendering every {render_every} frames for better performance.")
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
            
            # Render only every N frames for better performance
            if step_count % render_every == 0:
                env.render()
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"Step {step_count}, Reward: {total_reward:.2f}")
        
        print(f"Episode {episode + 1} completed!")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Total steps: {step_count}")
    
    env.close()
    print("Fast visualization completed!")

if __name__ == "__main__":
    # Example usage - modify these paths to your trained models
    print("Fast RL Agent Visualization")
    print("=" * 40)
    
    # You can modify these parameters
    AGENT = 'reinforce'  # 'reinforce', 'a2c', or 'ddpg'
    ENV = 'breakout'     # 'ant', 'breakout', or 'seaquest'
    MODEL_PATH = 'logs/20250928_190414/reinforce_breakout/final_model.pt'  # Path to your trained model
    EPISODES = 3
    RENDER_EVERY = 10  # Render every 10 frames (higher = faster)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Please train a model first or update the MODEL_PATH variable.")
        print("\nTo train a model, run:")
        print(f"python main.py --agent {AGENT} --env {ENV} --episodes 100")
        exit(1)
    
    try:
        fast_visualize(AGENT, ENV, MODEL_PATH, EPISODES, RENDER_EVERY)
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user.")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
