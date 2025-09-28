#!/usr/bin/env python3

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from agents import ReinforceAgent, A2CAgent, DDPGAgent
from environments import make_env, get_env_dimensions
from utils import set_seeds

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize trained RL agents')
    
    # Agent and environment selection
    parser.add_argument('--agent', type=str, required=True, choices=['reinforce', 'a2c', 'ddpg'],
                        help='Agent to visualize')
    parser.add_argument('--env', type=str, required=True, choices=['ant', 'breakout', 'seaquest'],
                        help='Environment to visualize on')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file')
    
    # Visualization parameters
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to visualize')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cuda or cpu)')
    parser.add_argument('--save_video', action='store_true',
                        help='Save video of the episodes')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for video saving')
    
    return parser.parse_args()

def map_env_name_to_id(env_name):
    """Map environment name to Gymnasium ID"""
    env_map = {
        'ant': 'Ant-v4',
        'breakout': 'BreakoutNoFrameskip-v4',
        'seaquest': 'SeaquestNoFrameskip-v4'
    }
    return env_map.get(env_name)

def create_agent(agent_name, state_dim, action_dim, is_continuous, args):
    """Create an agent based on the agent name"""
    if agent_name == 'reinforce':
        return ReinforceAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            is_continuous=is_continuous,
            lr=0.001,
            gamma=0.99,
            device=args.device
        )
    elif agent_name == 'a2c':
        return A2CAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            is_continuous=is_continuous,
            actor_lr=0.001,
            critic_lr=0.001,
            gamma=0.99,
            device=args.device
        )
    elif agent_name == 'ddpg':
        return DDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_lr=0.001,
            critic_lr=0.001,
            gamma=0.99,
            device=args.device
        )
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

def make_visualization_env(env_id, seed=42, save_video=False):
    """Create environment with visualization enabled"""
    from environments import make_env
    
    # Use the same preprocessing as training but with rendering enabled
    render_mode = "rgb_array" if save_video else "human"
    env = make_env(env_id, seed=seed, render_mode=render_mode)
    
    # Add video recording if requested
    if save_video:
        from gymnasium.wrappers import RecordVideo
        env = RecordVideo(env, video_folder="videos", episode_trigger=lambda x: True)
    
    return env

def visualize_episode(agent, env, episode_num, save_video=False):
    """Run one episode and visualize it"""
    print(f"\n=== Episode {episode_num + 1} ===")
    
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    step_count = 0
    
    while not (done or truncated):
        # Select action using the trained policy
        action = agent.select_action(state, stochastic=False)
        
        # Take action in environment
        next_state, reward, done, truncated, info = env.step(action)
        
        # Update state and metrics
        state = next_state
        total_reward += reward
        step_count += 1
        
        # Render the environment (this will show the visualization)
        if not save_video:
            env.render()
        
        # Print progress every 100 steps
        if step_count % 100 == 0:
            print(f"Step {step_count}, Reward: {total_reward:.2f}")
    
    print(f"Episode {episode_num + 1} completed!")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Total steps: {step_count}")
    
    return total_reward, step_count

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    set_seeds(args.seed)
    
    # Map environment name to ID
    env_id = map_env_name_to_id(args.env)
    
    # Create visualization environment
    env = make_visualization_env(env_id, seed=args.seed, save_video=args.save_video)
    
    # Get environment dimensions
    state_dim, action_dim, is_continuous = get_env_dimensions(env_id)
    
    # Create agent
    agent = create_agent(args.agent, state_dim, action_dim, is_continuous, args)
    
    # Load the trained model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    print(f"Loading model from: {args.model_path}")
    agent.load(args.model_path)
    
    # Print visualization information
    print(f"Visualizing {args.agent.upper()} on {args.env}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Continuous action space: {is_continuous}")
    print(f"Device: {args.device}")
    print(f"Episodes to visualize: {args.episodes}")
    
    if args.save_video:
        print(f"Videos will be saved to: videos/")
    
    # Run visualization episodes
    episode_rewards = []
    episode_lengths = []
    
    try:
        for episode in range(args.episodes):
            reward, length = visualize_episode(agent, env, episode, args.save_video)
            episode_rewards.append(reward)
            episode_lengths.append(length)
            
            # Small delay between episodes
            import time
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    
    finally:
        # Close environment
        env.close()
        
        # Print summary
        if episode_rewards:
            print(f"\n=== Visualization Summary ===")
            print(f"Average reward: {np.mean(episode_rewards):.2f}")
            print(f"Average episode length: {np.mean(episode_lengths):.2f}")
            print(f"Best reward: {np.max(episode_rewards):.2f}")
            print(f"Worst reward: {np.min(episode_rewards):.2f}")
        
        if args.save_video:
            print(f"\nVideos saved to: videos/")

if __name__ == "__main__":
    main()