#!/usr/bin/env python3
"""
Demo script to show how to visualize learning progress
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from agents import ReinforceAgent, A2CAgent, DDPGAgent
from environments import make_env, get_env_dimensions
from utils import set_seeds

def demo_ant_learning():
    """Demonstrate how to watch the Ant learn to walk"""
    print("🤖 Demo: Watching Ant Learn to Walk")
    print("=" * 50)
    
    # Set up environment
    env_id = 'Ant-v5'
    env = make_env(env_id, seed=42, render_mode='human')  # 'human' for live visualization
    
    # Get dimensions
    state_dim, action_dim, is_continuous = get_env_dimensions(env_id)
    print(f"Environment: {env_id}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Continuous actions: {is_continuous}")
    
    # Create agent
    agent = ReinforceAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        is_continuous=is_continuous,
        lr=0.001,
        gamma=0.99,
        device='cpu'
    )
    
    print("\n🎯 Training for 20 episodes with live visualization...")
    print("Watch the Ant learn to walk! (Close the window to stop)")
    
    rewards = []
    episode_lengths = []
    
    for episode in range(20):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store reward
            agent.store_reward(reward)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Update policy
        loss = agent.update_policy()
        
        # Store metrics
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode+1:2d} | Reward: {episode_reward:8.2f} | Length: {episode_length:3d} | Loss: {loss:.4f}")
    
    env.close()
    
    # Plot learning curve
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Reward Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Length Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ant_learning_progress.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Learning curve saved as 'ant_learning_progress.png'")
    print(f"🎯 Final reward: {rewards[-1]:.2f} (started at {rewards[0]:.2f})")
    print(f"⏱️  Final episode length: {episode_lengths[-1]} (started at {episode_lengths[0]})")

def demo_breakout_learning():
    """Demonstrate how to watch Breakout learning"""
    print("\n🎮 Demo: Watching Breakout Learning")
    print("=" * 50)
    
    # Set up environment
    env_id = 'ALE/Breakout-v5'
    env = make_env(env_id, seed=42, render_mode='human')
    
    # Get dimensions
    state_dim, action_dim, is_continuous = get_env_dimensions(env_id)
    print(f"Environment: {env_id}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Continuous actions: {is_continuous}")
    
    # Create agent
    agent = ReinforceAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        is_continuous=is_continuous,
        lr=0.001,
        gamma=0.99,
        device='cpu'
    )
    
    print("\n🎯 Training for 10 episodes with live visualization...")
    print("Watch the agent learn to play Breakout! (Close the window to stop)")
    
    rewards = []
    
    for episode in range(10):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.store_reward(reward)
            state = next_state
            episode_reward += reward
        
        loss = agent.update_policy()
        rewards.append(episode_reward)
        
        print(f"Episode {episode+1:2d} | Reward: {episode_reward:6.1f} | Loss: {loss:.4f}")
    
    env.close()
    
    # Plot learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(rewards)
    plt.title('Breakout Learning Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('breakout_learning_progress.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Learning curve saved as 'breakout_learning_progress.png'")
    print(f"🎯 Final reward: {rewards[-1]:.1f} (started at {rewards[0]:.1f})")

if __name__ == "__main__":
    print("🚀 RL Learning Visualization Demo")
    print("=" * 60)
    
    choice = input("Choose environment to demo:\n1. Ant (watch it learn to walk)\n2. Breakout (watch it learn to play)\n3. Both\nEnter choice (1/2/3): ")
    
    if choice == "1":
        demo_ant_learning()
    elif choice == "2":
        demo_breakout_learning()
    elif choice == "3":
        demo_ant_learning()
        demo_breakout_learning()
    else:
        print("Invalid choice. Running Ant demo...")
        demo_ant_learning()
