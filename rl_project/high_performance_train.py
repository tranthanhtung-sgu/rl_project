#!/usr/bin/env python3
"""
High-performance training script optimized for GPU utilization.
Uses larger batch sizes, parallel environments, and optimized network architectures.
"""

import os
import argparse
import time
import torch
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

from agents import ReinforceAgent, A2CAgent, DDPGAgent, DQNAgent, PPOAgent
from environments import make_env, get_env_dimensions
from utils import set_seeds, create_log_dir, MetricsLogger

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='High-performance RL training with GPU optimization')
    
    # Agent and environment selection
    parser.add_argument('--agent', type=str, required=True, choices=['reinforce', 'a2c', 'dqn'],
                        help='Agent to use for training')
    parser.add_argument('--env', type=str, required=True, choices=['seaquest'],
                        help='Environment to train on')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of episodes to train for')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--num_envs', type=int, default=16,
                        help='Number of parallel environments for faster data collection')
    
    # High-performance parameters
    parser.add_argument('--buffer_size', type=int, default=2000000,
                        help='Size of replay buffer')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for updates')
    parser.add_argument('--update_freq', type=int, default=4,
                        help='Update frequency (every N steps)')
    parser.add_argument('--target_update', type=int, default=1000,
                        help='Target network update frequency')
    
    # A2C-specific parameters
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient (for A2C)')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='Value loss coefficient (for A2C)')
    
    # Logging and saving
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='Frequency (in episodes) to save model checkpoints')
    parser.add_argument('--eval_freq', type=int, default=20,
                        help='Frequency (in episodes) to run evaluation')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of episodes to run for evaluation')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on (cuda or cpu)')
    
    return parser.parse_args()

def map_env_name_to_id(env_name):
    """Map environment name to Gymnasium ID"""
    env_map = {
        'seaquest': 'ALE/Seaquest-v5'
    }
    return env_map.get(env_name)

def create_parallel_envs(env_id, num_envs, seed):
    """Create parallel environments for faster data collection"""
    def make_env_fn(env_id, seed, rank):
        def _make():
            env = make_env(env_id, seed=seed + rank)
            return env
        return _make
    
    env_fns = [make_env_fn(env_id, seed, i) for i in range(num_envs)]
    return AsyncVectorEnv(env_fns)

def create_agent(agent_name, state_dim, action_dim, is_continuous, args):
    """Create an agent with high-performance settings"""
    if agent_name == 'reinforce':
        return ReinforceAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            is_continuous=is_continuous,
            lr=args.lr,
            gamma=args.gamma,
            device=args.device
        )
    elif agent_name == 'a2c':
        return A2CAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            is_continuous=is_continuous,
            actor_lr=args.lr,
            critic_lr=args.lr,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            value_loss_coef=args.value_loss_coef,
            device=args.device
        )
    elif agent_name == 'dqn':
        return DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            is_continuous=is_continuous,
            lr=args.lr,
            gamma=args.gamma,
            replay_size=args.buffer_size,
            batch_size=args.batch_size,
            target_update=args.target_update,
            device=args.device
        )
    elif agent_name == 'ppo':
        return PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            is_continuous=is_continuous,
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=0.90,
            clip_range=0.10,
            ent_coef=0.01,
            vf_coef=0.5,
            n_steps=128,
            batch_size=min(256, args.batch_size),
            n_epochs=4,
            max_grad_norm=0.5,
            device=args.device
        )
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

def evaluate_agent(agent, env_id, num_episodes=10):
    """Evaluate an agent on an environment"""
    eval_env = make_env(env_id, seed=np.random.randint(10000))
    
    rewards = []
    lengths = []
    
    for _ in range(num_episodes):
        state, _ = eval_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            if isinstance(agent, DDPGAgent):
                action = agent.select_action(state, evaluate=True)
            else:
                try:
                    action = agent.select_action(state, stochastic=False, store=False)
                except TypeError:
                    action = agent.select_action(state, stochastic=False)
            
            next_state, reward, done, truncated, _ = eval_env.step(action)
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
    
    eval_env.close()
    return np.mean(rewards), np.mean(lengths)

def train_high_performance(agent, envs, args, logger):
    """High-performance training with parallel environments"""
    print(f"Starting high-performance training with {args.num_envs} parallel environments")
    print(f"Batch size: {args.batch_size}, Buffer size: {args.buffer_size}")
    
    # Training loop
    total_steps = 0
    episode_count = 0
    
    # Reset all environments
    states, _ = envs.reset()
    
    # Progress bar
    pbar = tqdm(total=args.episodes, desc="Training")
    
    while episode_count < args.episodes:
        # Collect data from parallel environments
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        # Collect steps from all environments
        for step in range(args.update_freq):
            # Select actions for all environments
            if isinstance(agent, DQNAgent):
                actions = []
                for i in range(args.num_envs):
                    action = agent.select_action(states[i])
                    actions.append(action)
                actions = np.array(actions)
            elif isinstance(agent, PPOAgent):
                if agent.rollout is None:
                    # Initialize rollout buffer with observed shape (frames,h,w,1)
                    agent.start_rollout(obs_shape=states[0].shape)
                actions = []
                values = []
                logps = []
                for i in range(args.num_envs):
                    a, v, lp = agent.select_action(states[i], stochastic=True, store=True)
                    actions.append(a)
                    values.append(v)
                    logps.append(lp)
                actions = np.array(actions)
            else:
                actions = []
                for i in range(args.num_envs):
                    action = agent.select_action(states[i])
                    actions.append(action)
                actions = np.array(actions)
            
            # Take actions in all environments
            next_states, rewards, dones, truncated, _ = envs.step(actions)
            
            # Store transitions
            for i in range(args.num_envs):
                if isinstance(agent, DQNAgent):
                    agent.store_transition(states[i], actions[i], rewards[i], next_states[i], dones[i] or truncated[i])
                elif isinstance(agent, PPOAgent):
                    agent.store_step(states[i], actions[i], rewards[i], bool(dones[i] or truncated[i]), values[i], logps[i])
                else:
                    agent.store_reward(rewards[i])
            
            # Store batch data
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_rewards.extend(rewards)
            batch_next_states.extend(next_states)
            batch_dones.extend(dones | truncated)
            
            states = next_states
            total_steps += args.num_envs
            
            # Count completed episodes
            episode_count += np.sum(dones | truncated)
        
        # Update agent
        if isinstance(agent, DQNAgent):
            # DQN updates every step
            for _ in range(args.update_freq):
                losses = agent.update()
        elif isinstance(agent, PPOAgent):
            # Bootstrap last value from any env (use first)
            s0 = states[0]
            with torch.no_grad():
                s_t = agent._to_tensor_obs(s0)
                logits, v = agent.net(s_t)
                last_v = float(v.squeeze(0).item())
            losses = agent.update(last_v)
        else:
            # Policy gradient updates
            if len(agent.rewards) > 0:
                losses = agent.update_policy()
            else:
                losses = {'policy_loss': 0}
        
        # Log metrics
        if episode_count > 0:
            avg_reward = np.mean(batch_rewards) if batch_rewards else 0
            logger.end_episode(avg_reward, args.update_freq, losses)
        
        # Update progress bar
        pbar.update(min(episode_count - pbar.n, args.episodes - pbar.n))
        
        # Save metrics
        if episode_count % 50 == 0:
            logger.save()
        
        # Save model checkpoint
        if episode_count % args.save_freq == 0:
            checkpoint_path = os.path.join(logger.run_dir, f"checkpoint_{episode_count}.pt")
            agent.save(checkpoint_path)
        
        # Run evaluation
        if episode_count % args.eval_freq == 0:
            eval_reward, eval_length = evaluate_agent(agent, map_env_name_to_id(args.env), args.eval_episodes)
            print(f"\nEpisode {episode_count} | Eval Reward: {eval_reward:.2f} | Eval Length: {eval_length:.2f}")
            print(f"Total steps: {total_steps} | GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    
    pbar.close()

def main():
    """Main function"""
    args = parse_args()
    
    # Set random seeds
    set_seeds(args.seed)
    
    # Create log directory
    log_dir = create_log_dir(args.log_dir)
    
    # Map environment name to ID
    env_id = map_env_name_to_id(args.env)
    
    # Create parallel environments
    envs = create_parallel_envs(env_id, args.num_envs, args.seed)
    
    # Get environment dimensions
    state_dim, action_dim, is_continuous = get_env_dimensions(env_id)
    
    # Create agent
    agent = create_agent(args.agent, state_dim, action_dim, is_continuous, args)
    
    # Create metrics logger
    logger = MetricsLogger(log_dir, args.agent, args.env)
    
    # Print training information
    print(f"High-Performance Training: {args.agent.upper()} on {args.env}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Continuous action space: {is_continuous}")
    print(f"Device: {args.device}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Log directory: {log_dir}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    
    # Train agent
    try:
        train_high_performance(agent, envs, args, logger)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Save final metrics
        logger.save()
        
        # Save final model
        final_path = os.path.join(logger.run_dir, "final_model.pt")
        agent.save(final_path)
        
        # Close environments
        envs.close()
        
        # Print training summary
        metrics = logger.get_latest_metrics(window=100)
        total_time = logger.get_total_training_time()
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Final average reward (last 100 episodes): {metrics['reward']:.2f}")
        print(f"Final average episode length (last 100 episodes): {metrics['length']:.2f}")
        print(f"Models and metrics saved to {logger.run_dir}")

if __name__ == "__main__":
    main()
