import os
import argparse
import time
import torch
import numpy as np
from tqdm import tqdm

from agents import ReinforceAgent, A2CAgent, DDPGAgent
from environments import make_env, get_env_dimensions
from utils import set_seeds, create_log_dir, MetricsLogger

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train RL agents on various environments')
    
    # Agent and environment selection
    parser.add_argument('--agent', type=str, required=True, choices=['reinforce', 'a2c', 'ddpg'],
                        help='Agent to use for training')
    parser.add_argument('--env', type=str, required=True, choices=['ant', 'breakout', 'seaquest'],
                        help='Environment to train on')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train for')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    
    # DDPG-specific parameters
    parser.add_argument('--buffer_size', type=int, default=1000000,
                        help='Size of replay buffer (for DDPG)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for updates (for DDPG)')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient (for DDPG)')
    
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
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Frequency (in episodes) to run evaluation')
    parser.add_argument('--eval_episodes', type=int, default=5,
                        help='Number of episodes to run for evaluation')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on (cuda or cpu)')
    
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
    elif agent_name == 'ddpg':
        return DDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_lr=args.lr,
            critic_lr=args.lr * 10,  # Critic typically uses higher learning rate
            gamma=args.gamma,
            tau=args.tau,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            device=args.device
        )
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

def evaluate_agent(agent, env_id, num_episodes=5):
    """
    Evaluate an agent on an environment.
    
    Args:
        agent: The agent to evaluate
        env_id: The environment ID
        num_episodes: Number of episodes to evaluate for
        
    Returns:
        Tuple of (mean_reward, mean_length)
    """
    # Create evaluation environment
    eval_env = make_env(env_id, seed=np.random.randint(10000))
    
    # Run evaluation episodes
    rewards = []
    lengths = []
    
    for _ in range(num_episodes):
        state, _ = eval_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            # Select action
            action = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, truncated, _ = eval_env.step(action)
            
            # Update state
            state = next_state
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
        
        # Store episode metrics
        rewards.append(episode_reward)
        lengths.append(episode_length)
    
    # Close environment
    eval_env.close()
    
    # Calculate mean metrics
    mean_reward = np.mean(rewards)
    mean_length = np.mean(lengths)
    
    return mean_reward, mean_length

def train_reinforce(agent, env, args, logger):
    """
    Train a REINFORCE agent.
    
    Args:
        agent: The REINFORCE agent
        env: The environment
        args: Command line arguments
        logger: Metrics logger
    """
    # Training loop
    for episode in tqdm(range(1, args.episodes + 1)):
        # Reset environment
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        # Start episode timer
        logger.start_episode()
        
        # Episode loop
        while not (done or truncated):
            # Select action
            action = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store reward
            agent.store_reward(reward)
            
            # Update state
            state = next_state
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
        
        # Update policy
        loss = agent.update_policy()
        
        # Log episode metrics
        logger.end_episode(episode_reward, episode_length, {'policy_loss': loss})
        
        # Save metrics
        if episode % 10 == 0:
            logger.save()
        
        # Save model checkpoint
        if episode % args.save_freq == 0:
            checkpoint_path = os.path.join(logger.run_dir, f"checkpoint_{episode}.pt")
            agent.save(checkpoint_path)
        
        # Run evaluation
        if episode % args.eval_freq == 0:
            eval_reward, eval_length = evaluate_agent(agent, map_env_name_to_id(args.env), args.eval_episodes)
            print(f"Episode {episode} | Eval Reward: {eval_reward:.2f} | Eval Length: {eval_length:.2f}")

def train_a2c(agent, env, args, logger):
    """
    Train an A2C agent.
    
    Args:
        agent: The A2C agent
        env: The environment
        args: Command line arguments
        logger: Metrics logger
    """
    # Training loop
    for episode in tqdm(range(1, args.episodes + 1)):
        # Reset environment
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        # Start episode timer
        logger.start_episode()
        
        # Episode loop
        while not (done or truncated):
            # Select action
            action = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store reward
            agent.store_reward(reward)
            
            # Update state
            state = next_state
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
        
        # Update policy
        losses = agent.update_policy(next_state=None, done=True)
        
        # Log episode metrics
        logger.end_episode(episode_reward, episode_length, losses)
        
        # Save metrics
        if episode % 10 == 0:
            logger.save()
        
        # Save model checkpoint
        if episode % args.save_freq == 0:
            checkpoint_path = os.path.join(logger.run_dir, f"checkpoint_{episode}.pt")
            agent.save(checkpoint_path)
        
        # Run evaluation
        if episode % args.eval_freq == 0:
            eval_reward, eval_length = evaluate_agent(agent, map_env_name_to_id(args.env), args.eval_episodes)
            print(f"Episode {episode} | Eval Reward: {eval_reward:.2f} | Eval Length: {eval_length:.2f}")

def train_ddpg(agent, env, args, logger):
    """
    Train a DDPG agent.
    
    Args:
        agent: The DDPG agent
        env: The environment
        args: Command line arguments
        logger: Metrics logger
    """
    # Training loop
    total_steps = 0
    
    for episode in tqdm(range(1, args.episodes + 1)):
        # Reset environment
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        # Start episode timer
        logger.start_episode()
        
        # Episode loop
        while not (done or truncated):
            # Select action
            action = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store transition in replay buffer
            agent.store_transition(state, action, reward, next_state, done or truncated)
            
            # Update state
            state = next_state
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # Update policy (more frequently than other algorithms)
            if total_steps >= args.batch_size:
                # Update actor every 2 steps
                update_actor = total_steps % 2 == 0
                losses = agent.update(update_actor=update_actor)
                episode_losses.append(losses)
        
        # Calculate average losses for the episode
        if episode_losses:
            avg_losses = {
                'critic_loss': np.mean([loss['critic_loss'] for loss in episode_losses]),
                'actor_loss': np.mean([loss['actor_loss'] for loss in episode_losses])
            }
        else:
            avg_losses = {'critic_loss': 0, 'actor_loss': 0}
        
        # Log episode metrics
        logger.end_episode(episode_reward, episode_length, avg_losses)
        
        # Save metrics
        if episode % 10 == 0:
            logger.save()
        
        # Save model checkpoint
        if episode % args.save_freq == 0:
            checkpoint_path = os.path.join(logger.run_dir, f"checkpoint_{episode}.pt")
            agent.save(checkpoint_path)
        
        # Run evaluation
        if episode % args.eval_freq == 0:
            eval_reward, eval_length = evaluate_agent(agent, map_env_name_to_id(args.env), args.eval_episodes)
            print(f"Episode {episode} | Eval Reward: {eval_reward:.2f} | Eval Length: {eval_length:.2f}")

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    set_seeds(args.seed)
    
    # Create log directory
    log_dir = create_log_dir(args.log_dir)
    
    # Map environment name to ID
    env_id = map_env_name_to_id(args.env)
    
    # Create environment
    env = make_env(env_id, seed=args.seed)
    
    # Get environment dimensions
    state_dim, action_dim, is_continuous = get_env_dimensions(env_id)
    
    # Create agent
    agent = create_agent(args.agent, state_dim, action_dim, is_continuous, args)
    
    # Create metrics logger
    logger = MetricsLogger(log_dir, args.agent, args.env)
    
    # Print training information
    print(f"Training {args.agent.upper()} on {args.env}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Continuous action space: {is_continuous}")
    print(f"Device: {args.device}")
    print(f"Log directory: {log_dir}")
    
    # Train agent
    try:
        if args.agent == 'reinforce':
            train_reinforce(agent, env, args, logger)
        elif args.agent == 'a2c':
            train_a2c(agent, env, args, logger)
        elif args.agent == 'ddpg':
            train_ddpg(agent, env, args, logger)
        else:
            raise ValueError(f"Unknown agent: {args.agent}")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Save final metrics
        logger.save()
        
        # Save final model
        final_path = os.path.join(logger.run_dir, "final_model.pt")
        agent.save(final_path)
        
        # Close environment
        env.close()
        
        # Print training summary
        metrics = logger.get_latest_metrics(window=100)
        total_time = logger.get_total_training_time()
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Final average reward (last 100 episodes): {metrics['reward']:.2f}")
        print(f"Final average episode length (last 100 episodes): {metrics['length']:.2f}")
        print(f"Models and metrics saved to {logger.run_dir}")

if __name__ == "__main__":
    main()
