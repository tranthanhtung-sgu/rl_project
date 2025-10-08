import torch
import numpy as np
from environments import make_env, get_env_dimensions
from agents import ReinforceAgent, A2CAgent, DDPGAgent

def test_agent(agent_class, env_id, episodes=2):
    """Test an agent on an environment for a few episodes"""
    print(f"Testing {agent_class.__name__} on {env_id}")
    
    # Create environment
    env = make_env(env_id, seed=42)
    
    # Get environment dimensions
    state_dim, action_dim, is_continuous = get_env_dimensions(env_id)
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Continuous action space: {is_continuous}")
    
    # Create agent
    if agent_class == DDPGAgent and not is_continuous:
        print("Skipping DDPG on discrete action space")
        env.close()
        return
    
    if agent_class == DDPGAgent:
        agent = agent_class(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=float(env.action_space.high[0]) if is_continuous else 1.0,
            buffer_size=10000,
            batch_size=64
        )
    else:
        agent = agent_class(
            state_dim=state_dim,
            action_dim=action_dim,
            is_continuous=is_continuous
        )
    
    # Run a few episodes
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            # Select action
            action = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store transition
            if agent_class == DDPGAgent:
                agent.store_transition(state, action, reward, next_state, done)
            else:
                agent.store_reward(reward)
            
            # Update state
            state = next_state
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            
            # Update policy (for DDPG)
            if agent_class == DDPGAgent and len(agent.replay_buffer) >= 64:
                agent.update()
        
        # Skip policy updates for testing
        # This would normally update the policy, but we'll skip it for the test
        # as we're just testing if the agent can run without errors
        
        print(f"Episode {episode} | Reward: {episode_reward:.2f} | Length: {episode_length}")
    
    # Close environment
    env.close()
    print("Test completed successfully")
    print("--------------------------")

def main():
    """Main function"""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test agents on different environments
    test_agent(ReinforceAgent, 'ALE/Breakout-v5')
    test_agent(A2CAgent, 'ALE/Breakout-v5')
    test_agent(DDPGAgent, 'ALE/Breakout-v5')
    
    test_agent(ReinforceAgent, 'ALE/Seaquest-v5')
    test_agent(A2CAgent, 'ALE/Seaquest-v5')
    test_agent(DDPGAgent, 'ALE/Seaquest-v5')
    
    test_agent(ReinforceAgent, 'Ant-v5')
    test_agent(A2CAgent, 'Ant-v5')
    test_agent(DDPGAgent, 'Ant-v5')

if __name__ == "__main__":
    main()
