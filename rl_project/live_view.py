# live_view.py
import time
import torch
import numpy as np

from agents import ReinforceAgent, A2CAgent, DDPGAgent
from environments import make_env, get_env_dimensions

# ====== EDIT THESE THREE CONSTANTS ======
AGENT = 'a2c'  # 'reinforce', 'a2c', or 'ddpg'
ENV = 'ant'    # 'ant', 'breakout', or 'seaquest'
MODEL_PATH = 'logs/20250930_183942/a2c_ant/final_model.pt'  # path to your saved model
EPISODES = 2
# ========================================

ENV_MAP = {
    'ant': 'Ant-v5',
    'breakout': 'ALE/Breakout-v5',
    'seaquest': 'ALE/Seaquest-v5',
}

def create_agent(agent_name, env_id):
    state_dim, action_dim, is_continuous = get_env_dimensions(env_id)
    if agent_name == 'reinforce':
        agent = ReinforceAgent(state_dim, action_dim, is_continuous, lr=0.001, gamma=0.99, device='cpu')
    elif agent_name == 'a2c':
        agent = A2CAgent(state_dim, action_dim, is_continuous, actor_lr=0.001, critic_lr=0.001, gamma=0.99, device='cpu')
    elif agent_name == 'ddpg':
        agent = DDPGAgent(state_dim, action_dim, actor_lr=0.001, critic_lr=0.001, gamma=0.99, device='cpu')
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
    return agent

def main():
    env_id = ENV_MAP[ENV]
    print(f"Creating environment for live viewing: {env_id}")
    env = make_env(env_id, seed=42, render_mode='human')  # live window
    agent = create_agent(AGENT, env_id)

    print(f"Loading model: {MODEL_PATH}")
    agent.load(MODEL_PATH)

    print(f"Starting live viewing for {EPISODES} episode(s). Close the window to stop early.")
    for ep in range(EPISODES):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        print(f"\n=== Episode {ep + 1} ===")

        while not (done or truncated):
            # Deterministic action for viewing; avoid storing into training buffers if supported
            try:
                action = agent.select_action(state, stochastic=False, store=False)
            except TypeError:
                action = agent.select_action(state, stochastic=False)

            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            # Render happens via human mode window; small sleep for watchability
            time.sleep(0.01)

        print(f"Episode {ep + 1} finished. Reward: {total_reward:.2f}, Steps: {steps}")

    env.close()
    print("Live viewing completed.")

if __name__ == "__main__":
    main()