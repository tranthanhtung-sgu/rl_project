# COMP6008 Reinforcement Learning Project

This project implements and compares three reinforcement learning agents on three challenging environments.

## Project Overview

This project conducts a comparative study of three reinforcement learning algorithms:

1. **REINFORCE (Policy Gradient)**: A Monte Carlo policy gradient method that learns directly from episode returns.
2. **A2C (Advantage Actor-Critic)**: An actor-critic method that uses a value function to reduce variance in policy updates.
3. **DDPG (Deep Deterministic Policy Gradient)**: An off-policy algorithm that combines DQN and policy gradient methods for continuous control.

These agents are evaluated on three environments:

1. **Ant-v4**: A challenging robotics environment with continuous action space.
2. **BreakoutNoFrameskip-v4**: An Atari game with discrete action space.
3. **SeaquestNoFrameskip-v4**: Another Atari game with discrete action space.

## Project Structure

```
rl_project/
├── agents/                # Agent implementations
│   ├── __init__.py
│   ├── reinforce.py       # REINFORCE agent
│   ├── a2c.py             # Advantage Actor-Critic agent
│   └── ddpg.py            # Deep Deterministic Policy Gradient agent
├── environments/          # Environment setup and wrappers
│   ├── __init__.py
│   └── env_wrappers.py    # Preprocessing wrappers for environments
├── utils/                 # Utility functions and classes
│   ├── __init__.py
│   ├── replay_buffer.py   # Replay buffer for DDPG
│   ├── utils.py           # General utility functions
│   └── visualization.py   # Visualization and plotting functions
├── main.py                # Main training script
├── visualize.py           # Visualization script
├── test_agents.py         # Script to test agent implementations
├── run_all_experiments.sh # Script to run all experiments
├── environment.yml        # Conda environment specification
└── README.md              # This file
```

## Setup Instructions

### 1. Create and Activate Conda Environment

```bash
# Create the conda environment
./setup_conda.sh

# Activate the environment
conda activate rl_project
```

### 2. Test Agent Implementations

```bash
# Run a quick test of all agents on all environments
python test_agents.py
```

### 3. Run Individual Experiments

```bash
# Train REINFORCE on BreakoutNoFrameskip-v4
python main.py --agent reinforce --env breakout --episodes 500

# Train A2C on SeaquestNoFrameskip-v4
python main.py --agent a2c --env seaquest --episodes 500

# Train DDPG on Ant-v4
python main.py --agent ddpg --env ant --episodes 1000
```

### 4. Run All Experiments

```bash
# Run all agent-environment combinations
./run_all_experiments.sh
```

### 5. Visualize Results

```bash
# Generate plots from training logs
python visualize.py --log_dir logs --save_dir plots --show
```

## Command Line Arguments

The `main.py` script supports the following arguments:

```
--agent {reinforce,a2c,ddpg}   Agent to use for training
--env {ant,breakout,seaquest}  Environment to train on
--episodes EPISODES            Number of episodes to train for
--seed SEED                    Random seed for reproducibility
--lr LR                        Learning rate
--gamma GAMMA                  Discount factor
--buffer_size BUFFER_SIZE      Size of replay buffer (for DDPG)
--batch_size BATCH_SIZE        Batch size for updates (for DDPG)
--tau TAU                      Soft update coefficient (for DDPG)
--entropy_coef ENTROPY_COEF    Entropy coefficient (for A2C)
--value_loss_coef VALUE_LOSS_COEF  Value loss coefficient (for A2C)
--log_dir LOG_DIR              Directory to save logs
--save_freq SAVE_FREQ          Frequency to save model checkpoints
--eval_freq EVAL_FREQ          Frequency to run evaluation
--eval_episodes EVAL_EPISODES  Number of episodes for evaluation
--device DEVICE                Device to run on (cuda or cpu)
```

## Implementation Details

### REINFORCE Agent

- Implements the basic policy gradient algorithm
- Uses Monte Carlo returns for policy updates
- Supports both discrete and continuous action spaces
- Includes baseline normalization for stability

### A2C Agent

- Implements the Advantage Actor-Critic algorithm
- Uses a shared feature extractor for both actor and critic
- Computes advantages using TD errors
- Includes entropy regularization for exploration

### DDPG Agent

- Implements the Deep Deterministic Policy Gradient algorithm
- Uses separate actor and critic networks with target networks
- Includes exploration noise for action selection
- Implements soft target updates and replay buffer

### Environment Wrappers

- Standard preprocessing for Atari environments:
  - Frame stacking (4 frames)
  - Grayscale conversion
  - Resizing to 84x84
  - Frame skipping
- No preprocessing for Ant environment

## Results and Analysis

After training, you can analyze the results using the visualization script. The script generates:

1. Learning curves for each agent-environment combination
2. Comparison plots between different agents on the same environment
3. Comparison plots for the same agent on different environments

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning.
- Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. ICML.
- Lillicrap, T. P., et al. (2015). Continuous control with deep reinforcement learning. ICLR.