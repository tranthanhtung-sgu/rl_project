# COMP6008 Reinforcement Learning Project

This project implements and compares three reinforcement learning agents on the Atari environment Seaquest, with **high-performance GPU optimization** for maximum training efficiency.

## Project Overview

This project conducts a comparative study of three reinforcement learning algorithms (derived from Practicals 8 and 9):

1. **REINFORCE (Policy Gradient)** (Prac 9)
2. **PPO (Proximal Policy Optimization)** (extension of on-policy actor-critic)
3. **DQN (Deep Q-Network)** (Prac 8)

These agents are evaluated on:

1. **ALE/Seaquest-v5** (focus environment)

## 🚀 **GPU-Optimized Training**

This project includes **high-performance training capabilities** designed to fully utilize modern GPUs:

- **Larger Networks**: Increased CNN channels (64→128→256) and hidden layers (512,512,256)
- **Parallel Environments**: Up to 16 concurrent environments for faster data collection
- **Large Batch Sizes**: 512-1024 batch sizes for maximum GPU utilization
- **Big Replay Buffers**: 1M-2M capacity for better sample efficiency
- **Real-time Monitoring**: GPU memory and utilization tracking

## Project Structure

```
rl_project/
├── agents/                # Agent implementations
│   ├── __init__.py
│   ├── reinforce.py       # REINFORCE agent (Prac 9)
│   ├── a2c.py             # Advantage Actor-Critic agent (Prac 9)
│   ├── dqn.py             # Deep Q-Network agent (Prac 8)
│   └── ppo.py             # Proximal Policy Optimization (Atari-focused)
├── environments/          # Environment setup and wrappers
│   ├── __init__.py
│   └── env_wrappers.py    # Preprocessing wrappers for environments
├── utils/                 # Utility functions and classes
│   ├── __init__.py
│   ├── replay_buffer.py   # Replay buffer for DDPG
│   ├── utils.py           # General utility functions
│   └── visualization.py   # Visualization and plotting functions
├── main.py                # Main training script
├── high_performance_train.py  # 🚀 High-performance GPU-optimized training
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

### 3. Run Seaquest Experiments

#### **🚀 High-Performance Training (Recommended for GPU)**

```bash
# A2C with maximum GPU utilization (16 parallel environments, large batch)
python high_performance_train.py --agent a2c --env seaquest --episodes 2000 --num_envs 16 --batch_size 1024 --lr 0.0003

# DQN with large buffer and parallel environments
python high_performance_train.py --agent dqn --env seaquest --episodes 2000 --num_envs 16 --batch_size 1024 --buffer_size 2000000

# REINFORCE with parallel data collection
python high_performance_train.py --agent reinforce --env seaquest --episodes 2000 --num_envs 16 --lr 0.0002

# PPO with Atari-ready hyperparameters
python high_performance_train.py --agent ppo --env seaquest --episodes 2000 --num_envs 16 --batch_size 256 --lr 0.00025
```

#### **Standard Training (CPU/Small GPU)**

```bash
# REINFORCE (Prac 9)
python main.py --agent reinforce --env seaquest --episodes 800 --eval_freq 20 --lr 0.0002

# A2C with entropy regularization (Prac 9)
python main.py --agent a2c --env seaquest --episodes 800 --eval_freq 20 --lr 0.0003 --entropy_coef 0.01

# DQN with Double DQN + Huber + reward clipping (Prac 8)
python main.py --agent dqn --env seaquest --episodes 1000 --eval_freq 20 --lr 0.0001 --gamma 0.99

# PPO with Atari-ready hyperparameters
python main.py --agent ppo --env seaquest --episodes 800 --eval_freq 20 --lr 0.00025
```

### 4. Live Viewing During Evaluation

```bash
To show a live window only during evaluations, set in `main.py`:

```python
eval_env = make_env(env_id, seed=np.random.randint(10000), render_mode='human')
```

Then run e.g.:

```bash
python main.py --agent a2c --env seaquest --episodes 200 --eval_freq 10
```
```

### 5. Visualize Results

```bash
# Generate plots from training logs
python analyze_training.py

# Show a trained agent live (constants in live_view.py)
python live_view.py

# Save short videos of trained policy (requires moviepy)
python visualize.py --agent a2c --env seaquest --model_path logs/<run>/a2c_seaquest/final_model.pt --episodes 3 --save_video
```

## Command Line Arguments

### **High-Performance Training (`high_performance_train.py`)**

```bash
--agent {reinforce,a2c,dqn,ppo}    Agent to use for training
--env {seaquest}               Environment to train on
--episodes EPISODES            Number of episodes to train for (default: 2000)
--num_envs NUM_ENVS            Number of parallel environments (default: 16)
--batch_size BATCH_SIZE        Batch size for updates (default: 1024)
--buffer_size BUFFER_SIZE      Size of replay buffer (default: 2000000)
--update_freq UPDATE_FREQ      Update frequency (default: 4)
--target_update TARGET_UPDATE  Target network update frequency (default: 1000)
--lr LR                        Learning rate (default: 0.0003)
--gamma GAMMA                  Discount factor (default: 0.99)
--entropy_coef ENTROPY_COEF    Entropy coefficient for A2C (default: 0.01)
--value_loss_coef VALUE_LOSS_COEF  Value loss coefficient for A2C (default: 0.5)
--log_dir LOG_DIR              Directory to save logs
--save_freq SAVE_FREQ          Frequency to save model checkpoints (default: 100)
--eval_freq EVAL_FREQ          Frequency to run evaluation (default: 20)
--eval_episodes EVAL_EPISODES  Number of episodes for evaluation (default: 10)
--device DEVICE                Device to run on (cuda or cpu)
```

### **Standard Training (`main.py`)**

```bash
--agent {reinforce,a2c,dqn,ppo}    Agent to use for training
--env {seaquest}               Environment to train on
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

## Implementation Details & Changes from Practicals

### REINFORCE (Prac 9)
- Monte Carlo policy gradient; normalized returns for variance reduction.
- **GPU Optimized**: Larger networks with (512,512,256) hidden layers.

### A2C (Prac 9)
- Advantage Actor-Critic (shared feature extractor).
- Entropy regularization with cosine decay (new improvement).
- **GPU Optimized**: CNN channels (64→128→256), larger hidden layers (512,512,256).

### DQN (Prac 8)
- CNN + MLP head.
- Improvements (vs. minimal template): Huber loss, reward clipping [-1,1], Double DQN targets.
- **GPU Optimized**: CNN channels (64→128→256), hidden layers (512,512,256), batch size 512, replay buffer 1M.

### PPO (Atari-focused)
- On-policy actor-critic with clipped objective for stability.
- Atari-ready defaults: n_steps=128, batch_size=256, n_epochs=4, lr=2.5e-4, gamma=0.99, gae_lambda=0.90, clip_range=0.10, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5.

### Environment Wrappers

- Standard preprocessing for Atari environments:
  - Frame stacking (4 frames)
  - Grayscale conversion
  - Resizing to 84x84
  - Frame skipping (handled by ALE v5 environments)

## 🚀 **Performance Optimizations**

### **GPU Utilization Features**
- **Parallel Environments**: Up to 16 concurrent environments using `AsyncVectorEnv`
- **Large Batch Processing**: 512-1024 batch sizes for maximum GPU throughput
- **Memory-Efficient**: Optimized tensor operations and gradient accumulation
- **Real-time Monitoring**: GPU memory usage and training progress tracking

### **Network Architecture Improvements**
- **CNN Layers**: Increased from 32→64→64 to **64→128→256** channels
- **MLP Layers**: Expanded from (128,128) to **(512,512,256)** hidden units
- **Replay Buffers**: Increased from 100K to **1M-2M** capacity
- **Target Networks**: More frequent updates for better stability

### **Training Efficiency**
- **3-5x Faster**: Parallel environment data collection
- **Better Sample Efficiency**: Larger networks and batches
- **Improved Stability**: Larger replay buffers and target network updates
- **Scalable**: Automatically adapts to available GPU memory


## Results and Analysis

After training, you can analyze the results using the visualization script. The script generates:

1. Learning curves for each agent-environment combination
2. Comparison plots between different agents on the same environment
3. Comparison plots for the same agent on different environments

## 🎯 **Quick Start Guide**

### **For Maximum GPU Performance:**
```bash
# 1. Test your setup
python test_agents.py

# 2. Run high-performance training (recommended)
python high_performance_train.py --agent a2c --env seaquest --episodes 2000 --num_envs 16

# 3. Visualize results
python visualize.py --agent a2c --env seaquest --model_path logs/<run>/a2c_seaquest/final_model.pt --episodes 3 --save_video
```

### **Expected Performance:**
- **GPU Memory Usage**: 8-15GB (vs 1.4GB standard)
- **Training Speed**: 3-5x faster with parallel environments
- **Sample Efficiency**: Better learning with larger networks and batches
- **Convergence**: More stable training with larger replay buffers

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning.
- Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. ICML.
- Lillicrap, T. P., et al. (2015). Continuous control with deep reinforcement learning. ICLR.