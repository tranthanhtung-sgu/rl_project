# Gymnasium Setup Guide

This guide follows the official Gymnasium documentation to ensure proper setup and usage.

## ✅ Current Status

Your current setup:
- **Gymnasium**: 0.28.1 ✅
- **ALE-py**: 0.8.1 ✅ (for Atari environments)
- **MuJoCo**: 3.3.6 ✅ (for continuous control environments)

## 🚀 Complete Installation (Following Official Docs)

### 1. Base Installation
```bash
# You already have this, but for reference:
pip install gymnasium
```

### 2. Environment-Specific Dependencies

Based on the official documentation, install dependencies for specific environment families:

```bash
# For Atari environments (Breakout, Seaquest, etc.)
pip install "gymnasium[atari]"

# For MuJoCo environments (Ant, Humanoid, etc.)
pip install "gymnasium[mujoco]"

# For Box2D environments (CarRacing, BipedalWalker, etc.)
pip install "gymnasium[box2d]"

# For Classic Control environments (CartPole, Pendulum, etc.)
pip install "gymnasium[classic-control]"

# For Toy Text environments (FrozenLake, Taxi, etc.)
pip install "gymnasium[toy-text]"

# For PyTorch integration
pip install "gymnasium[torch]"

# For ALL dependencies (if you want everything)
pip install "gymnasium[all]"
```

### 3. Development Dependencies (Optional)

If you want to contribute to Gymnasium or run tests:

```bash
# Install pre-commit for code formatting
pip install pre-commit

# Install the git hooks
pre-commit install

# Run formatting checks
pre-commit run --all-files
```

## 🎯 Proper Usage Patterns

### 1. Basic Environment Creation
```python
import gymnasium as gym

# Create environment with proper render mode
env = gym.make("CartPole-v1", render_mode="human")  # For visualization
# or
env = gym.make("CartPole-v1", render_mode="rgb_array")  # For video recording
# or
env = gym.make("CartPole-v1")  # No rendering (fastest for training)
```

### 2. Proper Agent-Environment Loop
```python
import gymnasium as gym

# Create environment
env = gym.make("CartPole-v1", render_mode="human")

# Reset to start new episode
observation, info = env.reset(seed=42)

episode_reward = 0
done = False

while not done:
    # Agent selects action (replace with your agent's policy)
    action = env.action_space.sample()  # Random for now
    
    # Take action in environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Check if episode is done
    done = terminated or truncated
    episode_reward += reward
    
    # Render if needed
    env.render()

print(f"Episode finished with reward: {episode_reward}")
env.close()
```

### 3. Understanding Spaces
```python
import gymnasium as gym

env = gym.make("CartPole-v1")

# Check action space
print(f"Action space: {env.action_space}")
print(f"Action space type: {type(env.action_space)}")
print(f"Number of actions: {env.action_space.n}")

# Check observation space
print(f"Observation space: {env.observation_space}")
print(f"Observation space shape: {env.observation_space.shape}")
print(f"Observation space bounds: {env.observation_space.low} to {env.observation_space.high}")

# Sample from spaces
action = env.action_space.sample()
observation = env.observation_space.sample()
```

## 🔧 Environment-Specific Setup

### Atari Environments (Breakout, Seaquest)
```python
import gymnasium as gym

# Atari environments require specific setup
env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")

# Atari environments have specific observation spaces
print(f"Observation space: {env.observation_space}")
# Usually Box(0, 255, (210, 160, 3), uint8) for raw frames
```

### MuJoCo Environments (Ant, Humanoid)
```python
import gymnasium as gym

# MuJoCo environments for continuous control
env = gym.make("Ant-v4", render_mode="human")

# MuJoCo environments have continuous action spaces
print(f"Action space: {env.action_space}")
# Usually Box(-1, 1, (8,), float32) for Ant
```

## 🎮 Visualization Best Practices

### 1. Render Modes
```python
# Human rendering (opens window)
env = gym.make("CartPole-v1", render_mode="human")

# RGB array (for video recording)
env = gym.make("CartPole-v1", render_mode="rgb_array")

# No rendering (fastest for training)
env = gym.make("CartPole-v1")
```

### 2. Video Recording
```python
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# Create environment with video recording
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos", episode_trigger=lambda x: True)

# Run episode
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

## 🚨 Common Issues and Solutions

### 1. MuJoCo Compatibility Issues
If you get MuJoCo errors:
```bash
# Update MuJoCo
pip install --upgrade mujoco

# Or reinstall with specific version
pip install mujoco==3.3.6
```

### 2. Atari ROM Issues
If Atari environments don't work:
```bash
# Install ALE-py with ROM support
pip install "ale-py[accept-rom-license]"
```

### 3. Rendering Issues
If rendering doesn't work:
```python
# Try different render modes
env = gym.make("CartPole-v1", render_mode="rgb_array")
# or
env = gym.make("CartPole-v1", render_mode="human")
```

## 📚 Environment Families

### Classic Control
- **CartPole-v1**: Balance a pole on a cart
- **Pendulum-v1**: Swing up a pendulum
- **MountainCar-v0**: Drive a car up a mountain

### Atari
- **BreakoutNoFrameskip-v4**: Classic Breakout game
- **SeaquestNoFrameskip-v4**: Underwater submarine game
- **PongNoFrameskip-v4**: Classic Pong game

### MuJoCo
- **Ant-v4**: 4-legged robot locomotion
- **Humanoid-v4**: Humanoid robot control
- **HalfCheetah-v4**: 2D cheetah running

### Box2D
- **CarRacing-v3**: Top-down car racing
- **BipedalWalker-v3**: 2D bipedal walking
- **LunarLander-v2**: Lunar lander control

## 🎯 Next Steps

1. **Test your setup**:
   ```bash
   python -c "import gymnasium as gym; env = gym.make('CartPole-v1'); print('Setup successful!')"
   ```

2. **Run a simple test**:
   ```bash
   python -c "
   import gymnasium as gym
   env = gym.make('CartPole-v1', render_mode='human')
   obs, info = env.reset()
   for _ in range(100):
       action = env.action_space.sample()
       obs, reward, terminated, truncated, info = env.step(action)
       if terminated or truncated:
           break
   env.close()
   print('Test completed!')
   "
   ```

3. **Use with your RL agents**: Your existing agents should work perfectly with this setup!

## 📖 Official Resources

- **Documentation**: https://gymnasium.farama.org/
- **GitHub**: https://github.com/Farama-Foundation/Gymnasium
- **Discord**: https://discord.gg/bnJ6kubTg6
- **API Reference**: https://gymnasium.farama.org/api/

## ✅ Verification Checklist

- [ ] Gymnasium installed (`pip list | grep gymnasium`)
- [ ] Atari support (`pip list | grep ale`)
- [ ] MuJoCo support (`pip list | grep mujoco`)
- [ ] Basic environment creation works
- [ ] Rendering works (if needed)
- [ ] Your RL agents work with environments

Your setup is already excellent! The main thing is to follow the proper usage patterns shown above.
