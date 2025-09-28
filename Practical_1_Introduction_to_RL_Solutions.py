#!/usr/bin/env python3

# COMP6008 Reinforcement Learning
# Computing@Curtin University
#
# Practical 1 - Introduction to RL
#
# Learning outcomes: after completing this practical you will be able to
# * Set up software environment for RL
# * Create basic RL environments using OpenAI Gym
# * Implement RL loop and visualise the environments
# * Implement simple agents

# import necessary libraries
import numpy as np
import gymnasium as gym
import moviepy.editor as mpy

# (To use moviepy, "ffmpeg" needs to be installed, for Mac Silicon following link 
# https://github.com/Zulko/moviepy/issues/1619 can be used to install "ffmpeg")

# Create environment and reset its state to receive the first observation
env = gym.make('LunarLander-v2', render_mode="rgb_array_list")
obs, _ = env.reset()

# print the properties of observation and action spaces
print(f'Observation space: {env.observation_space}')
print(f'Action space: {env.action_space}')

# close the environment
env.close()

# Exercise 1: Create Taxi-v3 environment and print its observation and action spaces
env = gym.make('Taxi-v3', render_mode="rgb_array_list")
obs, _ = env.reset()

# print the properties of observation and action spaces
print(f'Observation space: {env.observation_space}')
print(f'Action space: {env.action_space}')

# close the environment
env.close()

# Implementing RL loop
env = gym.make('LunarLander-v2', render_mode="rgb_array_list")
obs, _ = env.reset()

# terminated is True when the environment terminates
terminated = False

# truncated is True when the environment is stopped but not terminated (e.g., step limit)
truncated = False

while not (terminated or truncated):
    # sample random action from action space
    action = env.action_space.sample()

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    obs, reward, terminated, truncated, info = env.step(action)

    # print the immediate reward received
    print(f'Reward: {reward}')

# close the environment
env.close()

# Exercise 3: Calculate the return for the initial state G_0 = R_1 + γR_2 + γ^2R_3 + ...
env = gym.make('LunarLander-v2', render_mode="rgb_array_list")
obs, _ = env.reset()

# discount factor
GAMMA = 0.9

# running gamma and return
gamma = 1
G0 = 0

# terminated is True when the environment terminates
terminated = False

# truncated is True when the environment is stopped but not terminated (e.g., step limit)
truncated = False

while not (terminated or truncated):
    # sample random action from action space
    action = env.action_space.sample()

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    obs, reward, terminated, truncated, info = env.step(action)

    G0 += gamma*reward
    gamma *= GAMMA

# print the total return for initial state
print(f'Return: {G0}')

# close the environment
env.close()

# Visualising environments
env = gym.make('LunarLander-v2', render_mode="rgb_array_list")
obs, _ = env.reset()

# terminated is True when the environment terminates
terminated = False

# truncated is True when the environment is stopped but not terminated (e.g., step limit)
truncated = False

while not (terminated or truncated):
    # sample random action from action space
    action = env.action_space.sample()

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    obs, reward, terminated, truncated, info = env.step(action)

# store RGB frames for the entire episode
frames = env.render()

# close the environment
env.close()

# create and play video clip using the frames and given fps
clip = mpy.ImageSequenceClip(frames, fps=50)
# Note: In script mode, you'd need to save the clip instead of displaying it inline
# clip.write_videofile("lunarlander_random.mp4")

# Exercise 4: Visualise the environments
# Taxi-v3
env = gym.make('Taxi-v3', render_mode="rgb_array_list")
obs, _ = env.reset()

# terminated is True when the environment terminates
terminated = False

# truncated is True when the environment is stopped but not terminated (e.g., step limit)
truncated = False

while not (terminated or truncated):
    # sample random action from action space
    action = env.action_space.sample()

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    obs, reward, terminated, truncated, info = env.step(action)

# store RGB frames for the entire episode
frames = env.render()

# close the environment
env.close()

# create and play video clip using the frames and given fps
clip = mpy.ImageSequenceClip(frames, fps=4)
# clip.write_videofile("taxi_random.mp4")

# Humanoid-v4
env = gym.make('Humanoid-v4', render_mode="rgb_array_list")
obs, _ = env.reset()

# terminated is True when the environment terminates
terminated = False

# truncated is True when the environment is stopped but not terminated (e.g., step limit)
truncated = False

while not (terminated or truncated):
    # sample random action from action space
    action = env.action_space.sample()

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    obs, reward, terminated, truncated, info = env.step(action)

# store RGB frames for the entire episode
frames = env.render()

# close the environment
env.close()

# create and play video clip using the frames and given fps
clip = mpy.ImageSequenceClip(frames, fps=67)
# clip.write_videofile("humanoid_random.mp4")

# MountainCar-v0
env = gym.make('MountainCar-v0', render_mode="rgb_array_list")
obs, _ = env.reset()

# terminated is True when the environment terminates
terminated = False

# truncated is True when the environment is stopped but not terminated (e.g., step limit)
truncated = False

while not (terminated or truncated):
    # sample random action from action space
    action = env.action_space.sample()

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    obs, reward, terminated, truncated, info = env.step(action)

# store RGB frames for the entire episode
frames = env.render()

# close the environment
env.close()

# create and play video clip using the frames and given fps
clip = mpy.ImageSequenceClip(frames, fps=30)
# clip.write_videofile("mountaincar_random.mp4")

# Note on visualising Atari environments
# ALE/Breakout-v5
env = gym.make('ALE/Breakout-v5', render_mode="rgb_array")
obs, _ = env.reset()

# list for storing RGB frames
frames = []

# terminated is True when the environment terminates
terminated = False

# truncated is True when the environment is stopped but not terminated (e.g., step limit)
truncated = False

while not (terminated or truncated):
    # sample random action from action space
    action = env.action_space.sample()

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    obs, reward, terminated, truncated, info = env.step(action)

    # RGB frames manually
    frames.append(env.render())

# close the environment
env.close()

# create and play video clip using the frames and given fps (.resize(2) doubles the video size)
clip = mpy.ImageSequenceClip(frames, fps=25)
# clip.write_videofile("breakout_random.mp4")

# Implementing simple agents
class Agent():
    def __init__(self, env):
        self.env = env

    def policy(self, state):
        # push cart to the left/right if the pole angle is negative/positive
        theta = state[2]
        if theta < 0:
            return 0
        if theta > 0:
            return 1
        return self.env.action_space.sample()

# Create environment and reset its state to receive the first observation
env = gym.make('CartPole-v1', render_mode="rgb_array_list")
obs, _ = env.reset()

# create agent
agent = Agent(env)

# terminated is True when the environment terminates
terminated = False

# truncated is True when the environment is stopped but not terminated (e.g., step limit)
truncated = False

while not (terminated or truncated):
    # take action based on agent policy
    action = agent.policy(obs)

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    obs, reward, terminated, truncated, info = env.step(action)

# store RGB frames for the entire episode
frames = env.render()

# close the environment
env.close()

# create and play video clip using the frames and given fps
clip = mpy.ImageSequenceClip(frames, fps=50)
# clip.write_videofile("cartpole_agent.mp4")

# Exercise 5: Change the agent policy
class Agent():
    def __init__(self, env):
        self.env = env

    def policy(self, state):
        # push cart to the left/right if the pole angle and angular velocity are negative/positive
        theta = state[2]
        theta_dot = state[3]
        if theta < 0 and theta_dot <= 0:
            return 0
        if theta > 0 and theta_dot >= 0:
            return 1
        return self.env.action_space.sample()
