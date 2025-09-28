#!/usr/bin/env python3

# COMP6008 Reinforcement Learning
# Computing@Curtin University
#
# Practical 7 - Value Function Approximation
#
# Learning outcomes: after completing this practical you will be able to
# * Construct features for continuous state spaces using tile coding
# * Implement on-policy control with linear action value approximation (semi-gradient linear SARSA)
# * Apply it to classic control environments in OpenAI Gym

# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import moviepy as mpy

# create random number generator
rng = np.random.default_rng()

# Feature construction using table lookup (discrete state spaces only)
class TableLookupFeatures():
    def __init__(self, env):
        # check if the state space has correct type
        assert isinstance(env.observation_space, spaces.Discrete), 'Observation space must be discrete'
        self.size = env.observation_space.n

    def __call__(self, state):
        # construct zero-one feature vector
        features = np.zeros((self.size,))
        features[state] = 1
        return features

# Feature construction using tile coding (continuous state spaces only)
class TileCodingFeatures():
    def __init__(self, env, tiling, offsets):
        # check if the state space has correct type
        continuous = isinstance(env.observation_space, spaces.Box) and len(env.observation_space.shape) == 1
        assert continuous, 'Observation space must be continuous with shape (n,)'

        # generate reference tiling without offset
        self.offsets = offsets
        self.bounds = []
        self.num_tiles = []
        for low, high, num_tiles in tiling:
            points = np.linspace(low, high, num_tiles+1)
            # bounds are points that separate tiles in each dimension
            self.bounds.append(np.array([*points[1:-1], np.inf]))
            self.num_tiles.append(num_tiles)

        self.tiling_size = np.prod(self.num_tiles)
        self.size = self.tiling_size*len(offsets)

    def __call__(self, state):
        # feature vector for all tilings
        features = np.zeros((self.tiling_size, len(self.offsets)))
        for n, offset in enumerate(self.offsets):
            # construct feature vector for one tiling with given offset
            indices = [np.searchsorted(self.bounds[k] + offset[k], state[k]) for k in range(state.size)]
            tiling_features = np.zeros((*self.num_tiles,))
            tiling_features[(*indices,)] = 1
            # stack feature vector for this tiling with the rest
            features[:, n] = tiling_features.reshape((self.tiling_size,))
        return features.reshape((self.size,))

# On-policy control with approximation (semi-gradient SARSA)
class LinearSARSA():
    def __init__(self, env, gamma, features, epsilon):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = env.action_space.n

        # since non-zero components of the feature vector are the same for each action
        self.features = features
        # ... we can account for different actions by changing the shape of weights instead of copying features
        self.weights = np.zeros((features.size, self.num_actions))

    def policy(self, state, qhat=None):
        # action depends on the approximated action values in qhat
        if qhat is None:
            qhat = self.weights.T @ self.features(state)

        if rng.uniform() >= self.epsilon:
            # exploitation with probability 1-epsilon; break ties randomly
            j = rng.permutation(self.num_actions)
            return j[np.argmax(qhat[j])]
        else:
            # exploration with probability epsilon
            return self.env.action_space.sample()

    def train(self, alpha=0.1, max_episodes=100):
        rewards = np.zeros((max_episodes,))
        num_steps = np.zeros((max_episodes,))
        for episode in range(max_episodes):
            state, _ = env.reset()
            terminated = False
            truncated = False

            # select the first action by following epsilon-greedy policy and linear action values
            features = self.features(state)
            qhat = self.weights.T @ features
            action = self.policy(state, qhat)

            while not (terminated or truncated):
                # send the action to the environment
                next_state, reward, terminated, truncated, _ = env.step(action)

                # select next action in the next state by following the same policy as before
                next_features = self.features(next_state)
                next_qhat = self.weights.T @ next_features
                next_action = self.policy(next_state, next_qhat)

                # update weights
                delta = reward + self.gamma*next_qhat[next_action] - qhat[action]
                self.weights[:, action] += alpha*delta*features

                features = next_features
                qhat = next_qhat
                state = next_state
                action = next_action

                rewards[episode] += reward
                num_steps[episode] += 1

            print(f'\rEpisode {episode+1} done: steps = {num_steps[episode]}, rewards = {rewards[episode]}', end='')

        return rewards, num_steps

    def reset(self):
        self.weights[:] = 0

# Validating semi-gradient linear SARSA in the cliff-walking environment
# create the cliff-walking environment
env = gym.make('CliffWalking-v1', render_mode="rgb_array_list")

gamma = 1
alpha = 0.5
epsilon = 0.1
max_episodes = 200

features = TableLookupFeatures(env)

agent = LinearSARSA(env, gamma, features, epsilon)
rewards, steps = agent.train(alpha, max_episodes)

# plot number of steps per episode during learning
episodes = range(1, max_episodes+1)
plt.figure(dpi=100)
plt.plot(episodes, steps, label=f'Steps')

plt.xlabel('Episodes')
plt.ylabel('Steps per episode')
plt.legend(loc='upper right')
plt.grid()
plt.savefig('cliff_walking_linear_sarsa_steps.png')
plt.close()

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
max_steps = 100
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = agent.policy(state)

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    state, reward, terminated, truncated, info = env.step(action)
    steps += 1

# store RGB frames for the entire episode
frames = env.render()

# close the environment
env.close()

# create and play video clip using the frames and given fps
clip = mpy.ImageSequenceClip(frames, fps=10)
# clip.write_videofile("cliff_walking_linear_sarsa.mp4")

# Applying semi-gradient linear SARSA to classic control environments
# create the mountain car environment
env = gym.make('MountainCar-v0', render_mode="rgb_array_list")

gamma = 1
alpha = 0.5/8
epsilon = 0.0
max_episodes = 200

# parameters for the reference tiling
position = [-1.2, 0.6]
velocity = [-0.07, 0.07]
num_tiles = 8
tiling = [[*position, num_tiles], [*velocity, num_tiles]]

# parameters for tiling offsets
num_offsets = 8
dpos = np.ptp(position)/num_tiles/num_offsets
dvel = np.ptp(velocity)/num_tiles/num_offsets
offsets = [[i*dpos, i*dvel] for i in range(num_offsets)]

features = TileCodingFeatures(env, tiling, offsets)

agent = LinearSARSA(env, gamma, features, epsilon)
rewards, steps = agent.train(alpha, max_episodes)

# plot cost-to-go
num_points = 50
ctg = np.zeros((num_points, num_points))
pos = np.zeros((num_points, num_points))
vel = np.zeros((num_points, num_points))
for i, p in enumerate(np.linspace(*position, num_points)):
    for j, v in enumerate(np.linspace(*velocity, num_points)):
        pos[i, j] = p
        vel[i, j] = v
        ctg[i, j] = -np.max(agent.weights.T @ features(np.array([p, v])))

# 3D plot similar to that shown in the textbook (page 245)
fig = plt.figure(dpi=150)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(pos, vel, ctg, color='w')
ax.view_init(50, -70)
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.savefig('mountain_car_cost_to_go.png')
plt.close()

# plot number of steps per episode during learning (fewer the better)
episodes = range(1, max_episodes+1)
plt.figure(dpi=100)
plt.plot(episodes, steps, label=f'Steps')

plt.xlabel('Episodes')
plt.ylabel('Steps per episode')
plt.legend(loc='upper right')
plt.grid()
plt.savefig('mountain_car_linear_sarsa_steps.png')
plt.close()

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
max_steps = 200
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = agent.policy(state)

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    state, reward, terminated, truncated, info = env.step(action)
    steps += 1

# store RGB frames for the entire episode
frames = env.render()

# close the environment
env.close()

# create and play video clip using the frames and given fps
clip = mpy.ImageSequenceClip(frames, fps=30)
# clip.write_videofile("mountain_car_linear_sarsa.mp4")

# Exercise 1: Apply semi-gradient linear SARSA to the cart pole environment
# create cart pole environment
env = gym.make('CartPole-v1', render_mode="rgb_array_list")

gamma = 1
alpha = 0.05
epsilon = 0.2
max_episodes = 200

# parameters for the reference tiling
position = [-4.8, 4.8]
velocity = [-2*4.8, 2*4.8]
angular_position = [-0.42, 0.42]
angular_velocity = [-2*0.42, 2*0.42]
num_tiles = 8
tiling = [[*position, num_tiles], [*velocity, num_tiles],
          [*angular_position, num_tiles], [*angular_velocity, num_tiles]]

# parameters for tiling offsets
num_offsets = 8
dpos = np.ptp(position)/num_tiles/num_offsets
dvel = np.ptp(velocity)/num_tiles/num_offsets
dangpos = np.ptp(angular_position)/num_tiles/num_offsets
dangvel = np.ptp(angular_velocity)/num_tiles/num_offsets
offsets = [[i*dpos, i*dvel, i*dangpos, i*dangvel] for i in range(num_offsets)]

features = TileCodingFeatures(env, tiling, offsets)

agent = LinearSARSA(env, gamma, features, epsilon)
rewards, steps = agent.train(alpha, max_episodes)

# plot number of steps per episode during learning (more the better)
episodes = range(1, max_episodes+1)
plt.figure(dpi=100)
plt.plot(episodes, steps, label=f'Steps')

plt.xlabel('Episodes')
plt.ylabel('Steps per episode')
plt.legend(loc='upper right')
plt.grid()
plt.savefig('cart_pole_linear_sarsa_steps.png')
plt.close()

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
max_steps = 500
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = agent.policy(state)

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    state, reward, terminated, truncated, info = env.step(action)
    steps += 1

# store RGB frames for the entire episode
frames = env.render()

# close the environment
env.close()

# create and play video clip using the frames and given fps
clip = mpy.ImageSequenceClip(frames, fps=50)
# clip.write_videofile("cart_pole_linear_sarsa.mp4")
