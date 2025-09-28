#!/usr/bin/env python3

# COMP6008 Reinforcement Learning
# Computing@Curtin University
#
# Practical 6 - Model-based RL
#
# Learning outcomes: after completing this practical you will be able to
# * Implement table lookup model for stochastic environments
# * Implement model-based Dyna-Q method for policy optimisation (control problem)
# * Apply Dyna-Q to gridworld environments

# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import moviepy as mpy

# create random number generator
rng = np.random.default_rng()

# Off-policy TD control (Q-learning)
def Qlearning(env, gamma, alpha=0.1, epsilon=1, max_episodes=100):
    # dimensionalities of state and action spaces
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # initialise behaviour policy
    behaviour = np.tile(1/num_actions, (num_states, num_actions))

    # action value estimate
    Q = np.zeros((num_states, num_actions))

    # steps per episode
    num_steps = np.zeros((max_episodes,))

    # sum of rewards per episode
    rewards = np.zeros((max_episodes,))

    # generate sample sequences (trajectories) for a number of episodes
    for episode in range(max_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            # select action by following the behaviour policy
            action = rng.choice(num_actions, p=behaviour[state, :])
            next_state, reward, terminated, truncated, _ = env.step(action)

            # update using the maximum action value for the next state
            Q[state, action] += alpha*(reward + gamma*np.max(Q[next_state, :]) - Q[state, action])

            state = next_state

            # update behaviour policy
            # note: do not overwrite the `state` variable; loop over `s` instead
            for s in range(num_states):
                # select best action with ties broken randomly
                j = rng.permutation(num_actions)
                best_action = j[np.argmax(Q[s, j])]

                # construct epsilon-greedy policy with respect to the action values
                behaviour[s, :] = epsilon/num_actions
                behaviour[s, best_action] = 1 - epsilon + epsilon/num_actions

            num_steps[episode] += 1
            rewards[episode] += reward

    # construct target policy from the action values
    pi = np.zeros((num_states, num_actions))
    for state in range(num_states):
        # select best action
        best_action = np.argmax(Q[state, :])
        pi[state, best_action] = 1

    return Q, pi, num_steps, rewards

# Table lookup model; assumes stochastic environment
class TableLookup():

    def __init__(self, env):
        # dimensionalities of state and action spaces
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n

        # lookup tables
        # state-transition probabilities p(s'|s,a) = p[s,a,s']
        self.p = np.zeros((self.num_states, self.num_actions, self.num_states))

        # rewards r(s,a) = r[s,a]
        self.r = np.zeros((self.num_states, self.num_actions))

        # the number of times each state-action pair has been visited N(s,a) = N[s,a]
        self.N = np.zeros((self.num_states, self.num_actions))

    def learn(self, state, action, reward, next_state):
        # update visit count and rewards
        self.N[state, action] += 1
        self.r[state, action] += 1/self.N[state, action]*(reward - self.r[state, action])

        # update state-transition probabilities
        targets = np.zeros((self.num_states,))
        targets[next_state] = 1
        self.p[state, action, :] += 1/self.N[state, action]*(targets - self.p[state, action, :])

    def simulate(self, state, action):
        # return reward and next state for the given state-action pair
        reward = self.r[state, action]
        state_next = rng.choice(self.num_states, p=self.p[state, action, :])
        return reward, state_next

# Dyna-Q; model assumes stochastic environment (original algorithm assumes deterministic environment)
def DynaQ(env, gamma, alpha=0.1, epsilon=1, planning_steps=0, max_episodes=100):
    # dimensionalities of state and action spaces
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # initialise behaviour policy
    behaviour = np.tile(1/num_actions, (num_states, num_actions))

    # action value estimate
    Q = np.zeros((num_states, num_actions))

    # visited state-action pairs
    visited = np.zeros((num_states, num_actions))

    # model of environment
    model = TableLookup(env)

    # steps per episode
    num_steps = np.zeros((max_episodes,))

    # sum of rewards per episode
    rewards = np.zeros((max_episodes,))

    # perform direct RL, model learning and planning for a number of episodes
    for episode in range(max_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            # select action by following the behaviour policy
            action = rng.choice(num_actions, p=behaviour[state, :])

            # update visited state-action pair
            visited[state, action] = 1

            # observe reward and next_state
            next_state, reward, terminated, truncated, _ = env.step(action)

            # direct RL update using the maximum action value for the next state
            Q[state, action] += alpha*(reward + gamma*np.max(Q[next_state, :]) - Q[state, action])

            # model learning
            model.learn(state, action, reward, next_state)

            # planning: do not overwrite `state` and `next_state` variables
            visited_list = np.argwhere(visited)
            for n in range(planning_steps):
                # select state-action pair that has been visited using uniform random distribution
                s, a = rng.choice(visited_list)

                # simulate experience (reward and next state)
                r, next_s = model.simulate(s, a)

                # perform planning update using the maximum action value for the next state
                Q[s, a] += alpha*(r + gamma*np.max(Q[next_s, :]) - Q[s, a])

            state = next_state

            # update behaviour policy
            # note: do not overwrite the `state` variable; loop over `s` instead
            for s in range(num_states):
                # select best action with ties broken randomly
                j = rng.permutation(num_actions)
                best_action = j[np.argmax(Q[s, j])]

                # construct epsilon-greedy policy with respect to the action values
                behaviour[s, :] = epsilon/num_actions
                behaviour[s, best_action] = 1 - epsilon + epsilon/num_actions

            num_steps[episode] += 1
            rewards[episode] += reward

    # construct target policy from the action values
    pi = np.zeros((num_states, num_actions))
    for state in range(num_states):
        # select best action
        best_action = np.argmax(Q[state, :])
        pi[state, best_action] = 1

    return Q, pi, num_steps, rewards

# Exercise 1: Apply Q-learning and Dyna-Q with 5 and 50 planning steps to the undiscounted cliff-walking environment
# create undiscounted cliff-walking environment
env = gym.make('CliffWalking-v1', render_mode="rgb_array_list")
gamma = 1

# dimensionalities of state and action spaces
num_states = env.observation_space.n
num_actions = env.action_space.n

max_episodes = 100

alpha = 0.1
epsilon = 0.1
planning_steps = [0, 5, 50]
num_steps = np.zeros((len(planning_steps), max_episodes))
rewards = np.zeros((len(planning_steps), max_episodes))

# Dyna-Q
for n, ps in enumerate(planning_steps):
    _, pi, num_steps[n], rewards[n] = DynaQ(env, gamma, alpha=alpha, epsilon=epsilon,
                                        planning_steps=ps, max_episodes=max_episodes)
    print(f'Dyna-Q with {ps} planning steps done.')

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
max_steps = 100
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = rng.choice(num_actions, p=pi[state, :])

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
# clip.write_videofile("cliff_walking_dyna_q.mp4")

# plot number of steps per episode during learning
episodes = range(1, len(num_steps[0]))
plt.figure(dpi=150)
colours = ['b', 'g', 'r']
for n, ps in enumerate(planning_steps):
    plt.plot(episodes, num_steps[n, 1:], colours[n], label=f'{ps} planning steps')

plt.xlabel('Episodes')
plt.ylabel('Steps per episode')
plt.legend(loc='upper right')
plt.grid()
plt.savefig('cliff_walking_dyna_q_steps.png')
plt.close()

# Exercise 2: Apply Q-learning and Dyna-Q with 5 and 50 planning steps to the undiscounted frozen lake and taxi environments
# create undiscounted Frozen Lake environment
env = gym.make('FrozenLake-v1', render_mode="rgb_array_list")
gamma = 1

# dimensionalities of state and action spaces
num_states = env.observation_space.n
num_actions = env.action_space.n

max_episodes = 100

alpha = 0.1
epsilon = 0.1
planning_steps = [0, 5, 50]
num_steps = np.zeros((len(planning_steps), max_episodes))
rewards = np.zeros((len(planning_steps), max_episodes))

# Dyna-Q
for n, ps in enumerate(planning_steps):
    _, pi, num_steps[n], rewards[n] = DynaQ(env, gamma, alpha=alpha, epsilon=epsilon,
                                        planning_steps=ps, max_episodes=max_episodes)
    print(f'Dyna-Q with {ps} planning steps done.')

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
max_steps = 100
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = rng.choice(num_actions, p=pi[state, :])

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
# clip.write_videofile("frozen_lake_dyna_q.mp4")

# SIMULATION 1
# plot cumulative rewards per episode during learning
episodes = range(1, len(num_steps[0]))
plt.figure(dpi=150)
colours = ['b', 'g', 'r']
for n, ps in enumerate(planning_steps):
    plt.plot(episodes, np.cumsum(rewards[n, 1:]), colours[n], label=f'{ps} planning steps')

plt.xlabel('Episodes')
plt.ylabel('Cumulative rewards per episode')
plt.legend(loc='upper left')
plt.grid()
plt.savefig('frozen_lake_dyna_q_rewards_sim1.png')
plt.close()

# SIMULATION 2
# plot cumulative rewards per episode during learning
episodes = range(1, len(num_steps[0]))
plt.figure(dpi=150)
colours = ['b', 'g', 'r']
for n, ps in enumerate(planning_steps):
    plt.plot(episodes, np.cumsum(rewards[n, 1:]), colours[n], label=f'{ps} planning steps')

plt.xlabel('Episodes')
plt.ylabel('Cumulative rewards per episode')
plt.legend(loc='upper left')
plt.grid()
plt.savefig('frozen_lake_dyna_q_rewards_sim2.png')
plt.close()

# SIMULATION 3
# plot cumulative rewards per episode during learning
episodes = range(1, len(num_steps[0]))
plt.figure(dpi=150)
colours = ['b', 'g', 'r']
for n, ps in enumerate(planning_steps):
    plt.plot(episodes, np.cumsum(rewards[n, 1:]), colours[n], label=f'{ps} planning steps')

plt.xlabel('Episodes')
plt.ylabel('Cumulative rewards per episode')
plt.legend(loc='upper left')
plt.grid()
plt.savefig('frozen_lake_dyna_q_rewards_sim3.png')
plt.close()

# create undiscounted Taxi environment
env = gym.make('Taxi-v3', render_mode="rgb_array_list")
gamma = 1

# dimensionalities of state and action spaces
num_states = env.observation_space.n
num_actions = env.action_space.n

max_episodes = 100

alpha = 0.5
epsilon = 0.2
planning_steps = [0, 5, 50]
num_steps = np.zeros((len(planning_steps), max_episodes))
rewards = np.zeros((len(planning_steps), max_episodes))

# Dyna-Q
for n, ps in enumerate(planning_steps):
    _, pi, num_steps[n], rewards[n] = DynaQ(env, gamma, alpha=alpha, epsilon=epsilon,
                                        planning_steps=ps, max_episodes=max_episodes)
    print(f'Dyna-Q with {ps} planning steps done.')

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
max_steps = 100
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = rng.choice(num_actions, p=pi[state, :])

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
# clip.write_videofile("taxi_dyna_q.mp4")

# plot cumulative rewards per episode during learning
episodes = range(1, len(num_steps[0]))
plt.figure(dpi=150)
colours = ['b', 'g', 'r']
for n, ps in enumerate(planning_steps):
    plt.plot(episodes, np.cumsum(rewards[n, 1:]), colours[n], label=f'{ps} planning steps')

plt.xlabel('Episodes')
plt.ylabel('Cumulative rewards per episode')
plt.legend(loc='upper right')
plt.grid()
plt.savefig('taxi_dyna_q_rewards.png')
plt.close()

# Dyna-Q; starts counting episodes only after observing the first non-zero reward
def DynaQwait(env, gamma, alpha=0.1, epsilon=1, planning_steps=0, max_episodes=100):
    # dimensionalities of state and action spaces
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # initialise behaviour policy
    behaviour = np.tile(1/num_actions, (num_states, num_actions))

    # action value estimate
    Q = np.zeros((num_states, num_actions))

    # visited state-action pairs
    visited = np.zeros((num_states, num_actions))

    # model of environment
    model = TableLookup(env)

    # perform direct RL until the first non-zero reward is observed
    zero_reward = True
    while zero_reward:
        state, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            # select action by following the behaviour policy
            action = rng.choice(num_actions, p=behaviour[state, :])

            # update visited state-action pair
            visited[state, action] = 1

            # observe reward and next_state
            next_state, reward, terminated, truncated, _ = env.step(action)
            if reward != 0:
                zero_reward = False

            # direct RL update using the maximum action value for the next state
            Q[state, action] += alpha*(reward + gamma*np.max(Q[next_state, :]) - Q[state, action])

            # model learning
            model.learn(state, action, reward, next_state)

            state = next_state

            # update behaviour policy
            # note: do not overwrite the `state` variable; loop over `s` instead
            for s in range(num_states):
                # select best action with ties broken randomly
                j = rng.permutation(num_actions)
                best_action = j[np.argmax(Q[s, j])]

                # construct epsilon-greedy policy with respect to the action values
                behaviour[s, :] = epsilon/num_actions
                behaviour[s, best_action] = 1 - epsilon + epsilon/num_actions

    # steps per episode
    num_steps = np.zeros((max_episodes,))

    # sum of rewards per episode
    rewards = np.zeros((max_episodes,))

    # perform direct RL, model learning and planning for a number of episodes
    for episode in range(max_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            # select action by following the behaviour policy
            action = rng.choice(num_actions, p=behaviour[state, :])

            # update visited state-action pair
            visited[state, action] = 1

            # observe reward and next_state
            next_state, reward, terminated, truncated, _ = env.step(action)

            # direct RL update using the maximum action value for the next state
            Q[state, action] += alpha*(reward + gamma*np.max(Q[next_state, :]) - Q[state, action])

            # model learning
            model.learn(state, action, reward, next_state)

            # planning: do not overwrite `state` and `next_state` variables
            visited_list = np.argwhere(visited)
            for n in range(planning_steps):
                # select state-action pair that has been visited using uniform random distribution
                s, a = rng.choice(visited_list)

                # simulate experience (reward and next state)
                r, next_s = model.simulate(s, a)

                # perform planning update using the maximum action value for the next state
                Q[s, a] += alpha*(r + gamma*np.max(Q[next_s, :]) - Q[s, a])

            state = next_state

            # update behaviour policy
            # note: do not overwrite the `state` variable; loop over `s` instead
            for s in range(num_states):
                # select best action with ties broken randomly
                j = rng.permutation(num_actions)
                best_action = j[np.argmax(Q[s, j])]

                # construct epsilon-greedy policy with respect to the action values
                behaviour[s, :] = epsilon/num_actions
                behaviour[s, best_action] = 1 - epsilon + epsilon/num_actions

            num_steps[episode] += 1
            rewards[episode] += reward

    # construct target policy from the action values
    pi = np.zeros((num_states, num_actions))
    for state in range(num_states):
        # select best action
        best_action = np.argmax(Q[state, :])
        pi[state, best_action] = 1

    return Q, pi, num_steps, rewards

# create undiscounted Frozen Lake environment
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="rgb_array_list")
gamma = 1

# dimensionalities of state and action spaces
num_states = env.observation_space.n
num_actions = env.action_space.n

max_episodes = 100

alpha = 0.1
epsilon = 0.1
planning_steps = [0, 5, 50]
num_steps = np.zeros((len(planning_steps), max_episodes))
rewards = np.zeros((len(planning_steps), max_episodes))

# Dyna-Q
for n, ps in enumerate(planning_steps):
    _, pi, num_steps[n], rewards[n] = DynaQwait(env, gamma, alpha=alpha, epsilon=epsilon,
                                            planning_steps=ps, max_episodes=max_episodes)
    print(f'Dyna-Q with {ps} planning steps done.')

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
max_steps = 100
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = rng.choice(num_actions, p=pi[state, :])

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
# clip.write_videofile("frozen_lake_dyna_q_wait.mp4")

# plot cumulative rewards per episode during learning
episodes = range(1, len(num_steps[0]))
plt.figure(dpi=150)
colours = ['b', 'g', 'r']
for n, ps in enumerate(planning_steps):
    plt.plot(episodes, np.cumsum(rewards[n, 1:]), colours[n], label=f'{ps} planning steps')

plt.xlabel('Episodes')
plt.ylabel('Cumulative rewards per episode')
plt.legend(loc='upper left')
plt.grid()
plt.savefig('frozen_lake_dyna_q_wait_rewards.png')
plt.close()
