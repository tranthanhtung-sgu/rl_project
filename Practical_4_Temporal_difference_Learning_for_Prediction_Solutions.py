#!/usr/bin/env python3

# COMP6008 Reinforcement Learning
# Computing@Curtin University
#
# Practical 4 - Temporal-difference Learning for Prediction
#
# Learning outcomes: after completing this practical you will be able to
# * Implement model-free methods for policy evaluation (prediction problem):
#     * Monte Carlo (MC) learning
#     * Temporal-difference (TD) learning - TD(0) version
# * Evaluate policies in the Recycling Robot and OpenAI Gym environments

# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# create random number generator
rng = np.random.default_rng()

# enumerate states and actions for clarity
low, high = 0, 1
wait, search, recharge = 0, 1, 2

# Recycling Robot finite MDP
class RecyclingRobot(gym.Env):

    # make this MDP episodic by enforcing maximum number of time steps
    max_time_steps = 1000

    def __init__(self, alpha, beta, gamma, r_search, r_wait):
        super().__init__()

        # time step in the current episode
        self.time_step = 0

        # dimensionalities of state and action spaces
        self.num_states = 2
        self.num_actions = 3

        # use OpenAI Gym spaces to be consistent with the API
        self.observation_space = spaces.Discrete(self.num_states)
        self.action_space = spaces.Discrete(self.num_actions)

        # state-transition probabilities p(s'|s,a) = p[s,a,s']
        self.p = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.p[high, search, high] = alpha
        self.p[high, search, low] = 1 - alpha
        self.p[low, search, high] = 1 - beta
        self.p[low, search, low] = beta
        self.p[high, wait, high] = 1
        self.p[low, wait, low] = 1
        self.p[low, recharge, high] = 1
        # add extra transition not present in the original MDP
        self.p[high, recharge, high] = 1

        # reward function r(s,a,s') = r[s,a,s']
        self.r = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.r[high, search, high] = r_search
        self.r[high, search, low] = r_search
        self.r[low, search, high] = -3
        self.r[low, search, low] = r_search
        self.r[high, wait, high] = r_wait
        self.r[low, wait, low] = r_wait
        # add extra reward not present in the original MDP
        self.r[high, recharge, high] = -1

        # reset initial state
        self.reset()

    def reset(self):
        # reset the time step counter and state
        self.time_steps = 0
        self.state = high

        # return the state as initial observation
        # note: MDP is fully observable
        return self.state, {}

    def step(self, action):
        # count the number of time steps for each episode
        self.time_steps += 1

        # receive action, transition to next state according to the MDP and emit reward
        # note: this function is for sampling only; not used when solving MDP using dynamic programming
        next_state = rng.choice(self.num_states, p=self.p[self.state, action, :])
        reward = self.r[self.state, action, next_state]

        # end this episode if maximum number of time steps is reached
        truncated = (self.time_steps >= self.max_time_steps)
        self.state = next_state

        # return observation (state), reward, terminated, truncated, additional information
        return self.state, reward, False, truncated, {}

# MDP parameters
alpha = 0.9
beta = 0.4
gamma = 0.9
r_search = 2
r_wait = 0
state_names = ('low', 'high')
action_names = ('wait', 'search', 'recharge')

# create an instance of the Recycling Robot MDP
env = RecyclingRobot(alpha, beta, gamma, r_search, r_wait)
num_states = env.observation_space.n
num_actions = env.action_space.n

# define stochastic policy pi(a|s) = pi[s, a]
pi = np.zeros((num_states, num_actions))
pi[low, wait] = 0.2
pi[low, search] = 0.4
pi[low, recharge] = 0.4
pi[high, wait] = 0.5
pi[high, search] = 0.5

# Reference state and action values
# For reference, the state and action values obtained last week using the policy evaluation algorithm are listed below.
#
# v(low) = 7.176579305737281
# v(high) = 9.11838696099175
#
# q(low, wait) = 6.458921375163553
# q(low, search) = 6.507497509000967
# q(low, recharge) = 8.206548264892575
# q(high, wait) = 8.206548264892575
# q(high, search) = 10.031785575919672
# q(high, recharge) = 7.206548264892575

# Monte Carlo prediction; every-visit version
def MCPrediction(env, pi, gamma, max_episodes=20):
    # dimensionalities of state and action spaces
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # state value estimate
    V = np.zeros((num_states,))

    # sum of returns for each state
    G = np.zeros((num_states,))

    # number of times states were visited
    N = np.zeros((num_states,))

    # generate sample sequences (trajectories) for a number of episodes
    for _ in range(max_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        rewards = []
        states = []
        while not (terminated or truncated):
            # select action by following the policy being evaluated
            action = rng.choice(num_actions, p=pi[state, :])
            next_state, reward, terminated, truncated, _ = env.step(action)

            # store states visited and rewards received
            states.append(state)
            rewards.append(reward)
            state = next_state

        # calculate returns starting from the end of the episode
        G_t = 0
        for reward, state in zip(rewards[::-1], states[::-1]):
            G_t = gamma*G_t + reward

            # update
            G[state] += G_t
            N[state] += 1
            V[state] = G[state]/N[state]

    return V

# evaluate given policy
v_pi = MCPrediction(env, pi, gamma)
for n, v in enumerate(v_pi):
    print(f'v({state_names[n]}) = {v}')

# Exercise 1: Modify the code for incremental updates
# Monte Carlo prediction; every-visit, incremental version
def MCPredictionIncremental(env, pi, gamma, max_episodes=20):
    # dimensionalities of state and action spaces
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # state value estimate
    V = np.zeros((num_states,))

    # number of times states were visited
    N = np.zeros((num_states,))

    # generate sample sequences (trajectories) for a number of episodes
    for _ in range(max_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        rewards = []
        states = []
        while not (terminated or truncated):
            # select action by following the policy being evaluated
            action = rng.choice(num_actions, p=pi[state, :])
            next_state, reward, terminated, truncated, _ = env.step(action)

            # store states visited and rewards received
            states.append(state)
            rewards.append(reward)
            state = next_state

        # calculate returns starting from the end of the episode
        G_t = 0
        for reward, state in zip(rewards[::-1], states[::-1]):
            G_t = gamma*G_t + reward

            # update
            N[state] += 1
            V[state] += 1/N[state]*(G_t - V[state])

    return V

# evaluate given policy
v_pi = MCPredictionIncremental(env, pi, gamma)
for n, v in enumerate(v_pi):
    print(f'v({state_names[n]}) = {v}')

# TD(0) prediction; incremental version
def TDPrediction(env, pi, gamma, alpha=0.1, max_episodes=20):
    # dimensionalities of state and action spaces
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # state value estimate
    V = np.zeros((num_states,))

    # number of times states were visited
    N = np.zeros((num_states,))

    # generate sample sequences (trajectories) for a number of episodes
    for _ in range(max_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            # select action by following the policy being evaluated
            action = rng.choice(num_actions, p=pi[state, :])
            next_state, reward, terminated, truncated, _ = env.step(action)

            # update
            V[state] += alpha*(reward + gamma*V[next_state] - V[state])
            state = next_state

    return V

# evaluate given policy
v_pi = TDPrediction(env, pi, gamma)
for n, v in enumerate(v_pi):
    print(f'v({state_names[n]}) = {v}')

# Exercise 2: Sample experiences and store them as a list of tuples
# sample experiences and store them as a list of tuples (state, reward, next_state) for a number of episodes
def SampleExperiences(env, pi, num_episodes):
    # dimensionalities of state and action spaces
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # generate batch of experiences
    # note: one experience is a tuple (state, reward, next_state)
    batch = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            # select action by following the policy being evaluated
            action = rng.choice(num_actions, p=pi[state, :])
            next_state, reward, terminated, truncated, _ = env.step(action)

            # store experience
            batch.append((state, reward, next_state))
            state = next_state

    return batch

# temporal-difference prediction; batch version
def TDPredictionBatch(env, pi, gamma, num_episodes, alpha=0.5, threshold=0.001, max_iters=500):
    # dimensionalities of state and action spaces
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # sample batch of experiences
    batch = SampleExperiences(env, pi, num_episodes)
    V = np.zeros((num_states,))
    for i in range(max_iters):
        # summing updates for the entire batch
        V_update = np.zeros((num_states,))
        for (state, reward, next_state) in batch:
            V_update[state] += alpha*(reward + gamma*V[next_state] - V[state])

        # apply average update
        # note: this is the same as averaging reward for each state -> next_state transition
        V += V_update/len(batch)

        # check convergence
        delta = np.max(abs(V_update)/len(batch))
        if delta <= threshold:
            print(f'Batch TD converged in {i+1} iteration(s)')
            return V

    # if we ever reach this point, then the batch TD did not converge
    print(f'Batch TD did not converge within {max_iters} iterations: |V_batch - V| = {delta} > {threshold}')
    return V

# evaluate given policy
v_pi = TDPredictionBatch(env, pi, gamma, 5)
for n, v in enumerate(v_pi):
    print(f'v({state_names[n]}) = {v}')

# TD learning for policy evaluation in OpenAI Gym environments
# create Frozen Lake environment
env = gym.make('FrozenLake-v1')

# dimensionalities of state and action spaces
num_states = env.observation_space.n
num_actions = env.action_space.n

# create a random policy (equivalent to random sampling of the action space)
pi = np.tile(1/num_actions, (num_states, num_actions))
gamma = 0.999

# evaluate given policy
v_pi = TDPrediction(env, pi, gamma, max_episodes=10000)
for n, v in enumerate(v_pi):
    print(f'v({n}) = {v}')

plt.figure(dpi=100)
plt.imshow(v_pi.reshape((4, 4), ))
plt.colorbar()
plt.title(r'State value $v_\pi$')
plt.savefig('frozen_lake_state_values.png')
plt.close()

# close the environment
env.close()

# Exercise 3: Change the discount factor to gamma=0.1 and run the code again
# Note: In a script, you'd need to rerun the above code with gamma=0.1
