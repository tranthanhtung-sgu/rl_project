#!/usr/bin/env python3

# COMP6008 Reinforcement Learning
# Computing@Curtin University
#
# Practical 2 - Multi-armed Bandits
#
# Learning outcomes: after completing this practical you will be able to:
# * Create custom multi-armed bandit environment in OpenAI Gym
# * Implement agents that use constant/variable step estimation of the action-value
# * Implement various exploration strategies: ε-greedy, optimistic initial values, upper confidence bound, and softmax of preferences
# * Evaluate the agents in 10-armed testbed

# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# create random number generator
rng = np.random.default_rng()

# custom multi-armed bandit environment based on OpenAI Gym
class MAB(gym.Env):
    def __init__(self, arms, mean=0.0):
        super().__init__()
        # number of arms to pull
        self.arms = arms
        # mean action-value
        self.mean = mean
        # use OpenAI Gym spaces to be consistent with the API
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(arms)
        self.reset()

    def reset(self):
        # draw action-values from Normal(mean, 1)
        self.q = rng.normal(loc=self.mean, size=self.arms)
        # calculate optimal action; we will use it for evaluation later on
        self.optimal_action = np.argmax(self.q)
        # return the first observation (irrelevant because single state)
        return None

    def step(self, action):
        # draw reward from Normal(q, 1)
        reward = rng.normal(loc=self.q[action])
        # return observation, reward, terminated, truncated, additional information
        return None, reward, False, False, {}

# create an instance of the 10-armed testbed
ARMS = 10
env = MAB(ARMS)

# Agent with epsilon-greedy policy
class AgentGreedy():
    def __init__(self, env, alpha=None, epsilon=0.0):
        # environment; we use it to sample from action space
        self.env = env
        # step size; if None then we assume variable step size
        self.alpha = alpha
        # policy parameter
        self.epsilon = epsilon
        # action-value estimates
        self.Q = np.zeros(env.action_space.n)
        # action selection count
        self.N = np.zeros(env.action_space.n)

    def policy(self):
        if rng.uniform() >= self.epsilon:
            # exploitation with probability 1-epsilon; break ties randomly
            j = rng.permutation(self.env.action_space.n)
            return j[np.argmax(self.Q[j])]
        else:
            # exploration with probability epsilon
            return self.env.action_space.sample()

    def update(self, action, reward):
        if self.alpha is None:
            # variable step size
            self.N[action] += 1
            self.Q[action] += 1/self.N[action]*(reward - self.Q[action])
        else:
            # constant step size
            self.Q[action] += self.alpha*(reward - self.Q[action])

    def reset(self):
        self.Q[:] = 0
        self.N[:] = 0

# testbed parameters
STEPS = 1000
RUNS = 500

# evaluate and plot performance of agents
def testbed(env, agents, labels, steps, runs):
    num_agents = len(agents)
    average_rewards = np.zeros((steps, num_agents))
    optimal_actions = np.zeros((steps, num_agents))

    # average results over all runs
    for _ in range(runs):
        env.reset()
        # n is the index of agent in the list of agents
        for n, agent in enumerate(agents):
            agent.reset()
            # run agent-environment interaction loop for a number of steps
            for t in range(steps):
                # agent takes action
                action = agent.policy()
                # environment receives action and emits reward
                _, reward, _, _, _ = env.step(action)
                # agent updates its estimate of action-value
                agent.update(action, reward)
                # measure agent's performance at this time step
                average_rewards[t, n] += reward/runs
                optimal_actions[t, n] += (action == env.optimal_action)*100/runs

    # plot the results; first the average reward
    t = range(steps)
    plt.figure(figsize=(16, 6), dpi=80)
    # n is the index of agent in the list of agents
    for n, label in enumerate(labels):
        plt.plot(t, average_rewards[:, n], label=label)
    plt.ylabel('Average reward')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('mab_average_rewards.png')
    plt.close()
    
    # ... and second the average number of times the optimal action was taken
    plt.figure(figsize=(16, 6), dpi=80)
    for n, label in enumerate(labels):
        plt.plot(t, optimal_actions[:, n], label=label)
    plt.ylabel('% Optimal action')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('mab_optimal_actions.png')
    plt.close()

# create agents and corresponding labels for plotting
agents = [AgentGreedy(env),
          AgentGreedy(env, epsilon=0.01),
          AgentGreedy(env, epsilon=0.1)]

# use raw strings for agents' names in the form of LaTeX expressions
labels = [r'Greedy',
          r'$\varepsilon=0.01$',
          r'$\varepsilon=0.1$']

# evaluate and plot performance of agents
testbed(env, agents, labels, STEPS, RUNS)

# Exercise 1: Optimistic initial values
# Agent with optimistic initial action-values
class AgentOptimistic():
    def __init__(self, env, alpha=None, initial=0.0):
        # environment; we use it to sample from action space
        self.env = env
        # step size; if None then we assume variable step size
        self.alpha = alpha
        # policy parameter
        self.initial = initial
        # action-value estimates
        self.Q = np.zeros(env.action_space.n)
        # ... with initial values
        self.Q[:] = initial
        # action selection count
        self.N = np.zeros(env.action_space.n)

    def policy(self):
        # exploitation; break ties randomly
        j = rng.permutation(self.env.action_space.n)
        return j[np.argmax(self.Q[j])]

    def update(self, action, reward):
        if self.alpha is None:
            # variable step size
            self.N[action] += 1
            self.Q[action] += 1/self.N[action]*(reward - self.Q[action])
        else:
            # constant step size
            self.Q[action] += self.alpha*(reward - self.Q[action])

    def reset(self):
        self.Q[:] = self.initial
        self.N[:] = 0

# create agents and corresponding labels for plotting
agents = [AgentGreedy(env, alpha=0.1, epsilon=0.1),
          AgentOptimistic(env, alpha=0.1, initial=5.0)]

# use raw strings for agents' names in the form of LaTeX expressions
labels = [r'Greedy, $\alpha=0.1$',
          r'Optimistic $Q_1=5$, $\alpha=0.1$']

# evaluate and plot performance of agents
testbed(env, agents, labels, STEPS, RUNS)

# Exercise 2: Upper confidence bound (UCB)
# Agent with upper-confidence-bound action selection
class AgentUCB():
    def __init__(self, env, alpha=None, c=0.0):
        # environment; we use it to sample from action space
        self.env = env
        # step size; if None then we assume variable step size
        self.alpha = alpha
        # policy parameter
        self.c = c
        # action-value estimates
        self.Q = np.zeros(env.action_space.n)
        # action selection count
        self.N = np.zeros(env.action_space.n)
        # total time steps
        self.t = 1

    def policy(self):
        # exploitation based on upper-confidence-bound; break ties randomly
        ucb = self.Q + self.c*np.sqrt(np.log(self.t)/np.maximum(self.N, 0.1))
        j = rng.permutation(self.env.action_space.n)
        return j[np.argmax(ucb[j])]

    def update(self, action, reward):
        self.t += 1
        if self.alpha is None:
            # variable step size
            self.N[action] += 1
            self.Q[action] += 1/self.N[action]*(reward - self.Q[action])
        else:
            # constant step size
            self.Q[action] += self.alpha*(reward - self.Q[action])

    def reset(self):
        self.Q[:] = 0
        self.N[:] = 0

# create agents and corresponding labels for plotting
agents = [AgentGreedy(env, epsilon=0.1),
          AgentUCB(env, c=2)]

# use raw strings for agents' names in the form of LaTeX expressions
labels = [r'Greedy',
          r'UCB $c=2$']

# evaluate and plot performance of agents
testbed(env, agents, labels, STEPS, RUNS)

# Exercise 3: Gradient-based using softmax of preferences
# Agent with softmax policy
class AgentGradient():
    def __init__(self, env, alpha_R=None, alpha_H=0.0):
        # environment; we use it to sample from action space
        self.env = env
        # step size; if None then we assume variable step size
        self.alpha_R = alpha_R
        # policy parameter
        self.alpha_H = alpha_H
        # action preference
        self.H = np.zeros(env.action_space.n)
        # baseline reward
        self.R = 0.0
        # reward count
        self.N = 0
        # action probabilities
        self.pi = np.zeros(env.action_space.n)

    def policy(self):
        self._softmax()
        # action is drawn using action probabilities
        return rng.choice(env.action_space.n, p=self.pi)

    def update(self, action, reward):
        # update preferences
        self._softmax()
        p = -self.pi
        p[action] += 1
        self.H += self.alpha_H*(reward - self.R)*p

        # update average reward
        if self.alpha_R is None:
            # variable step size
            self.N += 1
            self.R += 1/self.N*(reward - self.R)
        else:
            # constant step size
            self.R += self.alpha_R*(reward - self.R)

    def reset(self):
        self.H[:] = 0.0
        self.R = 0
        self.N = 0

    def _softmax(self):
        # calculate action probabilities using softmax of preferences
        x = np.exp(self.H - self.H.max())
        self.pi = x/x.sum()

# create an instance of MAB environment with mean rewards +4 (see textbook, pages 37-38)
arms = 10
env = MAB(arms, mean=4)

# create agents and corresponding labels for plotting
agents = [AgentGradient(env, alpha_R=0.0, alpha_H=0.1),
          AgentGradient(env, alpha_H=0.1)]

# use raw strings for agents' names in the form of LaTeX expressions
labels = [r'Gradient without baseline $\alpha=0.1$',
          r'Gradient with baseline $\alpha=0.1$']

# evaluate and plot performance of agents
testbed(env, agents, labels, STEPS, RUNS)
