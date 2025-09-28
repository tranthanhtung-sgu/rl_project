#!/usr/bin/env python3

# COMP6008 Reinforcement Learning
# Computing@Curtin University
#
# Practical 5 - Temporal-difference Learning for Control
#
# Learning outcomes: after completing this practical you will be able to
# * Implement model-free methods for policy optimisation (control problem):
#     * On-policy Monte Carlo control
#     * On-policy temporal-difference control (SARSA)
#     * Off-policy temporal-difference control (Q-learning)
# * Apply MC and TD control in the Recycling Robot and Cliff Walking environments

# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import moviepy as mpy
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

# On-policy Monte Carlo control; every-visit version
def MCControl(env, gamma, epsilon=0.05, max_episodes=100):
    # dimensionalities of state and action spaces
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # target policy
    pi = np.tile(1/num_actions, (num_states, num_actions))

    # action value estimate
    Q = np.zeros((num_states, num_actions))

    # number of times state-action pairs were visited
    N = np.zeros((num_states, num_actions))

    # generate sample sequences (trajectories) for a number of episodes
    for _ in range(max_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        experience = []
        while not (terminated or truncated):
            # select action by following the target policy
            action = rng.choice(num_actions, p=pi[state, :])
            next_state, reward, terminated, truncated, _ = env.step(action)

            # store states visited, actions taken and rewards received
            experience.append((state, action, reward))
            state = next_state

        # calculate returns starting from the end of the episode
        G_t = 0
        for state, action, reward in experience[::-1]:
            G_t = gamma*G_t + reward

            # update action value estimate
            N[state, action] += 1
            Q[state, action] += 1/N[state, action]*(G_t - Q[state, action])

        # update target policy
        for state in range(num_states):
            # select best action with ties broken randomly
            j = rng.permutation(num_actions)
            best_action = j[np.argmax(Q[state, j])]

            # construct epsilon-greedy policy with respect to the action values
            pi[state, :] = epsilon/num_actions
            pi[state, best_action] = 1 - epsilon + epsilon/num_actions

    return Q, pi

# find optimal policy
q_opt, pi_opt = MCControl(env, gamma)

for n in range(env.num_states):
    print(f'pi({state_names[n]}) = {pi_opt[n]}')

for n, qs in enumerate(q_opt):
    for k, qsa in enumerate(qs):
        print(f'q({state_names[n]}, {action_names[k]}) = {qsa}')

# Exercise 1: Change the every-visit on-policy Monte Carlo algorithm to be greedy in the limit of infinite exploration (GLIE)
# On-policy Monte Carlo control; every-visit version
def MCControlGLIE(env, gamma, max_episodes=100):
    # dimensionalities of state and action spaces
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # target policy
    pi = np.tile(1/num_actions, (num_states, num_actions))

    # action value estimate
    Q = np.zeros((num_states, num_actions))

    # number of times state-action pairs were visited
    N = np.zeros((num_states, num_actions))

    # generate sample sequences (trajectories) for a number of episodes
    for k in range(max_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        experience = []
        while not (terminated or truncated):
            # select action by following the target policy
            action = rng.choice(num_actions, p=pi[state, :])
            next_state, reward, terminated, truncated, _ = env.step(action)

            # store states visited, actions taken and rewards received
            experience.append((state, action, reward))
            state = next_state

        # calculate returns starting from the end of the episode
        G_t = 0
        for state, action, reward in experience[::-1]:
            G_t = gamma*G_t + reward

            # update action value estimate
            N[state, action] += 1
            Q[state, action] += 1/N[state, action]*(G_t - Q[state, action])

        # update target policy
        for state in range(num_states):
            # select best action with ties broken randomly
            j = rng.permutation(num_actions)
            best_action = j[np.argmax(Q[state, j])]

            # construct epsilon-greedy policy with respect to the action values
            epsilon = 1/(k+1)
            pi[state, :] = epsilon/num_actions
            pi[state, best_action] = 1 - epsilon + epsilon/num_actions

    return Q, pi

# find optimal policy
q_opt, pi_opt = MCControlGLIE(env, gamma)

for n in range(env.num_states):
    print(f'pi({state_names[n]}) = {pi_opt[n]}')

for n, qs in enumerate(q_opt):
    for k, qsa in enumerate(qs):
        print(f'q({state_names[n]}, {action_names[k]}) = {qsa}')

# On-policy TD control (SARSA)
def SARSA(env, gamma, alpha=0.1, epsilon=0.05, max_episodes=100):
    # dimensionalities of state and action spaces
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # initialise target policy
    pi = np.tile(1/num_actions, (num_states, num_actions))

    # action value estimate
    Q = np.zeros((num_states, num_actions))

    # generate sample sequences (trajectories) for a number of episodes
    for _ in range(max_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False

        # select the first action by following the target policy
        action = rng.choice(num_actions, p=pi[state, :])

        while not (terminated or truncated):
            # send the action to the environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            # select next action from next state by again following the target policy
            next_action = rng.choice(num_actions, p=pi[next_state, :])

            # update
            Q[state, action] += alpha*(reward + gamma*Q[next_state, next_action] - Q[state, action])

            state = next_state
            action = next_action

            # update target policy
            # note: do not overwrite the `state` variable; loop over `s` instead
            for s in range(num_states):
                # select best action with ties broken randomly
                j = rng.permutation(num_actions)
                best_action = j[np.argmax(Q[s, j])]

                # construct epsilon-greedy policy with respect to the action values
                pi[s, :] = epsilon/num_actions
                pi[s, best_action] = 1 - epsilon + epsilon/num_actions

    return Q, pi

# find optimal policy
q_opt, pi_opt = SARSA(env, gamma)

for n in range(env.num_states):
    print(f'pi({state_names[n]}) = {pi_opt[n]}')

for n, qs in enumerate(q_opt):
    for k, qsa in enumerate(qs):
        print(f'q({state_names[n]}, {action_names[k]}) = {qsa}')

# Exercise 2: Change the SARSA algorithm to Q-learning
# Off-policy TD control (Q-learning)
def Qlearning(env, gamma, alpha=0.1, epsilon=1, max_episodes=100):
    # dimensionalities of state and action spaces
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # initialise behaviour policy
    behaviour = np.tile(1/num_actions, (num_states, num_actions))

    # action value estimate
    Q = np.zeros((num_states, num_actions))

    # generate sample sequences (trajectories) for a number of episodes
    for _ in range(max_episodes):
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

    # construct target policy from the action values
    pi = np.zeros((num_states, num_actions))
    for state in range(num_states):
        # select best action
        best_action = np.argmax(Q[state, :])
        pi[state, best_action] = 1

    return Q, pi

# find optimal policy
q_opt, pi_opt = Qlearning(env, gamma)

for n in range(env.num_states):
    print(f'pi({state_names[n]}) = {pi_opt[n]}')

for n, qs in enumerate(q_opt):
    for k, qsa in enumerate(qs):
        print(f'q({state_names[n]}, {action_names[k]}) = {qsa}')

# TD control in OpenAI Gym environments
# create undiscounted cliff-walking environment
env = gym.make('CliffWalking-v1', render_mode="rgb_array_list")
gamma = 1

# dimensionalities of state and action spaces
num_states = env.observation_space.n
num_actions = env.action_space.n

# find optimal policy
q_opt, pi_opt = SARSA(env, gamma, alpha=0.5, epsilon=0.2, max_episodes=100)

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
max_steps = 100
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = rng.choice(num_actions, p=pi_opt[state, :])

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
# clip.write_videofile("cliff_walking_sarsa.mp4")

# create undiscounted cliff-walking environment
env = gym.make('CliffWalking-v1', render_mode="rgb_array_list")
gamma = 1

# dimensionalities of state and action spaces
num_states = env.observation_space.n
num_actions = env.action_space.n

# find optimal policy
q_opt, pi_opt = Qlearning(env, gamma, alpha=0.5, epsilon=0.2, max_episodes=100)

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
max_steps = 100
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = rng.choice(num_actions, p=pi_opt[state, :])

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
# clip.write_videofile("cliff_walking_qlearning.mp4")

# Exercise 3: Implement the SARSA(λ) algorithm with eligibility traces
# On-policy TD control (SARSA with eligibility traces)
def SARSATraces(env, gamma, alpha=0.1, epsilon=0.05, decay=0.1, max_episodes=100):
    # dimensionalities of state and action spaces
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # initialise target policy
    pi = np.tile(1/num_actions, (num_states, num_actions))

    # action value estimate
    Q = np.zeros((num_states, num_actions))

    # eligibility traces
    E = np.zeros((num_states, num_actions))

    # generate sample sequences (trajectories) for a number of episodes
    for _ in range(max_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False

        # select the first action by following the target policy
        action = rng.choice(num_actions, p=pi[state, :])

        while not (terminated or truncated):
            # send the action to the environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            # select next action from next state by following the target policy
            next_action = rng.choice(num_actions, p=pi[next_state, :])

            # calculate error
            delta = reward + gamma*Q[next_state, next_action] - Q[state, action]

            # update action values and eligibility traces
            # note: do not overwrite the `action` and `state` variables; loop over `a` and `s` instead
            E[state, action] += 1
            for s in range(num_states):
                for a in range(num_actions):
                    Q[s, a] += alpha*delta*E[s, a]
                    E[s, a] *= gamma*decay

            state = next_state
            action = next_action

            # update target policy
            # note: do not overwrite the `state` variable; loop over `s` instead
            for s in range(num_states):
                # select best action with ties broken randomly
                j = rng.permutation(num_actions)
                best_action = j[np.argmax(Q[s, j])]

                # construct epsilon-greedy policy with respect to the action values
                pi[s, :] = epsilon/num_actions
                pi[s, best_action] = 1 - epsilon + epsilon/num_actions

    return Q, pi

# create undiscounted cliff-walking environment
env = gym.make('CliffWalking-v1', render_mode="rgb_array_list")
gamma = 1

# dimensionalities of state and action spaces
num_states = env.observation_space.n
num_actions = env.action_space.n

# find optimal policy
q_opt, pi_opt = SARSATraces(env, gamma, alpha=0.5, epsilon=0.2, decay=0.9, max_episodes=100)

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
max_steps = 100
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = rng.choice(num_actions, p=pi_opt[state, :])

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
# clip.write_videofile("cliff_walking_sarsa_traces.mp4")
