#!/usr/bin/env python3

# COMP6008 Reinforcement Learning
# Computing@Curtin University
#
# Practical 3 - Markov Decision Process and Dynamic Programming
#
# Learning outcomes: after completing this practical you will be able to
# * Implement finite MDPs as custom OpenAI Gym environments
# * Evaluate a policy (prediction problem)
# * Find an optimal policy (control problem) using dynamic programming

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

    def __init__(self, alpha, beta, gamma, r_search, r_wait):
        super().__init__()

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

    # we will implement the reset and step function next week when we focus on learning instead of planning
    def reset(self):
        return False, {}

    def step(self, action):
        # return observation (state), reward, terminated, truncated, additional information
        return False, 0, False, False, {}

# MDP parameters
alpha = 0.9
beta = 0.4
gamma = 0.9
r_search = 2
r_wait = 0
state_names = ('low', 'high')
action_names = ('wait', 'search', 'recharge')

# create an instance of the Recycling Robot MDP
mdp = RecyclingRobot(alpha, beta, gamma, r_search, r_wait)

# define stochastic policy pi(a|s) = pi[s, a]
pi = np.zeros((mdp.num_states, mdp.num_actions))
pi[low, wait] = 0.2
pi[low, search] = 0.4
pi[low, recharge] = 0.4
pi[high, wait] = 0.5
pi[high, search] = 0.5

# iterative policy evaluation (prediction problem); in-place version
def PolicyEvaluation(mdp, pi, gamma, v=None, threshold=0.001, max_iters=1000):
    # if initial state value is not given, start from zero
    if v is None:
        v = np.zeros((mdp.num_states,))
    for i in range(max_iters):
        # difference in state values between last and current iterations
        delta = 0
        for state in range(mdp.num_states):
            v_old = v[state]
            # using Bellman equation as iterative update rule
            # note: the arguments are (s, a, s') but s is fixed and we don't have to expand along that dimension
            v[state] = np.sum(pi[state, :, np.newaxis] * mdp.p[state, :, :] 
                              * (mdp.r[state, :, :] + gamma*v[np.newaxis, :]))
            delta = max(delta, abs(v[state] - v_old))

        # check convergence
        if delta <= threshold:
            print(f'Policy evaluation converged in {i+1} iteration(s)')
            return v

    # if we ever reach this point, then the policy evaluation did not converge
    print(f'Policy evaluation did not converge within {max_iters} iterations: |v - v_old| = {delta} > {threshold}')
    return v

# evaluate given policy
v_pi = PolicyEvaluation(mdp, pi, gamma)
for n, v in enumerate(v_pi):
    print(f'v({state_names[n]}) = {v}')

# Exercise 1: Calculate action values using the state values
# calculate action values using the relationship formula between state and action values
q_pi = np.sum(mdp.p * (mdp.r + gamma*v_pi[np.newaxis, np.newaxis, :]), axis=2)
for n, qs in enumerate(q_pi):
    for k, qsa in enumerate(qs):
        print(f'q({state_names[n]}, {action_names[k]}) = {qsa}')

# Exercise 2: Define and evaluate two deterministic policies
# search when in high state, recharge when in low state
pi[low, wait] = 0
pi[low, search] = 0
pi[low, recharge] = 1
pi[high, wait] = 0
pi[high, search] = 1
pi[high, recharge] = 0

# evaluate given policy
v_pi = PolicyEvaluation(mdp, pi, gamma)
for n, v in enumerate(v_pi):
    print(f'v({state_names[n]}) = {v}')

# search when in high or low state
pi[low, wait] = 0
pi[low, search] = 1
pi[low, recharge] = 0
pi[high, wait] = 0
pi[high, search] = 1
pi[high, recharge] = 0

# evaluate given policy
v_pi = PolicyEvaluation(mdp, pi, gamma)
for n, v in enumerate(v_pi):
    print(f'v({state_names[n]}) = {v}')

# policy iteration (control problem); full policy evaluation and policy improvement
def PolicyIteration(mdp, pi, gamma, max_iters=20):
    v = np.zeros((mdp.num_states,))
    for i in range(max_iters):
        # policy evaluation
        v = PolicyEvaluation(mdp, pi, gamma, v)
        # policy improvement
        policy_stable = True
        for state in range(mdp.num_states):
            action_old = np.argmax(pi[state])
            # check if there is a different action with higher value
            # note: the arguments are (s, a, s') but s is fixed and we don't have to expand along that dimension
            qstate = np.sum(mdp.p[state, :, :] * (mdp.r[state, :, :] + gamma*v[np.newaxis, :]), axis=1)
            action_new = np.argmax(qstate)
            # make policy deterministic for state s using the best action
            pi[state, :] = 0
            pi[state, action_new] = 1
            if action_new != action_old:
                policy_stable = False

        # if there are no more changes to the policy, then the policy is optimal
        if policy_stable:
            print(f'Policy iteration converged in {i+1} iteration(s)')
            return v, pi

    # if we ever reach this point, then the policy iteration did not converge
    print(f'Policy iteration did not converge within {max_iters} iterations')
    return v, pi

# lazy policy
pi[low, wait] = 1
pi[low, search] = 0
pi[low, recharge] = 0
pi[high, wait] = 1
pi[high, search] = 0
pi[high, recharge] = 0

# find optimal policy
v_opt, pi_opt = PolicyIteration(mdp, pi, gamma)
for n, v in enumerate(v_opt):
    print(f'v({state_names[n]}) = {v}')
    print(f'pi({state_names[n]}) = {pi_opt[n]}')

# calculate action values using the relationship formula between state and action values
q_opt = np.sum(mdp.p * (mdp.r + gamma*v_opt[np.newaxis, np.newaxis, :]), axis=2)
for n, qs in enumerate(q_opt):
    for k, qsa in enumerate(qs):
        print(f'q({state_names[n]}, {action_names[k]}) = {qsa}')

# helper function that generates greedy policy from value function
def ValueToPolicy(mdp, gamma, v):
    pi = np.zeros((mdp.num_states, mdp.num_actions))
    for state in range(mdp.num_states):
        # calculate action values from the state values
        qstate = np.sum(mdp.p[state, :, :] * (mdp.r[state, :, :] + gamma*v[np.newaxis, :]), axis=1)
        action = np.argmax(qstate)
        pi[state, action] = 1
    return pi

# value iteration (control problem)
def ValueIteration(mdp, gamma, threshold=0.001, max_iters=1000):
    v = np.zeros((mdp.num_states,))
    for i in range(max_iters):
        # difference in state values between last and current iterations
        delta = 0
        for state in range(mdp.num_states):
            v_old = v[state]
            # using Bellman optimality equation as iterative update rule
            # note: the arguments are (s, a, s') but s is fixed and we don't have to expand along that dimension
            qstate = np.sum(mdp.p[state, :, :] * (mdp.r[state, :, :] + gamma*v[np.newaxis, :]), axis=1)
            v[state] = np.max(qstate)
            delta = max(delta, abs(v[state] - v_old))

        # check convergence
        if delta <= threshold:
            pi = ValueToPolicy(mdp, gamma, v)
            print(f'Value iteration converged in {i+1} iteration(s)')
            return v, pi

    # if we ever reach this point, then the value iteration did not converge
    print(f'Value iteration did not converge within {max_iters} iterations: |v - v_old| = {delta} > {threshold}')
    pi = ValueToPolicy(mdp, gamma, v)
    return v, pi

# find optimal policy
v_opt, pi_opt = ValueIteration(mdp, gamma)
for n, v in enumerate(v_opt):
    print(f'v({state_names[n]}) = {v}')
    print(f'pi({state_names[n]}) = {pi_opt[n]}')

# calculate action values using the relationship formula between state and action values
q_opt = np.sum(mdp.p * (mdp.r + gamma*v_opt[np.newaxis, np.newaxis, :]), axis=2)
for n, qs in enumerate(q_opt):
    for k, qsa in enumerate(qs):
        print(f'q({state_names[n]}, {action_names[k]}) = {qsa}')

# Exercise 3: Create a new instance of the Recycling Robot MDP with different parameters
# MDP parameters
alpha = 0.6
beta = 0.2
gamma = 0.99
r_search = 1
r_wait = -0.5

# create an instance of the Recycling Robot MDP
mdp = RecyclingRobot(alpha, beta, gamma, r_search, r_wait)

# create and evaluate always-search policy
pi[low, wait] = 0
pi[low, search] = 1
pi[low, recharge] = 0
pi[high, wait] = 0
pi[high, search] = 1
pi[high, recharge] = 0

# evaluate given policy
v_pi = PolicyEvaluation(mdp, pi, gamma)
for n, v in enumerate(v_pi):
    print(f'v({state_names[n]}) = {v}')

# calculate action values using the relationship formula between state and action values
q_pi = np.sum(mdp.p * (mdp.r + gamma*v_pi[np.newaxis, np.newaxis, :]), axis=2)
for n, qs in enumerate(q_pi):
    for k, qsa in enumerate(qs):
        print(f'q({state_names[n]}, {action_names[k]}) = {qsa}')

# find optimal policy
v_opt, pi_opt = ValueIteration(mdp, gamma)
for n, v in enumerate(v_opt):
    print(f'v({state_names[n]}) = {v}')
    print(f'pi({state_names[n]}) = {pi_opt[n]}')

# calculate action values using the relationship formula between state and action values
q_opt = np.sum(mdp.p * (mdp.r + gamma*v_opt[np.newaxis, np.newaxis, :]), axis=2)
for n, qs in enumerate(q_opt):
    for k, qsa in enumerate(qs):
        print(f'q({state_names[n]}, {action_names[k]}) = {qsa}')

# Exercise 4: Jack's Car Rental finite MDP
from scipy.stats import poisson

# Jack's Car Rental finite MDP
class JacksCarRental(gym.Env):

    def __init__(self,
                 gamma=0.9,
                 max_cars=20,
                 max_move=5,
                 credit=10,
                 cost=2,
                 expected_requests=(3, 4),
                 expected_returns=(3, 2)):

        super().__init__()

        # MDP parameters
        # note: there are from 0 to max_cars at each of the two locations
        self.num_states = (max_cars + 1)**2
        # note: agent can move from -max_move to +max_move cars from location 1 to location 2 overnight
        self.num_actions = 2*max_move + 1

        self.max_cars = max_cars

        # use OpenAI Gym spaces to be consistent with the API
        self.observation_space = spaces.Discrete(self.num_states)
        self.action_space = spaces.Discrete(self.num_actions)

        # state-transition probabilities p(s'|s,a) = p[s,a,s']
        self.p = np.zeros((self.num_states, self.num_actions, self.num_states))

        # reward function r(s,a,s') = r[s,a,s']
        self.r = np.zeros((self.num_states, self.num_actions, self.num_states))

        # note: the order of events is:
        # 1 environment emits state (number of cars at each location) at the end of the day
        # 2 agent takes action to move a number of cars between locations overnight
        # 3 environment forwards time by +1 day
        # 4 customers rent cars; only those cars that are at the location in the morning can be rented out
        # 5 customers return cars; those cars cannot be rented out on the same day
        # 6 any additional cars above max_cars limit are returned to the nationwide company

        # calculate probabilities for returns
        prob_returns1 = poisson.pmf(range(max_cars), expected_returns[0])
        prob_returns1 = np.append(prob_returns1, 1 - prob_returns1.sum())
        prob_returns2 = poisson.pmf(range(max_cars), expected_returns[1])
        prob_returns2 = np.append(prob_returns2, 1 - prob_returns2.sum())

        # calculate state-transition probability and reward for each state action pair
        for cars1 in range(max_cars+1):
            for cars2 in range(max_cars+1):
                for move in range(-max_move, max_move+1):
                    # convert subscripts to linear index (row-wise)
                    idx = cars1*(max_cars + 1) + cars2

                    # ignore impossible actions (moving more cars than there are at location etc.)
                    if move > cars1 or move > (max_cars - cars2) or move < -cars2 or move < -(max_cars - cars1):
                        continue

                    # convert the number of cars to be moved to action (from 0 to num_actions-1)
                    action = move + max_move

                    # calculate the cost of moving cars
                    reward = -cost*abs(move)

                    # calculate the number of cars available at each location after moving
                    cars1_avail = cars1 - move
                    cars2_avail = cars2 + move

                    # calculate probabilities for requests
                    prob_requests1 = poisson.pmf(range(cars1_avail), expected_requests[0])
                    prob_requests1 = np.append(prob_requests1, 1 - prob_requests1.sum())[::-1]
                    prob_requests2 = poisson.pmf(range(cars2_avail), expected_requests[1])
                    prob_requests2 = np.append(prob_requests2, 1 - prob_requests2.sum())[::-1]

                    # calculate probabilities of there being a number of cars after requests and returns,
                    # and the corresponding expected reward
                    cars1_mat = np.arange(max_cars+1)[:, np.newaxis] + np.arange(cars1_avail+1)[np.newaxis, :]
                    cars1_mat = np.minimum(cars1_mat, max_cars)
                    prob1_mat = np.outer(prob_returns1, prob_requests1)
                    prob1 = np.zeros((max_cars+1,))
                    reward1_mat = np.tile(credit*np.arange(cars1_avail, -1, -1), (max_cars+1, 1))
                    reward1 = np.zeros((max_cars+1,))
                    for c, p, r in zip(cars1_mat.reshape(-1), prob1_mat.reshape(-1), reward1_mat.reshape(-1)):
                        prob1[c] += p
                        reward1[c] += p*r
                    reward1 = np.divide(reward1, prob1, where=prob1>0)

                    cars2_mat = np.arange(max_cars+1)[:, np.newaxis] + np.arange(cars2_avail+1)[np.newaxis, :]
                    cars2_mat = np.minimum(cars2_mat, max_cars)
                    prob2_mat = np.outer(prob_returns2, prob_requests2)
                    prob2 = np.zeros((max_cars+1,))
                    reward2_mat = np.tile(credit*np.arange(cars2_avail, -1, -1), (max_cars+1, 1))
                    reward2 = np.zeros((max_cars+1,))
                    for c, p, r in zip(cars2_mat.reshape(-1), prob2_mat.reshape(-1), reward2_mat.reshape(-1)):
                        prob2[c] += p
                        reward2[c] += p*r
                    reward2 = np.divide(reward2, prob2, where=prob2>0)

                    self.p[idx, action, :] = np.outer(prob1, prob2).reshape(-1)
                    self.r[idx, action, :] = (reward + reward1[:, np.newaxis] + reward2[np.newaxis, :]).reshape(-1)

    def reset(self):
        return False, {}

    def step(self, action):
        # return observation (state), reward, terminated, truncated, additional information
        return False, 0, False, False, {}

# MDP parameters used for the value iteration and plotting
gamma = 0.9
max_cars = 20
max_move = 5

mdp2 = JacksCarRental(gamma, max_cars, max_move)

# find optimal policy and visualise the results
v_opt, pi_opt = ValueIteration(mdp2, gamma)
pi_opt = np.argmax(pi_opt, axis=1) - max_move
plt.figure(dpi=100)
plt.imshow(pi_opt.reshape((max_cars+1, max_cars+1)), origin='lower')
plt.colorbar()
plt.title(r'Optimal policy $\pi_*$')
plt.xlabel('#Cars at second location')
plt.ylabel('#Cars at first location')
plt.savefig('jacks_car_rental_policy.png')
plt.close()

plt.figure(dpi=100)
plt.imshow(v_opt.reshape((max_cars+1, max_cars+1)), origin='lower')
plt.colorbar()
plt.title(r'Optimal value $v_*$')
plt.xlabel('#Cars at second location')
plt.ylabel('#Cars at first location')
plt.savefig('jacks_car_rental_value.png')
plt.close()
