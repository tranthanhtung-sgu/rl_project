#!/usr/bin/env python3

# COMP6008 Reinforcement Learning
# Computing@Curtin University
#
# Practical 9 - Policy Gradient and Actor-Critic
#
# Learning outcomes: after completing this practical you will be able to
# * Implement the policy gradient algorithm REINFORCE
# * Implement basic Actor-Critic algorithm with deep neural network approximation for both policy and value functions
# * Apply these algorithms to various environments with continuous state spaces in OpenAI Gym

# import necessary libraries
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from collections import deque
# from pyvirtualdisplay import Display
import moviepy as mpy

# create random number generator
rng = np.random.default_rng()

# create and start virtual display
# display = Display(backend='xvfb')
# display.start()

# Policy network for approximating policy function
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate):
        super().__init__()
        # create network layers
        layers = nn.ModuleList()

        # input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        # hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())

        # output layer (preferences/logits/unnormalised log-probabilities)
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # combine layers into feed-forward network
        self.net = nn.Sequential(*layers)

        # select optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def forward(self, x):
        # return output of policy network
        return self.net(x)

    def update(self, states, actions, returns):
        # update network weights for a given transition or trajectory
        self.optimizer.zero_grad()
        logits = self.net(states)
        dist = torch.distributions.Categorical(logits=logits)
        loss = torch.mean(-dist.log_prob(actions)*returns)
        loss.backward()
        self.optimizer.step()

# REINFORCE policy gradient with returns and without baseline; batch version (one sample trajectory)
class REINFORCE():
    def __init__(self, env, gamma, hidden_sizes=(32, 32), learning_rate=0.001):
        # check if the state space has correct type
        continuous = isinstance(env.observation_space, spaces.Box) and len(env.observation_space.shape) == 1
        assert continuous, 'Observation space must be continuous with shape (n,)'
        self.state_dims = env.observation_space.shape[0]

        # check if the action space has correct type
        assert isinstance(env.action_space, spaces.Discrete), 'Action space must be discrete'
        self.num_actions = env.action_space.n

        # create policy network
        self.policynet = PolicyNetwork(self.state_dims, hidden_sizes, self.num_actions, learning_rate)

        self.env = env
        self.gamma = gamma

    def policy(self, state, stochastic=True):
        # convert state to torch format
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float)

        # calculate action probabilities
        logits = self.policynet(state).detach()
        dist = torch.distributions.Categorical(logits=logits)
        if stochastic:
            # sample action using action probabilities
            return dist.sample().item()
        else:
            # select action with the highest probability
            # note: we ignore breaking ties randomly (low chance of happening)
            return dist.probs.argmax().item()

    def update(self, trajectory):
        # unpack trajectory
        states, actions, rewards = list(zip(*trajectory))

        # calculate return for each time step
        T = len(trajectory)
        returns = torch.zeros((T,))
        returns[-1] = rewards[-1]
        for t in reversed(range(T-1)):
            returns[t] = rewards[t] + self.gamma*returns[t+1]

        # convert list of states from trajectory to torch format
        states = torch.stack(states)
        actions = torch.tensor(actions)
        self.policynet.update(states, actions, returns)

    def train(self, max_episodes, stop_criterion, criterion_episodes):
        # train the agent for a number of episodes
        num_steps = 0
        episode_rewards = []
        for episode in range(max_episodes):
            state, _ = env.reset()

            # convert state to torch format
            state = torch.tensor(state, dtype=torch.float)
            terminated = False
            truncated = False
            episode_rewards.append(0)
            rewards = []
            trajectory = []

            # generate trajectory
            while not (terminated or truncated):
                # select action by following policy
                action = self.policy(state)

                # send the action to the environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                rewards.append(reward)
                episode_rewards[-1] += reward

                # add transition to trajectory
                trajectory.append((state, action, reward))

                # convert next state to torch format
                next_state = torch.tensor(next_state, dtype=torch.float)

                state = next_state
                num_steps += 1

            # update policy network
            self.update(trajectory)

            print(f'\rEpisode {episode+1} done: steps = {num_steps}, '
                  f'rewards = {episode_rewards[episode]}     ', end='')

            if episode >= criterion_episodes-1 and stop_criterion(episode_rewards[-criterion_episodes:]):
                print(f'\nStopping criterion satisfied after {episode} episodes')
                break

        # plot rewards received during training
        plt.figure(dpi=100)
        plt.plot(range(1, len(episode_rewards)+1), episode_rewards, label=f'Rewards')

        plt.xlabel('Episodes')
        plt.ylabel('Rewards per episode')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig('reinforce_training_rewards.png')
        plt.close()

    def save(self, path):
        # save network weights to a file
        torch.save(self.policynet.state_dict(), path)

    def load(self, path):
        # load network weights from a file
        self.policynet.load_state_dict(torch.load(path))

# Applying policy gradient to OpenAI Gym environments
# Cart Pole
env = gym.make('CartPole-v1', render_mode="rgb_array_list")

gamma = 0.99
hidden_sizes = (64, 64)
learning_rate = 0.001
max_episodes = 200
max_steps = 500
criterion_episodes = 5

agent = REINFORCE(env, gamma=gamma, hidden_sizes=hidden_sizes, learning_rate=learning_rate)

#agent.load('cartpole.64x64.REINFORCE.pt')
agent.train(max_episodes, lambda x : min(x) >= 400, criterion_episodes)

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
total_reward = 0
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = agent.policy(state, stochastic=False)

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1

print(f'Reward: {total_reward}')

# store RGB frames for the entire episode
frames = env.render()

# close the environment
env.close()

# create and play video clip using the frames and given fps
clip = mpy.ImageSequenceClip(frames, fps=50)
# clip.write_videofile("cartpole_reinforce.mp4")

#agent.save('cartpole.64x64.REINFORCE.pt')

# Mountain Car
env = gym.make('MountainCar-v0', render_mode="rgb_array_list")

gamma = 0.99
hidden_sizes = (64, 64)
learning_rate = 0.001
max_episodes = 20
max_steps = 200
criterion_episodes = 5

agent = REINFORCE(env, gamma=gamma, hidden_sizes=hidden_sizes, learning_rate=learning_rate)

#agent.load('mountaincar.64x64.REINFORCE.pt')
agent.train(max_episodes, lambda x : min(x) >= -150, criterion_episodes)

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
total_reward = 0
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = agent.policy(state, stochastic=False)

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1

print(f'Reward: {total_reward}')

# store RGB frames for the entire episode
frames = env.render()

# close the environment
env.close()

# create and play video clip using the frames and given fps
clip = mpy.ImageSequenceClip(frames, fps=50)
# clip.write_videofile("mountaincar_reinforce.mp4")

#agent.save('mountaincar.64x64.REINFORCE.pt')

# Lunar Lander
env = gym.make('LunarLander-v3', render_mode="rgb_array_list")

gamma = 0.99
hidden_sizes = (128, 128)
learning_rate = 0.001
max_episodes = 300
max_steps = 2000
criterion_episodes = 5

agent = REINFORCE(env, gamma=gamma, hidden_sizes=hidden_sizes, learning_rate=learning_rate)

#agent.load('lunarlander.128x128.REINFORCE.pt')
agent.train(max_episodes, lambda x : min(x) >= 200, criterion_episodes)

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
total_reward = 0
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = agent.policy(state, stochastic=False)

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1

print(f'Reward: {total_reward}')

# store RGB frames for the entire episode
frames = env.render()

# close the environment
env.close()

# create and play video clip using the frames and given fps
clip = mpy.ImageSequenceClip(frames, fps=50)
# clip.write_videofile("lunarlander_reinforce.mp4")

#agent.save('lunarlander.128x128.REINFORCE.pt')

# Exercise 1: Reward-to-go
# REINFORCE policy gradient with reward-to-go and without baseline; batch version (one sample trajectory)
class REINFORCE():
    def __init__(self, env, gamma, hidden_sizes=(32, 32), learning_rate=0.001):
        # check if the state space has correct type
        continuous = isinstance(env.observation_space, spaces.Box) and len(env.observation_space.shape) == 1
        assert continuous, 'Observation space must be continuous with shape (n,)'
        self.state_dims = env.observation_space.shape[0]

        # check if the action space has correct type
        assert isinstance(env.action_space, spaces.Discrete), 'Action space must be discrete'
        self.num_actions = env.action_space.n

        # create policy network
        self.policynet = PolicyNetwork(self.state_dims, hidden_sizes, self.num_actions, learning_rate)

        self.env = env
        self.gamma = gamma

    def policy(self, state, stochastic=True):
        # convert state to torch format
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float)

        # calculate action probabilities
        logits = self.policynet(state).detach()
        dist = torch.distributions.Categorical(logits=logits)
        if stochastic:
            # sample action using action probabilities
            return dist.sample().item()
        else:
            # select action with the highest probability
            # note: we ignore breaking ties randomly (low chance of happening)
            return dist.probs.argmax().item()

    def update(self, trajectory):
        # unpack trajectory
        states, actions, rewards = list(zip(*trajectory))

        # calculate return for each time step
        T = len(trajectory)
        returns = torch.zeros((T,))
        returns[-1] = rewards[-1]
        for t in reversed(range(T-1)):
            returns[t] = rewards[t] + self.gamma*returns[t+1]

        # reward-to-go
        gammas = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        for t in range(T):
            returns[t] = gammas[t] * returns[t]

        # convert list of states from trajectory to torch format
        states = torch.stack(states)
        actions = torch.tensor(actions)
        self.policynet.update(states, actions, returns)

    def train(self, max_episodes, stop_criterion, criterion_episodes):
        # train the agent for a number of episodes
        num_steps = 0
        episode_rewards = []
        for episode in range(max_episodes):
            state, _ = env.reset()

            # convert state to torch format
            state = torch.tensor(state, dtype=torch.float)
            terminated = False
            truncated = False
            episode_rewards.append(0)
            rewards = []
            trajectory = []

            # generate trajectory
            while not (terminated or truncated):
                # select action by following policy
                action = self.policy(state)

                # send the action to the environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                rewards.append(reward)
                episode_rewards[-1] += reward

                # add transition to trajectory
                trajectory.append((state, action, reward))

                # convert next state to torch format
                next_state = torch.tensor(next_state, dtype=torch.float)

                state = next_state
                num_steps += 1

            # update policy network
            self.update(trajectory)

            print(f'\rEpisode {episode+1} done: steps = {num_steps}, '
                  f'rewards = {episode_rewards[episode]}     ', end='')

            if episode >= criterion_episodes-1 and stop_criterion(episode_rewards[-criterion_episodes:]):
                print(f'\nStopping criterion satisfied after {episode} episodes')
                break

        # plot rewards received during training
        plt.figure(dpi=100)
        plt.plot(range(1, len(episode_rewards)+1), episode_rewards, label=f'Rewards')

        plt.xlabel('Episodes')
        plt.ylabel('Rewards per episode')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig('reinforce_reward_to_go_training_rewards.png')
        plt.close()

    def save(self, path):
        # save network weights to a file
        torch.save(self.policynet.state_dict(), path)

    def load(self, path):
        # load network weights from a file
        self.policynet.load_state_dict(torch.load(path))

# Value network for approximating value function
class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, learning_rate):
        super().__init__()
        # create network layers
        layers = nn.ModuleList()

        # input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        # hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())

        # output layer (there is only one unit representing state value)
        layers.append(nn.Linear(hidden_sizes[-1], 1))

        # combine layers into feed-forward network
        self.net = nn.Sequential(*layers)

        # select loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def forward(self, x):
        # return output of value network for the input x
        return self.net(x)

    def update(self, inputs, targets):
        # update network weights for given input(s) and target(s)
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

# Actor-Critic algorithm with one-step TD target
class ActorCritic():
    def __init__(self, env, gamma, hidden_sizes=(32, 32), lr_policy=0.001, lr_value=0.001):
        # check if the state space has correct type
        continuous = isinstance(env.observation_space, spaces.Box) and len(env.observation_space.shape) == 1
        assert continuous, 'Observation space must be continuous with shape (n,)'
        self.state_dims = env.observation_space.shape[0]

        # check if the action space has correct type
        assert isinstance(env.action_space, spaces.Discrete), 'Action space must be discrete'
        self.num_actions = env.action_space.n

        # create policy network
        self.policynet = PolicyNetwork(self.state_dims, hidden_sizes, self.num_actions, lr_policy)

        # create value network
        self.valuenet = ValueNetwork(self.state_dims, hidden_sizes, lr_value)

        self.env = env
        self.gamma = gamma

    def policy(self, state, stochastic=True):
        # convert state to torch format
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float)

        # calculate action probabilities
        logits = self.policynet(state).detach()
        dist = torch.distributions.Categorical(logits=logits)
        if stochastic:
            # sample action using action probabilities
            return dist.sample().item()
        else:
            # select action with the highest probability
            # note: we ignore breaking ties randomly (low chance of happening)
            return dist.probs.argmax().item()

    def update(self, state, action, reward, next_state, terminated):
        # calculate TD target for value network update
        if terminated:
            target = reward
        else:
            target = reward + self.gamma*self.valuenet(next_state).detach()

        # convert target to torch format
        target = torch.tensor([target], dtype=torch.float)

        # calculate TD error for policy network update (equal to the action advantage)
        delta = target - self.valuenet(state).detach()

        # update networks
        action = torch.tensor(action)
        self.policynet.update(state, action, delta)
        self.valuenet.update(state, target)

    def train(self, max_episodes, stop_criterion, criterion_episodes):
        # train the agent for a number of episodes
        num_steps = 0
        episode_rewards = []
        for episode in range(max_episodes):
            state, _ = env.reset()

            # convert state to torch format
            state = torch.tensor(state, dtype=torch.float)
            terminated = False
            truncated = False
            episode_rewards.append(0)
            while not (terminated or truncated):
                # select action by following policy
                action = self.policy(state)

                # send the action to the environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_rewards[-1] += reward

                # convert next state to torch format
                next_state = torch.tensor(next_state, dtype=torch.float)

                # update policy and value networks
                self.update(state, action, reward, next_state, terminated)

                state = next_state
                num_steps += 1

            print(f'\rEpisode {episode+1} done: steps = {num_steps}, '
                  f'rewards = {episode_rewards[episode]}     ', end='')

            if episode >= criterion_episodes-1 and stop_criterion(episode_rewards[-criterion_episodes:]):
                print(f'\nStopping criterion satisfied after {episode} episodes')
                break

        # plot rewards received during training
        plt.figure(dpi=100)
        plt.plot(range(1, len(episode_rewards)+1), episode_rewards, label=f'Rewards')

        plt.xlabel('Episodes')
        plt.ylabel('Rewards per episode')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig('actor_critic_training_rewards.png')
        plt.close()

    def save(self, path):
        # save network weights to a file
        torch.save({'policy': self.policynet.state_dict(),
                    'value': self.valuenet.state_dict()}, path)

    def load(self, path):
        # load network weights from a file
        networks = torch.load(path)
        self.policynet.load_state_dict(networks['policy'])
        self.valuenet.load_state_dict(networks['value'])

# Applying Actor-Critic to OpenAI Gym environments
# Cart Pole
env = gym.make('CartPole-v1', render_mode="rgb_array_list")

gamma = 0.99
hidden_sizes = (64, 64)
lr_policy = 0.001
lr_value = 0.005
max_episodes = 200
max_steps = 500
criterion_episodes = 5

agent = ActorCritic(env, gamma=gamma, hidden_sizes=hidden_sizes, lr_policy=lr_policy, lr_value=lr_value)

#agent.load('cartpole.64x64.AC.pt')
agent.train(max_episodes, lambda x : min(x) >= 400, criterion_episodes)

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
total_reward = 0
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = agent.policy(state, stochastic=False)

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1

print(f'Reward: {total_reward}')

# store RGB frames for the entire episode
frames = env.render()

# close the environment
env.close()

# create and play video clip using the frames and given fps
clip = mpy.ImageSequenceClip(frames, fps=50)
# clip.write_videofile("cartpole_actor_critic.mp4")

#agent.save('cartpole.64x64.AC.pt')

# Mountain Car
env = gym.make('MountainCar-v0', render_mode="rgb_array_list")

gamma = 0.99
hidden_sizes = (64, 64)
lr_policy = 0.001
lr_value = 0.005
max_episodes = 20
max_steps = 200
criterion_episodes = 5

agent = ActorCritic(env, gamma=gamma, hidden_sizes=hidden_sizes, lr_policy=lr_policy, lr_value=lr_value)

#agent.load('mountaincar.64x64.AC.pt')
agent.train(max_episodes, lambda x : min(x) >= -150, criterion_episodes)

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
total_reward = 0
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = agent.policy(state, stochastic=False)

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1

print(f'Reward: {total_reward}')

# store RGB frames for the entire episode
frames = env.render()

# close the environment
env.close()

# create and play video clip using the frames and given fps
clip = mpy.ImageSequenceClip(frames, fps=50)
# clip.write_videofile("mountaincar_actor_critic.mp4")

#agent.save('mountaincar.64x64.AC.pt')

# Lunar Lander
env = gym.make('LunarLander-v3', render_mode="rgb_array_list")

gamma = 0.99
hidden_sizes = (128, 128)
lr_policy = 0.001
lr_value = 0.005
max_episodes = 100
max_steps = 5000
criterion_episodes = 5

agent = ActorCritic(env, gamma=gamma, hidden_sizes=hidden_sizes, lr_policy=lr_policy, lr_value=lr_value)

#agent.load('lunarlander.128x128.AC.pt')
agent.train(max_episodes, lambda x : min(x) >= 200, criterion_episodes)

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
total_reward = 0
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = agent.policy(state, stochastic=False)

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1

print(f'Reward: {total_reward}')

# store RGB frames for the entire episode
frames = env.render()

# close the environment
env.close()

# create and play video clip using the frames and given fps
clip = mpy.ImageSequenceClip(frames, fps=50)
# clip.write_videofile("lunarlander_actor_critic.mp4")

#agent.save('lunarlander.128x128.AC.pt')
