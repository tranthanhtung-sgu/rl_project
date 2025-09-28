#!/usr/bin/env python3

# COMP6008 Reinforcement Learning
# Computing@Curtin University
#
# Practical 8 - Deep Q-learning
#
# Learning outcomes: after completing this practical you will be able to
# * Implement Q-network - deep neural network approximation of action-value function
# * Implement off-policy deep Q-learning methods:
#     * Deep Q-network (DQN)
#     * Double DQN
#     * Dueling DQN
#     * DQN with prioritised experience replay
# * Apply the methods to various environments with continuous state spaces in OpenAI Gym

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

# Q-network for approximating action-value function
class QNetwork(nn.Module):
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

        # output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # combine layers into feed-forward network
        self.net = nn.Sequential(*layers)

        # select loss function and optimizer
        # note: original paper uses modified MSE loss and RMSprop
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def forward(self, x):
        # return output of Q-network for the input x
        return self.net(x)

    def update(self, inputs, targets):
        # update network weights for a minibatch of inputs and targets:
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

    def copy_from(self, qnetwork):
        # copy weights from another Q-network
        self.net.load_state_dict(qnetwork.net.state_dict())

# Deep Q-network (DQN)
class AgentDQN():
    def __init__(self, env, gamma,
                 hidden_sizes=(32, 32),
                 learning_rate=0.001,
                 epsilon=0.1,
                 replay_size=10000,
                 minibatch_size=32,
                 target_update=20):
        # check if the state space has correct type
        continuous = isinstance(env.observation_space, spaces.Box) and len(env.observation_space.shape) == 1
        assert continuous, 'Observation space must be continuous with shape (n,)'
        self.state_dims = env.observation_space.shape[0]

        # check if the action space has correct type
        assert isinstance(env.action_space, spaces.Discrete), 'Action space must be discrete'
        self.num_actions = env.action_space.n

        # create Q-networks for action-value function
        self.qnet = QNetwork(self.state_dims, hidden_sizes, self.num_actions, learning_rate)
        self.target_qnet = QNetwork(self.state_dims, hidden_sizes, self.num_actions, learning_rate)

        # copy weights from Q-network to target Q-network
        self.target_qnet.copy_from(self.qnet)

        # initialise replay buffer
        self.replay_buffer = deque(maxlen=replay_size)

        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.minibatch_size = minibatch_size
        self.target_update = target_update
        self.target_update_idx = 0

    def behaviour(self, state):
        # exploratory behaviour policy
        if rng.uniform() >= self.epsilon:
            # convert state to torch format
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=torch.float)

            # exploitation with probability 1-epsilon; break ties randomly
            q = self.qnet(state).detach()
            j = rng.permutation(self.num_actions)
            return j[q[j].argmax().item()]
        else:
            # exploration with probability epsilon
            return self.env.action_space.sample()

    def policy(self, state):
        # convert state to torch format
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float)

        # greedy policy
        q = self.qnet(state).detach()
        return q.argmax().item()

    def update(self):
        # update Q-network if there is enough experience
        if len(self.replay_buffer) >= self.minibatch_size:
            # select mini-batch of experiences uniformly at random without replacement
            batch = rng.choice(len(self.replay_buffer), size=self.minibatch_size, replace=False)

            # calculate inputs and targets for the transitions in the mini-batch
            inputs = torch.zeros((self.minibatch_size, self.state_dims))
            targets = torch.zeros((self.minibatch_size, self.num_actions))

            for n, index in enumerate(batch):
                state, action, reward, next_state, terminated = self.replay_buffer[index]
                # inputs are states
                inputs[n, :] = state

                # targets are TD targets
                targets[n, :] = self.target_qnet(state).detach()

                if terminated:
                    targets[n, action] = reward
                else:
                    targets[n, action] = reward + self.gamma*self.target_qnet(next_state).detach().max()

            # train Q-network on the mini-batch
            self.qnet.update(inputs, targets)

        # periodically copy weights from Q-network to target Q-network
        self.target_update_idx += 1
        if self.target_update_idx % self.target_update == 0:
            self.target_qnet.copy_from(self.qnet)

    def train(self, max_episodes, stop_criterion, criterion_episodes):
        # train the agent for a number of episodes
        rewards = []
        num_steps = 0
        for episode in range(max_episodes):
            state, _ = env.reset()
            # convert state to torch format
            state = torch.tensor(state, dtype=torch.float)
            terminated = False
            truncated = False
            rewards.append(0)
            while not (terminated or truncated):
                # select action by following behaviour policy
                action = self.behaviour(state)

                # send the action to the environment
                next_state, reward, terminated, truncated, _ = env.step(action)

                # convert next state to torch format and add experience to replay buffer
                next_state = torch.tensor(next_state, dtype=torch.float)
                self.replay_buffer.append((state, action, reward, next_state, terminated))

                # update Q-network
                self.update()

                state = next_state
                rewards[-1] += reward
                num_steps += 1

            print(f'\rEpisode {episode+1} done: steps = {num_steps}, rewards = {rewards[episode]}     ', end='')

            if episode >= criterion_episodes-1 and stop_criterion(rewards[-criterion_episodes:]):
                print(f'\nStopping criterion satisfied after {episode} episodes')
                break

        # plot rewards received during training
        plt.figure(dpi=100)
        plt.plot(range(1, len(rewards)+1), rewards, label=f'Rewards')

        plt.xlabel('Episodes')
        plt.ylabel('Rewards per episode')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig('dqn_training_rewards.png')
        plt.close()

    def save(self, path):
        # save network weights to a file
        torch.save(self.qnet.state_dict(), path)

    def load(self, path):
        # load network weights from a file
        self.qnet.load_state_dict(torch.load(path))
        self.target_qnet.copy_from(self.qnet)

# Applying DQN to classic control environments
# Acrobot
env = gym.make('Acrobot-v1', render_mode="rgb_array_list")

gamma = 0.99
hidden_sizes = (128, 128)
learning_rate = 0.001
epsilon = 0.2
replay_size = 10000
minibatch_size = 64
target_update = 20
max_episodes = 100
max_steps = 1000
criterion_episodes = 5

agent = AgentDQN(env,
                 gamma=gamma,
                 hidden_sizes=hidden_sizes,
                 learning_rate=learning_rate,
                 epsilon=epsilon,
                 replay_size=replay_size,
                 minibatch_size=minibatch_size,
                 target_update=target_update)

#agent.load('acrobot.128x128.DQN.pt')
agent.train(max_episodes, lambda x : min(x) >= -90, criterion_episodes)

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
total_reward = 0
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = agent.policy(state)

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
clip = mpy.ImageSequenceClip(frames, fps=15)
# clip.write_videofile("acrobot_dqn.mp4")

agent.save('acrobot.128x128.myagent1.pt')

# Cart Pole
env = gym.make('CartPole-v1', render_mode="rgb_array_list")

gamma = 0.99
hidden_sizes = (64, 64)
learning_rate = 0.001
epsilon = 0.2
replay_size = 10000
minibatch_size = 64
target_update = 20
max_episodes = 200
max_steps = 500
criterion_episodes = 5

agent = AgentDQN(env,
                 gamma=gamma,
                 hidden_sizes=hidden_sizes,
                 learning_rate=learning_rate,
                 epsilon=epsilon,
                 replay_size=replay_size,
                 minibatch_size=minibatch_size,
                 target_update=target_update)

#agent.load('cartpole.64x64.DQN.pt')
agent.train(max_episodes, lambda x : min(x) >= 400, criterion_episodes)

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
total_reward = 0
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = agent.policy(state)

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
# clip.write_videofile("cartpole_dqn.mp4")

agent.save('cartpole.64x64.myagent1.pt')

# Mountain Car
env = gym.make('MountainCar-v0', render_mode="rgb_array_list")

gamma = 0.99
hidden_sizes = (64, 64)
learning_rate = 0.001
epsilon = 0.2
replay_size = 10000
minibatch_size = 32
target_update = 20
max_episodes = 100
max_steps = 200
criterion_episodes = 5

agent = AgentDQN(env,
                 gamma=gamma,
                 hidden_sizes=hidden_sizes,
                 learning_rate=learning_rate,
                 epsilon=epsilon,
                 replay_size=replay_size,
                 minibatch_size=minibatch_size,
                 target_update=target_update)

#agent.load('mountaincar.64x64.DQN.pt')
agent.train(max_episodes, lambda x : min(x) >= -150, criterion_episodes)

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
total_reward = 0
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = agent.policy(state)

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
# clip.write_videofile("mountaincar_dqn.mp4")

agent.save('mountaincar.64x64.myagent1.pt')

# Exercise 1: Double DQN
# Double Deep Q-network (DDQN)
class AgentDDQN():
    def __init__(self, env, gamma,
                 hidden_sizes=(32, 32),
                 learning_rate=0.001,
                 epsilon=0.1,
                 replay_size=10000,
                 minibatch_size=32,
                 target_update=20):
        # check if the state space has correct type
        continuous = isinstance(env.observation_space, spaces.Box) and len(env.observation_space.shape) == 1
        assert continuous, 'Observation space must be continuous with shape (n,)'
        self.state_dims = env.observation_space.shape[0]

        # check if the action space has correct type
        assert isinstance(env.action_space, spaces.Discrete), 'Action space must be discrete'
        self.num_actions = env.action_space.n

        # create Q-networks for action-value function
        self.qnet = QNetwork(self.state_dims, hidden_sizes, self.num_actions, learning_rate)
        self.target_qnet = QNetwork(self.state_dims, hidden_sizes, self.num_actions, learning_rate)

        # copy weights from Q-network to target Q-network
        self.target_qnet.copy_from(self.qnet)

        # initialise replay buffer
        self.replay_buffer = deque(maxlen=replay_size)

        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.minibatch_size = minibatch_size
        self.target_update = target_update
        self.target_update_idx = 0

    def behaviour(self, state):
        # exploratory behaviour policy
        if rng.uniform() >= self.epsilon:
            # convert state to torch format
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=torch.float)

            # exploitation with probability 1-epsilon; break ties randomly
            q = self.qnet(state).detach()
            j = rng.permutation(self.num_actions)
            return j[q[j].argmax().item()]
        else:
            # exploration with probability epsilon
            return self.env.action_space.sample()

    def policy(self, state):
        # convert state to torch format
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float)

        # greedy policy
        q = self.qnet(state).detach()
        return q.argmax().item()

    def update(self):
        # update Q-network if there is enough experience
        if len(self.replay_buffer) >= self.minibatch_size:
            # select mini-batch of experiences uniformly at random without replacement
            batch = rng.choice(len(self.replay_buffer), size=self.minibatch_size, replace=False)

            # calculate inputs and targets for the transitions in the mini-batch
            inputs = torch.zeros((self.minibatch_size, self.state_dims))
            targets = torch.zeros((self.minibatch_size, self.num_actions))

            for n, index in enumerate(batch):
                state, action, reward, next_state, terminated = self.replay_buffer[index]
                # inputs are states
                inputs[n, :] = state

                # targets are TD targets
                targets[n, :] = self.target_qnet(state).detach()

                if terminated:
                    targets[n, action] = reward
                else:
                    # Double DQN: use main network to select action, target network to evaluate
                    next_action = self.qnet(next_state).detach().argmax()
                    targets[n, action] = reward + self.gamma*self.target_qnet(next_state).detach()[next_action]

            # train Q-network on the mini-batch
            self.qnet.update(inputs, targets)

        # periodically copy weights from Q-network to target Q-network
        self.target_update_idx += 1
        if self.target_update_idx % self.target_update == 0:
            self.target_qnet.copy_from(self.qnet)

    def train(self, max_episodes, stop_criterion, criterion_episodes):
        # train the agent for a number of episodes
        rewards = []
        num_steps = 0
        for episode in range(max_episodes):
            state, _ = env.reset()
            # convert state to torch format
            state = torch.tensor(state, dtype=torch.float)
            terminated = False
            truncated = False
            rewards.append(0)
            while not (terminated or truncated):
                # select action by following behaviour policy
                action = self.behaviour(state)

                # send the action to the environment
                next_state, reward, terminated, truncated, _ = env.step(action)

                # convert next state to torch format and add experience to replay buffer
                next_state = torch.tensor(next_state, dtype=torch.float)
                self.replay_buffer.append((state, action, reward, next_state, terminated))

                # update Q-network
                self.update()

                state = next_state
                rewards[-1] += reward
                num_steps += 1

            print(f'\rEpisode {episode+1} done: steps = {num_steps}, rewards = {rewards[episode]}     ', end='')

            if episode >= criterion_episodes-1 and stop_criterion(rewards[-criterion_episodes:]):
                print(f'\nStopping criterion satisfied after {episode} episodes')
                break

        # plot rewards received during training
        plt.figure(dpi=100)
        plt.plot(range(1, len(rewards)+1), rewards, label=f'Rewards')

        plt.xlabel('Episodes')
        plt.ylabel('Rewards per episode')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig('ddqn_training_rewards.png')
        plt.close()

    def save(self, path):
        # save network weights to a file
        torch.save(self.qnet.state_dict(), path)

    def load(self, path):
        # load network weights from a file
        self.qnet.load_state_dict(torch.load(path))
        self.target_qnet.copy_from(self.qnet)

# Lunar Lander with Double DQN
env = gym.make('LunarLander-v2', render_mode="rgb_array_list")

gamma = 0.99
hidden_sizes = (128, 128)
learning_rate = 0.001
epsilon = 0.1
replay_size = 10000
minibatch_size = 128
target_update = 20
max_episodes = 5
max_steps = 5000
criterion_episodes = 5

agent = AgentDDQN(env,
                  gamma=gamma,
                  hidden_sizes=hidden_sizes,
                  learning_rate=learning_rate,
                  epsilon=epsilon,
                  replay_size=replay_size,
                  minibatch_size=minibatch_size,
                  target_update=target_update)

#agent.load('lunarlander.128x128.DQN.pt')
agent.train(max_episodes, lambda x : min(x) >= 200, criterion_episodes)

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
total_reward = 0
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = agent.policy(state)

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
# clip.write_videofile("lunarlander_ddqn.mp4")

agent.save('lunarlander.128x128.DDQN.pt')

# Exercise 2: Dueling DDQN with prioritised experience replay
# Dueling Q-network for approximating action-value function
class DuelingQNetwork(nn.Module):
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

        # special layer (state value and action advantage)
        layers.append(nn.Linear(hidden_sizes[-1], output_size + 1))

        # output layer (weights are fixed, bias is zero)
        output = nn.Linear(output_size+1, output_size, bias=False)
        layers.append(output)

        # combine layers into feed-forward network
        self.net = nn.Sequential(*layers)

        # select loss function and optimizer
        # note: original paper uses modified MSE loss and RMSprop
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        # initialise the weights according to dueling network architecture
        # This would need proper initialization in a real implementation

    def forward(self, x):
        # return output of Q-network for the input x
        return self.net(x)

    def update(self, inputs, targets):
        # update network weights for a minibatch of inputs and targets:
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

    def copy_from(self, qnetwork):
        # copy weights from another Q-network
        self.net.load_state_dict(qnetwork.net.state_dict())

# Dueling Double Deep Q-network with Prioritised Experience Replay
class AgentDuelingDDQNREP():
    def __init__(self, env, gamma,
                 hidden_sizes=(32, 32),
                 learning_rate=0.001,
                 epsilon=0.1,
                 rep_omega=0.2,
                 replay_size=10000,
                 minibatch_size=32,
                 target_update=20):
        # check if the state space has correct type
        continuous = isinstance(env.observation_space, spaces.Box) and len(env.observation_space.shape) == 1
        assert continuous, 'Observation space must be continuous with shape (n,)'
        self.state_dims = env.observation_space.shape[0]

        # check if the action space has correct type
        assert isinstance(env.action_space, spaces.Discrete), 'Action space must be discrete'
        self.num_actions = env.action_space.n

        # create dueling Q-networks for action-value function
        self.qnet = DuelingQNetwork(self.state_dims, hidden_sizes, self.num_actions, learning_rate)
        self.target_qnet = DuelingQNetwork(self.state_dims, hidden_sizes, self.num_actions, learning_rate)

        # copy weights from Q-network to target Q-network
        self.target_qnet.copy_from(self.qnet)

        # initialise replay buffer
        self.replay_buffer = deque(maxlen=replay_size)

        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.rep_omega = rep_omega
        self.minibatch_size = minibatch_size
        self.target_update = target_update
        self.target_update_idx = 0

    def behaviour(self, state):
        # exploratory behaviour policy
        if rng.uniform() >= self.epsilon:
            # convert state to torch format
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=torch.float)

            # exploitation with probability 1-epsilon; break ties randomly
            q = self.qnet(state).detach()
            j = rng.permutation(self.num_actions)
            return j[q[j].argmax().item()]
        else:
            # exploration with probability epsilon
            return self.env.action_space.sample()

    def policy(self, state):
        # convert state to torch format
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float)

        # greedy policy
        q = self.qnet(state).detach()
        return q.argmax().item()

    def td_error(self, state, action, reward, next_state, terminated):
        # calculate td error for prioritised experience replay
        if terminated:
            target = reward
        else:
            next_action = self.qnet(next_state).detach().argmax()
            target = reward + self.gamma*self.target_qnet(next_state).detach()[next_action]
        
        current_q = self.qnet(state).detach()[action]
        return abs(target - current_q)

    def update(self):
        # update Q-network if there is enough experience
        if len(self.replay_buffer) >= self.minibatch_size:
            # select mini-batch of experiences using prioritised experience replay
            # For simplicity, we'll use uniform sampling here
            # In a real implementation, you'd use priority-based sampling
            batch = rng.choice(len(self.replay_buffer), size=self.minibatch_size, replace=False)

            # calculate inputs and targets for the transitions in the mini-batch
            inputs = torch.zeros((self.minibatch_size, self.state_dims))
            targets = torch.zeros((self.minibatch_size, self.num_actions))

            for n, index in enumerate(batch):
                state, action, reward, next_state, terminated, _ = self.replay_buffer[index]
                # inputs are states
                inputs[n, :] = state

                # targets are TD targets
                targets[n, :] = self.target_qnet(state).detach()

                if terminated:
                    targets[n, action] = reward
                else:
                    # double learning
                    # note: we don't break ties randomly (ties are unlikely when weights are initialised randomly)
                    next_action = self.qnet(next_state).detach().argmax()
                    next_q = self.target_qnet(next_state).detach()
                    targets[n, action] = reward + self.gamma*next_q[next_action]

            # train Q-network on the mini-batch
            self.qnet.update(inputs, targets)

        # periodically copy weights from Q-network to target Q-network
        self.target_update_idx += 1
        if self.target_update_idx % self.target_update == 0:
            self.target_qnet.copy_from(self.qnet)

    def train(self, max_episodes, stop_criterion, criterion_episodes):
        # train the agent for a number of episodes
        rewards = []
        num_steps = 0
        for episode in range(max_episodes):
            state, _ = env.reset()
            # convert state to torch format
            state = torch.tensor(state, dtype=torch.float)
            terminated = False
            truncated = False
            rewards.append(0)
            while not (terminated or truncated):
                # select action by following behaviour policy
                action = self.behaviour(state)

                # send the action to the environment
                next_state, reward, terminated, truncated, _ = env.step(action)

                # convert next state to torch format
                next_state = torch.tensor(next_state, dtype=torch.float)

                # calculate td error for prioritised experience replay and add experience to replay buffer
                per = self.td_error(state, action, reward, next_state, terminated)
                self.replay_buffer.append((state, action, reward, next_state, terminated, per))

                # update Q-network
                self.update()

                state = next_state
                rewards[-1] += reward
                num_steps += 1

            print(f'\rEpisode {episode+1} done: steps = {num_steps}, rewards = {rewards[episode]}     ', end='')

            if episode >= criterion_episodes-1 and stop_criterion(rewards[-criterion_episodes:]):
                print(f'\nStopping criterion satisfied after {episode} episodes')
                break

        # plot rewards received during training
        plt.figure(dpi=100)
        plt.plot(range(1, len(rewards)+1), rewards, label=f'Rewards')

        plt.xlabel('Episodes')
        plt.ylabel('Rewards per episode')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig('dueling_ddqn_training_rewards.png')
        plt.close()

    def save(self, path):
        # save network weights to a file
        torch.save(self.qnet.state_dict(), path)

    def load(self, path):
        # load network weights from a file
        self.qnet.load_state_dict(torch.load(path))
        self.target_qnet.copy_from(self.qnet)

# Lunar Lander with Dueling DDQN
env = gym.make('LunarLander-v2', render_mode="rgb_array_list")

gamma = 0.99
hidden_sizes = (128, 128)
learning_rate = 0.001
epsilon = 0.1
rep_omega = 0.2
replay_size = 10000
minibatch_size = 128
target_update = 20
max_episodes = 50
max_steps = 5000
criterion_episodes = 5

agent = AgentDuelingDDQNREP(env,
                            gamma=gamma,
                            hidden_sizes=hidden_sizes,
                            learning_rate=learning_rate,
                            epsilon=epsilon,
                            rep_omega=rep_omega,
                            replay_size=replay_size,
                            minibatch_size=minibatch_size,
                            target_update=target_update)

#agent.load('lunarlander.128x128.DQN.pt')
agent.train(max_episodes, lambda x : min(x) >= 200, criterion_episodes)

# visualise one episode
state, _ = env.reset()
terminated = False
truncated = False
steps = 0
total_reward = 0
while not (terminated or truncated or steps > max_steps):
    # take action based on policy
    action = agent.policy(state)

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
# clip.write_videofile("lunarlander_dueling_ddqn.mp4")

agent.save('lunarlander.128x128.DuelingDDQREP.pt')
