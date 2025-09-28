import numpy as np
import torch
from collections import deque
import random

class ReplayBuffer:
    """
    Replay Buffer for storing and sampling transitions.
    Used for off-policy algorithms like DDPG.
    """
    def __init__(self, capacity, state_dim, action_dim, device="cpu"):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum capacity of the buffer
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            device: Device to store tensors on (cpu or cuda)
        """
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        
        # Store dimensions for batch creation
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Keep track of current size
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Convert to numpy arrays if they are not already
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
            
        # Store as a tuple
        self.buffer.append((state, action, reward, next_state, done))
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as torch tensors
        """
        # Sample batch_size elements from the buffer
        batch = random.sample(self.buffer, min(batch_size, self.size))
        
        # Separate the tuple elements
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.array(next_states)
        dones = np.array(dones).reshape(-1, 1)
        
        # Convert to torch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer"""
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Replay Buffer for storing and sampling transitions.
    Uses importance sampling for more efficient learning.
    """
    def __init__(self, capacity, state_dim, action_dim, alpha=0.6, beta=0.4, device="cpu"):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum capacity of the buffer
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            alpha: Exponent for prioritization (0 = uniform sampling)
            beta: Exponent for importance sampling correction (1 = full correction)
            device: Device to store tensors on (cpu or cuda)
        """
        super().__init__(capacity, state_dim, action_dim, device)
        
        # Parameters for prioritized replay
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        self.epsilon = 1e-6  # Small constant to avoid zero priority
        
        # Initialize priorities with maximum priority for new experiences
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer with maximum priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Get the index where the experience will be stored
        idx = len(self.buffer) % self.capacity
        
        # Store the experience with maximum priority
        super().add(state, action, reward, next_state, done)
        self.priorities[idx] = self.max_priority
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer based on priorities.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(self.size, min(batch_size, self.size), replace=False, p=probabilities)
        
        # Get the experiences
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.array(next_states)
        dones = np.array(dones).reshape(-1, 1)
        weights = np.array(weights).reshape(-1, 1)
        
        # Convert to torch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Increment beta for annealing
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors.
        
        Args:
            indices: Indices of the experiences to update
            td_errors: TD errors for the experiences
        """
        for idx, td_error in zip(indices, td_errors):
            # Calculate priority as |td_error| + epsilon
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            
            # Update priority
            self.priorities[idx] = priority
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority)
