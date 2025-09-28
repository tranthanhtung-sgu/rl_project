import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal
import numpy as np

class PolicyNetwork(nn.Module):
    """
    Policy Network for the REINFORCE algorithm.
    Based on Practical 9 implementation with support for both discrete and continuous action spaces.
    """
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate, is_continuous=False):
        super(PolicyNetwork, self).__init__()
        
        self.is_continuous = is_continuous
        
        # Handle different input types (vector vs image)
        if isinstance(input_size, tuple) and (len(input_size) == 3 or len(input_size) == 4):
            # Input is an image (for Atari environments)
            if len(input_size) == 4:  # (frames, h, w, channels)
                frames, h, w, channels = input_size
                c = frames  # Use number of frames as channels
            else:  # (c, h, w)
                c, h, w = input_size
            
            # CNN for processing images
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            
            # Calculate the size of the flattened features
            feature_size = self._get_conv_output_size(input_size)
            
            # MLP head for policy
            layers = []
            layers.append(nn.Linear(feature_size, hidden_sizes[0]))
            layers.append(nn.ReLU())
            for i in range(len(hidden_sizes)-1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                layers.append(nn.ReLU())
            
            self.feature_head = nn.Sequential(*layers)
            
            if is_continuous:
                # For continuous actions, output mean and log_std
                self.mean = nn.Linear(hidden_sizes[-1], output_size)
                self.log_std = nn.Parameter(torch.zeros(output_size))
            else:
                # For discrete actions, output action logits
                self.policy_head = nn.Linear(hidden_sizes[-1], output_size)
        else:
            # Input is a vector (for Ant environment)
            if isinstance(input_size, tuple):
                input_size = np.prod(input_size)
                
            # MLP for processing vector inputs - based on Practical 9
            layers = []
            layers.append(nn.Linear(input_size, hidden_sizes[0]))
            layers.append(nn.ReLU())
            for i in range(len(hidden_sizes)-1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                layers.append(nn.ReLU())
            
            self.feature_head = nn.Sequential(*layers)
            
            if is_continuous:
                # For continuous actions, output mean and log_std
                self.mean = nn.Linear(hidden_sizes[-1], output_size)
                self.log_std = nn.Parameter(torch.zeros(output_size))
            else:
                # For discrete actions, output action logits
                self.policy_head = nn.Linear(hidden_sizes[-1], output_size)
    
    def _get_conv_output_size(self, shape):
        """Calculate the output size of the CNN feature extractor"""
        batch_size = 1
        if len(shape) == 4:  # (frames, h, w, channels)
            frames, h, w, channels = shape
            # Create tensor of shape (batch_size, frames, h, w)
            input_tensor = torch.zeros(batch_size, frames, h, w)
        else:  # (c, h, w)
            c, h, w = shape
            input_tensor = torch.zeros(batch_size, c, h, w)
        
        output_tensor = self.feature_extractor(input_tensor)
        return int(np.prod(output_tensor.shape))
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: The input state tensor
            
        Returns:
            If discrete: Action logits (preferences/unnormalised log-probabilities)
            If continuous: mean and std for action distribution
        """
        # Extract features from state
        if hasattr(self, 'feature_extractor'):
            features = self.feature_extractor(state)
            features = self.feature_head(features)
        else:
            features = self.feature_head(state)
        
        if self.is_continuous:
            # For continuous actions, return mean and std
            mean = self.mean(features)
            std = torch.exp(self.log_std).expand_as(mean)
            return mean, std
        else:
            # For discrete actions, return action logits
            return self.policy_head(features)
    
    def update(self, states, actions, returns):
        """
        Update network weights for a given transition or trajectory.
        Based on Practical 9 implementation.
        """
        # This method will be implemented in the agent class
        pass


class ReinforceAgent:
    """
    REINFORCE Agent implementation.
    Based on Practical 9 implementation with support for both discrete and continuous action spaces.
    """
    def __init__(self, state_dim, action_dim, is_continuous=False, lr=0.001, gamma=0.99, device="cpu"):
        """
        Initialize the REINFORCE agent.
        
        Args:
            state_dim: Dimension of the state space (int or tuple)
            action_dim: Dimension of the action space
            is_continuous: Whether the action space is continuous
            lr: Learning rate
            gamma: Discount factor
            device: Device to run the model on (cpu or cuda)
        """
        self.device = device
        self.gamma = gamma
        self.is_continuous = is_continuous
        
        # Set default hidden sizes based on Practical 9
        hidden_sizes = (64, 64) if not is_continuous else (128, 128)
        
        # Initialize policy network based on Practical 9
        self.policy_network = PolicyNetwork(state_dim, hidden_sizes, action_dim, lr, is_continuous).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        
        # Storage for trajectory data
        self.saved_log_probs = []
        self.rewards = []
        self.actions = []
        self.states = []
        
    def preprocess_state(self, state):
        """
        Preprocess the state before passing it to the network.
        
        Args:
            state: The raw state from the environment
            
        Returns:
            Preprocessed state tensor
        """
        # Handle LazyFrames from FrameStack wrapper
        if hasattr(state, "_frames"):
            state = np.array(state)
            
        if isinstance(state, np.ndarray):
            # Convert numpy array to torch tensor
            state = torch.FloatTensor(state).to(self.device)
            
            # Handle different state shapes
            if len(state.shape) == 1:
                # Vector state (e.g., Ant)
                state = state.unsqueeze(0)  # Add batch dimension
            elif len(state.shape) == 4:
                # For Atari with FrameStack, shape is (4, 84, 84, 1)
                # Reshape to (1, 4, 84, 84)
                state = state.squeeze(-1).unsqueeze(0)
        
        return state
    
    def select_action(self, state, stochastic=True):
        """
        Select an action based on the current policy.
        Based on Practical 9 implementation.
        
        Args:
            state: The current state
            stochastic: Whether to sample stochastically or deterministically
            
        Returns:
            Selected action
        """
        # Preprocess state
        state = self.preprocess_state(state)
        
        # Store the state for trajectory
        self.states.append(state.detach())
        
        if self.is_continuous:
            # For continuous actions - use MultivariateNormal distribution
            mean, std = self.policy_network(state)
            # Create diagonal covariance matrix properly
            cov_matrix = torch.diag_embed(std)
            distribution = MultivariateNormal(mean, cov_matrix)
            
            if stochastic:
                action = distribution.sample()
            else:
                action = mean  # Deterministic action
            
            log_prob = distribution.log_prob(action)
            
            # Store log probability and action
            self.saved_log_probs.append(log_prob.detach())
            self.actions.append(action.detach())
            
            return action.cpu().detach().numpy().flatten()
        else:
            # For discrete actions - based on Practical 9
            logits = self.policy_network(state).detach()
            distribution = Categorical(logits=logits)
            if stochastic:
                # sample action using action probabilities
                action = distribution.sample()
            else:
                # select action with the highest probability
                # note: we ignore breaking ties randomly (low chance of happening)
                action = distribution.probs.argmax()
            
            log_prob = distribution.log_prob(action)
            
            # Store log probability (detached for test_agents.py)
            self.saved_log_probs.append(log_prob.detach())
            self.actions.append(action.detach())
            
            return action.item()
    
    def store_reward(self, reward):
        """
        Store a reward from the environment.
        
        Args:
            reward: The reward received
        """
        self.rewards.append(reward)
    
    def update_policy(self):
        """
        Update the policy network using the REINFORCE algorithm.
        Based on Practical 9 implementation.
        """
        # For test_agents.py, we'll just return 0 if there are no rewards
        if len(self.rewards) == 0:
            return 0.0
            
        # Calculate returns (discounted rewards) - based on Practical 9
        T = len(self.rewards)
        returns = torch.zeros((T,), device=self.device)
        returns[-1] = self.rewards[-1]
        for t in reversed(range(T-1)):
            returns[t] = self.rewards[t] + self.gamma * returns[t+1]
        
        # Convert list of states from trajectory to torch format
        # Remove the batch dimension from each state before stacking
        states = torch.stack([state.squeeze(0) for state in self.states])
        actions = torch.stack(self.actions)
        
        # Update policy network based on Practical 9
        self.optimizer.zero_grad()
        
        if self.is_continuous:
            # For continuous actions
            mean, std = self.policy_network(states)
            cov_matrix = torch.diag_embed(std)
            dist = torch.distributions.MultivariateNormal(mean, cov_matrix)
            loss = torch.mean(-dist.log_prob(actions) * returns)
        else:
            # For discrete actions
            logits = self.policy_network(states)
            dist = torch.distributions.Categorical(logits=logits)
            loss = torch.mean(-dist.log_prob(actions) * returns)
        
        loss.backward()
        self.optimizer.step()
        
        loss_value = loss.item()
        
        # Clear trajectory data
        self.saved_log_probs = []
        self.rewards = []
        self.actions = []
        self.states = []
        
        return loss_value
    
    def save(self, path):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
