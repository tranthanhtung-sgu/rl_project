import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal
import numpy as np

class FeatureExtractor(nn.Module):
    """
    Feature extractor network that can be shared between Actor and Critic.
    Based on Practical 9 implementation with support for both image and vector inputs.
    """
    def __init__(self, input_dim, hidden_sizes=(128, 128)):
        super(FeatureExtractor, self).__init__()
        
        # Handle different input types (vector vs image)
        if isinstance(input_dim, tuple) and (len(input_dim) == 3 or len(input_dim) == 4):
            # Input is an image (for Atari environments)
            if len(input_dim) == 4:  # (frames, h, w, channels)
                frames, h, w, channels = input_dim
                c = frames  # Use number of frames as channels
            else:  # (c, h, w)
                c, h, w = input_dim
            
            # CNN for processing images - increased channels for GPU utilization
            self.features = nn.Sequential(
                nn.Conv2d(c, 64, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
            
            # Calculate the size of the flattened features
            self.feature_size = self._get_conv_output_size(input_dim)
        else:
            # Input is a vector (for Ant environment) - based on Practical 9
            if isinstance(input_dim, tuple):
                input_dim = np.prod(input_dim)
                
            # MLP for processing vector inputs - based on Practical 9
            layers = []
            layers.append(nn.Linear(input_dim, hidden_sizes[0]))
            layers.append(nn.ReLU())
            for i in range(len(hidden_sizes)-1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                layers.append(nn.ReLU())
            
            self.features = nn.Sequential(*layers)
            self.feature_size = hidden_sizes[-1]
    
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
        
        output_tensor = self.features(input_tensor)
        return int(np.prod(output_tensor.shape))
    
    def forward(self, state):
        """Forward pass through the feature extractor"""
        return self.features(state)


class PolicyNetwork(nn.Module):
    """
    Policy Network (Actor) for the A2C algorithm.
    Based on Practical 9 implementation with support for both discrete and continuous action spaces.
    """
    def __init__(self, feature_extractor, feature_size, action_dim, is_continuous=False):
        super(PolicyNetwork, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.is_continuous = is_continuous
        
        # Output layer depends on action space type - based on Practical 9
        if is_continuous:
            # For continuous actions, output mean and log_std
            self.mean = nn.Linear(feature_size, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # For discrete actions, output action logits (preferences/unnormalised log-probabilities)
            self.policy_head = nn.Linear(feature_size, action_dim)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: The input state tensor
            
        Returns:
            If discrete action space: action logits
            If continuous action space: mean and std for action distribution
        """
        # Extract features from state
        features = self.feature_extractor(state)
        
        if self.is_continuous:
            # For continuous actions, return mean and std
            mean = self.mean(features)
            std = torch.exp(self.log_std).expand_as(mean)
            return mean, std
        else:
            # For discrete actions, return action logits (based on Practical 9)
            return self.policy_head(features)


class ValueNetwork(nn.Module):
    """
    Value Network (Critic) for the A2C algorithm.
    Based on Practical 9 implementation - estimates the value function V(s).
    """
    def __init__(self, feature_extractor, feature_size):
        super(ValueNetwork, self).__init__()
        
        self.feature_extractor = feature_extractor
        # Output layer (there is only one unit representing state value) - based on Practical 9
        self.value_head = nn.Linear(feature_size, 1)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: The input state tensor
            
        Returns:
            Estimated state value V(s)
        """
        features = self.feature_extractor(state)
        return self.value_head(features).squeeze(-1)
    
    def update(self, inputs, targets):
        """
        Update network weights for given input(s) and target(s).
        Based on Practical 9 implementation.
        """
        # This method will be implemented in the agent class
        pass


class A2CAgent:
    """
    Advantage Actor-Critic (A2C) Agent implementation.
    Based on Practical 9 implementation with support for both discrete and continuous action spaces.
    """
    def __init__(self, state_dim, action_dim, is_continuous=False, 
                 actor_lr=0.001, critic_lr=0.001, gamma=0.99, 
                 entropy_coef=0.01, value_loss_coef=0.5, device="cpu"):
        """
        Initialize the A2C agent.
        
        Args:
            state_dim: Dimension of the state space (int or tuple)
            action_dim: Dimension of the action space
            is_continuous: Whether the action space is continuous
            actor_lr: Learning rate for the actor (policy network)
            critic_lr: Learning rate for the critic (value network)
            gamma: Discount factor
            entropy_coef: Coefficient for the entropy bonus
            value_loss_coef: Coefficient for the value loss
            device: Device to run the model on (cpu or cuda)
        """
        self.device = device
        self.gamma = gamma
        self.is_continuous = is_continuous
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        # Entropy decay schedule (simple linear/cosine hybrid)
        self.initial_entropy_coef = entropy_coef
        self.min_entropy_coef = entropy_coef * 0.1
        self.decay_steps = 200000  # tune per run length
        
        # Set default hidden sizes based on Practical 9 - increased for GPU utilization
        hidden_sizes = (512, 512, 256) if not is_continuous else (512, 512, 256)
        
        # Initialize shared feature extractor based on Practical 9
        self.feature_extractor = FeatureExtractor(state_dim, hidden_sizes).to(device)
        feature_size = self.feature_extractor.feature_size
        
        # Initialize policy network (actor)
        self.policy_network = PolicyNetwork(
            self.feature_extractor, feature_size, action_dim, is_continuous
        ).to(device)
        
        # Initialize value network (critic)
        self.value_network = ValueNetwork(
            self.feature_extractor, feature_size
        ).to(device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.policy_network.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.value_network.parameters(), lr=critic_lr)
        
        # Storage for trajectory data (mirrors Practical 9 pattern)
        self.saved_log_probs = []
        self.saved_values = []
        self.entropies = []
        self.rewards = []
        
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
    
    def select_action(self, state, stochastic=True, store=True):
        """
        Select an action based on the current policy and estimate its value.
        Based on Practical 9 implementation.
        
        Args:
            state: The current state
            stochastic: Whether to sample stochastically or deterministically
            
        Returns:
            Selected action
        """
        # Preprocess state
        state = self.preprocess_state(state)
        
        # Estimate the state value (keep gradients for critic update)
        value = self.value_network(state)
        if value.dim() > 0:
            value = value.squeeze(0)
        if store:
            self.saved_values.append(value)
        
        if self.is_continuous:
            # For continuous actions
            mean, std = self.policy_network(state)
            mean = mean.squeeze(0)
            std = std.squeeze(0)
            covariance = torch.diag(std.pow(2))
            distribution = MultivariateNormal(mean, covariance)
            if stochastic:
                action = distribution.sample()
            else:
                action = mean
            log_prob = distribution.log_prob(action)
            entropy = distribution.entropy()
            
            # Convert to numpy for environment
            action_np = action.cpu().detach().numpy().flatten()
            
            # Store log probability and entropy for update
            if store:
                self.saved_log_probs.append(log_prob)
                self.entropies.append(entropy)
            
            return action_np
        else:
            # For discrete actions - based on Practical 9
            logits = self.policy_network(state).squeeze(0)
            distribution = Categorical(logits=logits)
            if stochastic:
                # sample action using action probabilities
                action = distribution.sample()
            else:
                # select action with the highest probability
                action = distribution.probs.argmax()
            
            log_prob = distribution.log_prob(action)
            entropy = distribution.entropy()
            
            # Store log probability and entropy
            if store:
                self.saved_log_probs.append(log_prob)
                self.entropies.append(entropy)
            
            return action.item()
    
    def store_reward(self, reward):
        """
        Store a reward from the environment.
        
        Args:
            reward: The reward received
        """
        self.rewards.append(reward)
    
    def compute_returns(self, next_value, done):
        """Compute discounted returns for the collected trajectory."""
        returns = []
        R = 0.0 if done else next_value.item()
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)
    
    def update_policy(self, next_state=None, done=True):
        """
        Update the policy and value networks using the A2C algorithm.
        Based on Practical 9 implementation.
        
        Args:
            next_state: The next state (used for bootstrapping if episode is not done)
            done: Whether the episode is done
            
        Returns:
            Dictionary with loss information
        """
        # If we have no trajectory data, return
        if len(self.rewards) == 0:
            return {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0, 'total_loss': 0}
        
        # Calculate next value for bootstrapping
        next_value = torch.zeros(1, device=self.device)
        if not done and next_state is not None:
            next_state = self.preprocess_state(next_state)
            with torch.no_grad():
                next_value = self.value_network(next_state)
                if next_value.dim() > 0:
                    next_value = next_value.squeeze(0)
        
        # Compute returns and advantages - based on Practical 9
        returns = self.compute_returns(next_value, done)
        values = torch.stack(self.saved_values)
        advantages = returns - values.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Set networks to training mode
        self.policy_network.train()
        self.value_network.train()
        
        # Calculate critic loss - based on Practical 9
        critic_loss = nn.MSELoss()(values, returns)
        
        # Stack stored log probabilities and entropies
        log_probs = torch.stack(self.saved_log_probs)
        entropies = torch.stack(self.entropies)
        
        # Actor loss and entropy bonus (Practical 9 pattern)
        actor_loss = -(log_probs * advantages).mean()
        entropy = entropies.mean()
        # Update entropy coefficient via schedule
        self._update_entropy_coef()
        
        total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
        
        # Update actor and critic networks
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        # Clear trajectory data
        self.saved_log_probs = []
        self.saved_values = []
        self.entropies = []
        self.rewards = []
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }

    def _update_entropy_coef(self):
        # Cosine decay to 10% of initial over decay_steps, then clamp
        t = getattr(self, 'steps_seen', 0)
        self.steps_seen = t + 1
        if self.decay_steps <= 0:
            return
        import math
        frac = min(1.0, self.steps_seen / float(self.decay_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * frac))  # 1->0
        target = self.min_entropy_coef + (self.initial_entropy_coef - self.min_entropy_coef) * cosine
        self.entropy_coef = max(self.min_entropy_coef, float(target))
    
    def save(self, path):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'feature_extractor': self.feature_extractor.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.value_network.load_state_dict(checkpoint['value_network'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
