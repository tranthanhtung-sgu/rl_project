import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from utils.replay_buffer import ReplayBuffer

class Actor(nn.Module):
    """
    Actor network for DDPG.
    Based on Practical 8 DQN implementation adapted for continuous actions.
    Maps states to deterministic actions.
    """
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate, max_action=1.0):
        super(Actor, self).__init__()
        
        # Handle different input types (vector vs image)
        if isinstance(input_size, tuple) and (len(input_size) == 3 or len(input_size) == 4):
            # Input is an image
            if len(input_size) == 4:  # (frames, h, w, channels)
                frames, h, w, channels = input_size
                c = frames  # Use number of frames as channels
            else:  # (c, h, w)
                c, h, w = input_size
            
            # CNN for processing images - based on Practical 8
            self.features = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
            
            # Calculate the size of the flattened features
            feature_size = self._get_conv_output_size(input_size)
            
            # MLP head - based on Practical 8 structure
            layers = []
            layers.append(nn.Linear(feature_size, hidden_sizes[0]))
            layers.append(nn.ReLU())
            for i in range(len(hidden_sizes)-1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_sizes[-1], output_size))
            layers.append(nn.Tanh())  # Output in [-1, 1]
            
            self.head = nn.Sequential(*layers)
        else:
            # Input is a vector - based on Practical 8
            if isinstance(input_size, tuple):
                input_size = np.prod(input_size)
                
            # MLP for processing vector inputs - based on Practical 8
            layers = []
            layers.append(nn.Linear(input_size, hidden_sizes[0]))
            layers.append(nn.ReLU())
            for i in range(len(hidden_sizes)-1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_sizes[-1], output_size))
            layers.append(nn.Tanh())  # Output in [-1, 1]
            
            self.head = nn.Sequential(*layers)
        
        self.max_action = max_action
    
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
        
        if hasattr(self, 'feature_extractor'):
            output_tensor = self.feature_extractor(input_tensor)
        else:
            output_tensor = self.head(input_tensor)
        return int(np.prod(output_tensor.shape))
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: The input state tensor
            
        Returns:
            Action scaled to the action space
        """
        if hasattr(self, 'feature_extractor'):
            x = self.feature_extractor(state)
            action = self.head(x)
        else:
            action = self.head(state)
        # Scale from [-1, 1] to the action space
        return self.max_action * action


class Critic(nn.Module):
    """
    Critic network for DDPG.
    Based on Practical 8 DQN implementation adapted for continuous actions.
    Maps state-action pairs to Q-values.
    """
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate):
        super(Critic, self).__init__()
        
        # Handle different input types (vector vs image)
        if isinstance(input_size, tuple) and (len(input_size) == 3 or len(input_size) == 4):
            # Input is an image
            if len(input_size) == 4:  # (frames, h, w, channels)
                frames, h, w, channels = input_size
                c = frames  # Use number of frames as channels
            else:  # (c, h, w)
                c, h, w = input_size
            
            # CNN for processing images - based on Practical 8
            self.state_features = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
            
            # Calculate the size of the flattened features
            feature_size = self._get_conv_output_size(input_size)
            
            # Process state features and concatenate with action - based on Practical 8
            layers = []
            layers.append(nn.Linear(feature_size + output_size, hidden_sizes[0]))
            layers.append(nn.ReLU())
            for i in range(len(hidden_sizes)-1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_sizes[-1], 1))
            
            self.q1_head = nn.Sequential(*layers)
            
            # Second critic for TD3-style learning (optional)
            self.q2_head = nn.Sequential(*layers)
        else:
            # Input is a vector - based on Practical 8
            if isinstance(input_size, tuple):
                input_size = np.prod(input_size)
                
            # Process state - based on Practical 8
            self.state_features = nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0]),
                nn.ReLU()
            )
            
            feature_size = hidden_sizes[0]
            
            # Process state features and concatenate with action - based on Practical 8
            layers = []
            layers.append(nn.Linear(feature_size + output_size, hidden_sizes[0]))
            layers.append(nn.ReLU())
            for i in range(len(hidden_sizes)-1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_sizes[-1], 1))
            
            self.q1_head = nn.Sequential(*layers)
            
            # Second critic for TD3-style learning (optional)
            self.q2_head = nn.Sequential(*layers)
    
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
        
        output_tensor = self.state_features(input_tensor)
        return int(np.prod(output_tensor.shape))
    
    def forward(self, state, action):
        """
        Forward pass through the network.
        
        Args:
            state: The input state tensor
            action: The input action tensor
            
        Returns:
            Q-value for the state-action pair
        """
        state_features = self.state_features(state)
        x = torch.cat([state_features, action], dim=1)
        q1 = self.q1_head(x)
        q2 = self.q2_head(x)
        return q1, q2
    
    def q1(self, state, action):
        """Get Q1 value only (used for actor updates)"""
        state_features = self.state_features(state)
        x = torch.cat([state_features, action], dim=1)
        return self.q1_head(x)


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) Agent implementation.
    Based on Practical 8 DQN implementation adapted for continuous actions.
    Designed for continuous action spaces.
    """
    def __init__(self, state_dim, action_dim, max_action=1.0,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99,
                 tau=0.005, buffer_size=1000000, batch_size=256,
                 policy_noise=0.2, noise_clip=0.5, exploration_noise=0.1,
                 device="cpu"):
        """
        Initialize the DDPG agent.
        
        Args:
            state_dim: Dimension of the state space (int or tuple)
            action_dim: Dimension of the action space
            max_action: Maximum action value
            actor_lr: Learning rate for the actor
            critic_lr: Learning rate for the critic
            gamma: Discount factor
            tau: Soft update coefficient
            buffer_size: Size of the replay buffer
            batch_size: Batch size for updates
            policy_noise: Noise added to target policy during critic update
            noise_clip: Range to clip target policy noise
            exploration_noise: Standard deviation of exploration noise
            device: Device to run the model on (cpu or cuda)
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.exploration_noise = exploration_noise * max_action
        
        # Set default hidden sizes based on Practical 8
        hidden_sizes = (128, 128)
        
        # Initialize actor networks - based on Practical 8
        self.actor = Actor(state_dim, hidden_sizes, action_dim, actor_lr, max_action).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Initialize critic networks - based on Practical 8
        self.critic = Critic(state_dim, hidden_sizes, action_dim, critic_lr).to(device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim, device)
        
        # Training info
        self.total_it = 0
    
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
    
    def select_action(self, state, evaluate=False):
        """
        Select an action based on the current policy.
        
        Args:
            state: The current state
            evaluate: Whether to add exploration noise
            
        Returns:
            Selected action
        """
        # Preprocess state
        state = self.preprocess_state(state)
        
        # Set actor to evaluation mode
        self.actor.eval()
        
        with torch.no_grad():
            # Get action from actor
            action = self.actor(state).cpu().numpy().flatten()
        
        # Add exploration noise if not evaluating
        if not evaluate:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update(self, update_actor=True):
        """
        Update the actor and critic networks.
        
        Args:
            update_actor: Whether to update the actor network
            
        Returns:
            Dictionary with loss information
        """
        # If buffer doesn't have enough samples, return
        if len(self.replay_buffer) < self.batch_size:
            return {'critic_loss': 0, 'actor_loss': 0}
        
        self.total_it += 1
        
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Update critic
        with torch.no_grad():
            # Select action according to target policy and add clipped noise
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            
            next_actions = self.actor_target(next_states) + noise
            next_actions = torch.clamp(next_actions, -self.max_action, self.max_action)
            
            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic(states, actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = 0
        
        # Delayed policy updates
        if update_actor:
            # Compute actor loss
            actor_loss = -self.critic.q1(states, self.actor(states)).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.actor, self.actor_target)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss
        }
    
    def _soft_update(self, local_model, target_model):
        """
        Soft update of target network parameters.
        θ_target = τ*θ_local + (1-τ)*θ_target
        
        Args:
            local_model: Source model
            target_model: Target model to update
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, path):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
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
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
