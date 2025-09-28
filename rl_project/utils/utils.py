import os
import numpy as np
import pandas as pd
import torch
import random
import time
from datetime import datetime

def set_seeds(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: The seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Set deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_log_dir(base_dir="logs"):
    """
    Create a directory for logging with a timestamp.
    
    Args:
        base_dir: Base directory for logs
        
    Returns:
        Path to the created log directory
    """
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create a timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    return log_dir

def save_metrics(metrics, filepath):
    """
    Save metrics to a CSV file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save the CSV file
    """
    # Convert metrics to DataFrame
    df = pd.DataFrame(metrics)
    
    # Save to CSV
    df.to_csv(filepath, index=False)

def load_metrics(filepath):
    """
    Load metrics from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with metrics
    """
    return pd.read_csv(filepath)

class MetricsLogger:
    """
    Logger for tracking and saving training metrics.
    """
    def __init__(self, log_dir, agent_name, env_name):
        """
        Initialize the metrics logger.
        
        Args:
            log_dir: Directory to save logs
            agent_name: Name of the agent
            env_name: Name of the environment
        """
        self.log_dir = log_dir
        self.agent_name = agent_name
        self.env_name = env_name
        
        # Create agent-environment specific directory
        self.run_dir = os.path.join(log_dir, f"{agent_name}_{env_name}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.losses = []
        
        # Tracking variables
        self.episode_start_time = None
        self.training_start_time = time.time()
    
    def start_episode(self):
        """Mark the start of an episode for timing"""
        self.episode_start_time = time.time()
    
    def end_episode(self, reward, length, loss=None):
        """
        Log metrics at the end of an episode.
        
        Args:
            reward: Total reward for the episode
            length: Length of the episode
            loss: Loss information (optional)
        """
        # Calculate episode time
        if self.episode_start_time is not None:
            episode_time = time.time() - self.episode_start_time
            self.episode_times.append(episode_time)
        
        # Store metrics
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        if loss is not None:
            self.losses.append(loss)
    
    def save(self, checkpoint=False):
        """
        Save metrics to CSV files.
        
        Args:
            checkpoint: Whether this is a checkpoint save
        """
        # Create metrics dictionary
        metrics = {
            'episode': list(range(1, len(self.episode_rewards) + 1)),
            'reward': self.episode_rewards,
            'length': self.episode_lengths,
            'time': self.episode_times
        }
        
        # Save metrics
        metrics_path = os.path.join(self.run_dir, "metrics.csv")
        save_metrics(metrics, metrics_path)
        
        # Save losses if available
        if self.losses:
            # Convert losses to DataFrame
            loss_df = pd.DataFrame(self.losses)
            loss_path = os.path.join(self.run_dir, "losses.csv")
            loss_df.to_csv(loss_path, index=False)
    
    def get_latest_metrics(self, window=10):
        """
        Get the latest metrics averaged over a window.
        
        Args:
            window: Number of episodes to average over
            
        Returns:
            Dictionary with latest metrics
        """
        if not self.episode_rewards:
            return {'reward': 0, 'length': 0, 'time': 0}
        
        # Calculate window size
        window = min(window, len(self.episode_rewards))
        
        # Calculate averages
        avg_reward = np.mean(self.episode_rewards[-window:])
        avg_length = np.mean(self.episode_lengths[-window:])
        avg_time = np.mean(self.episode_times[-window:])
        
        return {
            'reward': avg_reward,
            'length': avg_length,
            'time': avg_time
        }
    
    def get_total_training_time(self):
        """Get the total training time in seconds"""
        return time.time() - self.training_start_time
