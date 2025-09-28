import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStack,
    GrayScaleObservation,
    ResizeObservation,
    RecordEpisodeStatistics,
    RecordVideo
)
import numpy as np

def make_env(env_id, seed=42, capture_video=False, run_name=None, render_mode=None):
    """
    Create and configure an environment with appropriate wrappers.
    
    Args:
        env_id (str): The ID of the environment to create ('Ant-v4', 
                     'BreakoutNoFrameskip-v4', or 'SeaquestNoFrameskip-v4')
        seed (int): Random seed for reproducibility
        capture_video (bool): Whether to capture videos of episodes
        run_name (str): Name of the run for video capturing
        render_mode (str): Render mode ('human', 'rgb_array', or None)
        
    Returns:
        gym.Env: The configured environment
    """
    # Determine if the environment is Atari
    is_atari = 'NoFrameskip' in env_id
    
    # Create the base environment
    if render_mode is None:
        render_mode = "rgb_array" if capture_video else None
    env = gym.make(env_id, render_mode=render_mode)
    
    # Add episode statistics recording
    env = RecordEpisodeStatistics(env)
    
    # Add video recording if requested
    if capture_video and run_name:
        env = RecordVideo(env, video_folder=f"videos/{run_name}", episode_trigger=lambda x: True)
    
    # Set the seed for reproducibility
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    if is_atari:
        # Apply standard Atari preprocessing
        env = AtariPreprocessing(
            env,
            terminal_on_life_loss=False,
            scale_obs=False,
            grayscale_obs=True,
            grayscale_newaxis=True,
            frame_skip=4,
            noop_max=30
        )
        # Resize observation to 84x84
        env = ResizeObservation(env, (84, 84))
        # Stack 4 frames
        env = FrameStack(env, 4)
    
    return env

def get_env_dimensions(env_id):
    """
    Get the dimensions of the observation and action spaces for an environment.
    
    Args:
        env_id (str): The ID of the environment
        
    Returns:
        tuple: (observation_dim, action_dim, is_continuous)
    """
    # Create a temporary environment to get dimensions
    env = make_env(env_id)
    
    # Get observation space dimensions
    if isinstance(env.observation_space, gym.spaces.Box):
        if len(env.observation_space.shape) == 4:  # Image observation (Atari with FrameStack)
            # For Atari with FrameStack, shape is (4, 84, 84, 1)
            # We'll keep the shape as is for the CNN
            obs_dim = (4, 84, 84, 1)
        else:  # Vector observation (Ant)
            obs_dim = np.prod(env.observation_space.shape)
    else:
        raise ValueError(f"Unsupported observation space: {env.observation_space}")
    
    # Get action space dimensions and type
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        is_continuous = False
    elif isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        is_continuous = True
    else:
        raise ValueError(f"Unsupported action space: {env.action_space}")
    
    # Close the environment
    env.close()
    
    return obs_dim, action_dim, is_continuous
