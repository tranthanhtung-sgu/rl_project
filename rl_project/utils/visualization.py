import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

def moving_average(data, window_size):
    """
    Calculate the moving average of a list.
    
    Args:
        data: List of values
        window_size: Size of the moving average window
        
    Returns:
        List of moving averages
    """
    if window_size > len(data):
        window_size = len(data)
    
    if window_size <= 1:
        return data
    
    # Calculate cumulative sum
    cumsum = np.cumsum(np.insert(data, 0, 0))
    
    # Calculate moving average
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

def plot_learning_curve(data, title, xlabel, ylabel, window=10, figsize=(10, 6), save_path=None):
    """
    Plot a learning curve with a moving average.
    
    Args:
        data: List of values to plot
        title: Title of the plot
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
        window: Size of the moving average window
        figsize: Figure size
        save_path: Path to save the figure (optional)
        
    Returns:
        Figure and axes objects
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot raw data
    ax.plot(data, alpha=0.3, label='Raw')
    
    # Plot moving average
    if len(data) > window:
        ma_data = moving_average(data, window)
        ma_x = np.arange(window - 1, len(data))
        ax.plot(ma_x, ma_data, linewidth=2, label=f'Moving Avg (w={window})')
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Set integer ticks for x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_metrics_from_csv(csv_path, window=10, figsize=(15, 10), save_dir=None):
    """
    Plot metrics from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        window: Size of the moving average window
        figsize: Figure size
        save_dir: Directory to save the figures (optional)
        
    Returns:
        Dictionary of figure objects
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Extract agent and environment names from path
    path_parts = os.path.basename(os.path.dirname(csv_path)).split('_')
    agent_name = path_parts[0]
    env_name = '_'.join(path_parts[1:])
    
    # Create figures
    figs = {}
    
    # Plot reward
    if 'reward' in df.columns:
        title = f"{agent_name} on {env_name} - Reward"
        fig, ax = plot_learning_curve(
            df['reward'].values, title, 'Episode', 'Reward', window, figsize
        )
        figs['reward'] = fig
        
        if save_dir:
            save_path = os.path.join(save_dir, f"{agent_name}_{env_name}_reward.png")
            plt.figure(fig.number)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Plot episode length
    if 'length' in df.columns:
        title = f"{agent_name} on {env_name} - Episode Length"
        fig, ax = plot_learning_curve(
            df['length'].values, title, 'Episode', 'Length', window, figsize
        )
        figs['length'] = fig
        
        if save_dir:
            save_path = os.path.join(save_dir, f"{agent_name}_{env_name}_length.png")
            plt.figure(fig.number)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return figs

def plot_agent_comparison(metrics_paths, metric='reward', window=10, figsize=(12, 8), save_path=None):
    """
    Plot a comparison of different agents on the same environment.
    
    Args:
        metrics_paths: Dictionary mapping agent names to metrics CSV paths
        metric: Metric to plot ('reward' or 'length')
        window: Size of the moving average window
        figsize: Figure size
        save_path: Path to save the figure (optional)
        
    Returns:
        Figure and axes objects
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract environment name from the first path
    path_parts = os.path.basename(os.path.dirname(list(metrics_paths.values())[0])).split('_')
    env_name = '_'.join(path_parts[1:])
    
    # Plot data for each agent
    for agent_name, path in metrics_paths.items():
        # Load data
        df = pd.read_csv(path)
        
        # Plot raw data
        ax.plot(df[metric].values, alpha=0.1)
        
        # Plot moving average
        if len(df) > window:
            ma_data = moving_average(df[metric].values, window)
            ma_x = np.arange(window - 1, len(df))
            ax.plot(ma_x, ma_data, linewidth=2, label=agent_name)
    
    # Set labels and title
    ax.set_xlabel('Episode')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Agent Comparison on {env_name} - {metric.capitalize()}")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Set integer ticks for x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_environment_comparison(metrics_paths, agent_name, metric='reward', window=10, figsize=(12, 8), save_path=None):
    """
    Plot a comparison of an agent on different environments.
    
    Args:
        metrics_paths: Dictionary mapping environment names to metrics CSV paths
        agent_name: Name of the agent
        metric: Metric to plot ('reward' or 'length')
        window: Size of the moving average window
        figsize: Figure size
        save_path: Path to save the figure (optional)
        
    Returns:
        Figure and axes objects
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data for each environment
    for env_name, path in metrics_paths.items():
        # Load data
        df = pd.read_csv(path)
        
        # Plot raw data
        ax.plot(df[metric].values, alpha=0.1)
        
        # Plot moving average
        if len(df) > window:
            ma_data = moving_average(df[metric].values, window)
            ma_x = np.arange(window - 1, len(df))
            ax.plot(ma_x, ma_data, linewidth=2, label=env_name)
    
    # Set labels and title
    ax.set_xlabel('Episode')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{agent_name} on Different Environments - {metric.capitalize()}")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Set integer ticks for x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def create_summary_plots(log_dir, save_dir=None, window=10):
    """
    Create summary plots for all agents and environments.
    
    Args:
        log_dir: Directory containing the logs
        save_dir: Directory to save the plots (optional)
        window: Size of the moving average window
        
    Returns:
        Dictionary of figure objects
    """
    # Create save directory if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Get all metrics files
    metrics_files = {}
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file == 'metrics.csv':
                # Extract agent and environment names from directory
                dir_name = os.path.basename(root)
                if '_' in dir_name:
                    agent_name, env_name = dir_name.split('_', 1)
                    
                    # Store path
                    if agent_name not in metrics_files:
                        metrics_files[agent_name] = {}
                    
                    metrics_files[agent_name][env_name] = os.path.join(root, file)
    
    # Create individual plots
    individual_figs = {}
    for agent_name, env_dict in metrics_files.items():
        for env_name, path in env_dict.items():
            figs = plot_metrics_from_csv(
                path, window=window,
                save_dir=save_dir
            )
            
            for metric, fig in figs.items():
                individual_figs[f"{agent_name}_{env_name}_{metric}"] = fig
    
    # Create agent comparison plots for each environment
    env_names = set()
    for agent_dict in metrics_files.values():
        env_names.update(agent_dict.keys())
    
    comparison_figs = {}
    for env_name in env_names:
        # Collect paths for agents that have data for this environment
        env_paths = {}
        for agent_name, agent_dict in metrics_files.items():
            if env_name in agent_dict:
                env_paths[agent_name] = agent_dict[env_name]
        
        # Create comparison plots
        if len(env_paths) > 1:
            for metric in ['reward', 'length']:
                save_path = None
                if save_dir:
                    save_path = os.path.join(save_dir, f"comparison_{env_name}_{metric}.png")
                
                fig, ax = plot_agent_comparison(
                    env_paths, metric=metric, window=window,
                    save_path=save_path
                )
                
                comparison_figs[f"comparison_{env_name}_{metric}"] = fig
    
    # Create environment comparison plots for each agent
    for agent_name, agent_dict in metrics_files.items():
        # Create comparison plots
        if len(agent_dict) > 1:
            for metric in ['reward', 'length']:
                save_path = None
                if save_dir:
                    save_path = os.path.join(save_dir, f"{agent_name}_env_comparison_{metric}.png")
                
                fig, ax = plot_environment_comparison(
                    agent_dict, agent_name, metric=metric, window=window,
                    save_path=save_path
                )
                
                comparison_figs[f"{agent_name}_env_comparison_{metric}"] = fig
    
    # Combine all figures
    all_figs = {**individual_figs, **comparison_figs}
    
    return all_figs
