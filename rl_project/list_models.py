#!/usr/bin/env python3
"""
List all available trained models in the logs directory.
"""

import os
import glob

def list_models():
    """List all available trained models"""
    print("Available Trained Models:")
    print("=" * 50)
    
    # Find all model files
    model_files = glob.glob("logs/**/final_model.pt", recursive=True)
    
    if not model_files:
        print("No trained models found in logs/ directory.")
        print("\nTo train a model, run:")
        print("python main.py --agent reinforce --env ant --episodes 100")
        print("python main.py --agent a2c --env breakout --episodes 100")
        print("python main.py --agent ddpg --env ant --episodes 100")
        return
    
    for model_path in sorted(model_files):
        # Extract information from path
        parts = model_path.split('/')
        if len(parts) >= 3:
            timestamp = parts[1]
            agent_env = parts[2]
            
            # Parse agent and environment
            if '_' in agent_env:
                agent, env = agent_env.split('_', 1)
                print(f"Agent: {agent.upper()}")
                print(f"Environment: {env}")
                print(f"Timestamp: {timestamp}")
                print(f"Model path: {model_path}")
                print("-" * 30)
    
    print(f"\nTotal models found: {len(model_files)}")
    print("\nTo visualize a model, use:")
    print("python quick_visualize.py")
    print("(Edit the MODEL_PATH variable in the script)")

if __name__ == "__main__":
    list_models()
