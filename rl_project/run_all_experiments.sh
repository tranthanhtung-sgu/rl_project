#!/bin/bash
# Script to run all experiments

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rl_project

# Create logs directory
mkdir -p logs

# Define agents and environments
AGENTS=("reinforce" "a2c" "ddpg")
ENVS=("ant" "breakout" "seaquest")

# Run all combinations
for agent in "${AGENTS[@]}"; do
  for env in "${ENVS[@]}"; do
    echo "Running $agent on $env"
    
    # Skip REINFORCE on Ant (too sample inefficient)
    if [ "$agent" == "reinforce" ] && [ "$env" == "ant" ]; then
      echo "Skipping REINFORCE on Ant (too sample inefficient)"
      continue
    fi
    
    # Adjust episodes based on environment
    if [ "$env" == "ant" ]; then
      EPISODES=1000
    else
      EPISODES=500
    fi
    
    # Run experiment
    python main.py --agent $agent --env $env --episodes $EPISODES
    
    echo "Finished $agent on $env"
    echo "------------------------"
  done
done

# Create visualization plots
echo "Creating visualization plots"
python visualize.py

echo "All experiments completed!"
