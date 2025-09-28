# RL Agent Visualization Guide

This guide shows you how to visualize your trained RL agents running in their environments, just like the examples on the Gymnasium website!

## 🎯 Quick Start

### Method 1: Quick Visualization (Recommended for beginners)

1. **List available models:**
   ```bash
   python list_models.py
   ```

2. **Edit the quick visualization script:**
   ```bash
   nano quick_visualize.py
   ```
   
   Update these variables:
   ```python
   AGENT = 'ddpg'  # 'reinforce', 'a2c', or 'ddpg'
   ENV = 'ant'     # 'ant', 'breakout', or 'seaquest'
   MODEL_PATH = 'logs/20250928_191125/ddpg_ant/final_model.pt'
   EPISODES = 3
   ```

3. **Run visualization:**
   ```bash
   python quick_visualize.py
   ```

### Method 2: Advanced Visualization

Use the full visualization script with more options:

```bash
python visualize.py --agent ddpg --env ant --model_path logs/20250928_191125/ddpg_ant/final_model.pt --episodes 3
```

**Options:**
- `--agent`: reinforce, a2c, or ddpg
- `--env`: ant, breakout, or seaquest
- `--model_path`: Path to your trained model
- `--episodes`: Number of episodes to visualize
- `--save_video`: Save videos of the episodes
- `--device`: cpu or cuda

## 🎮 What You'll See

### Ant Environment (Continuous Control)
- A 3D ant robot learning to walk
- The ant will try to move forward as fast as possible
- You'll see the ant's joints moving and body orientation

### Breakout Environment (Atari Game)
- Classic Atari Breakout game
- The agent controls the paddle to bounce the ball
- You'll see the game screen with the paddle and ball

### Seaquest Environment (Atari Game)
- Underwater submarine game
- The agent controls a submarine to shoot enemies
- You'll see the underwater environment with enemies and obstacles

## 📹 Recording Videos

To save videos of your agent's performance:

```bash
python visualize.py --agent ddpg --env ant --model_path logs/20250928_191125/ddpg_ant/final_model.pt --save_video --episodes 5
```

Videos will be saved in the `videos/` directory.

## 🔧 Troubleshooting

### "No module named 'gymnasium'" Error
Make sure you're in the correct conda environment:
```bash
conda activate rl_project
```

### "Model not found" Error
1. Check available models: `python list_models.py`
2. Train a model first: `python main.py --agent ddpg --env ant --episodes 100`

### Visualization Window Not Appearing
- Make sure you have a display (X11 forwarding if using SSH)
- Try running with `--device cpu` if you have GPU issues
- For headless servers, use `--save_video` instead

### Performance Issues
- Use `--device cpu` for better compatibility
- Reduce `--episodes` for faster testing
- Close other applications to free up memory

## 🎯 Example Commands

### Visualize DDPG on Ant (Continuous Control)
```bash
python visualize.py --agent ddpg --env ant --model_path logs/20250928_191125/ddpg_ant/final_model.pt --episodes 3
```

### Visualize REINFORCE on Breakout (Atari)
```bash
python visualize.py --agent reinforce --env breakout --model_path logs/20250928_190414/reinforce_breakout/final_model.pt --episodes 3
```

### Visualize A2C on Seaquest (Atari)
```bash
python visualize.py --agent a2c --env seaquest --model_path logs/20250928_190958/a2c_breakout/final_model.pt --episodes 3
```

### Record Video of Agent Performance
```bash
python visualize.py --agent ddpg --env ant --model_path logs/20250928_191125/ddpg_ant/final_model.pt --save_video --episodes 5
```

## 📊 Understanding the Output

When you run visualization, you'll see:
- **Episode progress**: Step count and current reward
- **Episode summary**: Total reward and steps for each episode
- **Final summary**: Average performance across all episodes

## 🚀 Tips for Better Visualization

1. **Train longer**: Models trained for more episodes will perform better
2. **Use deterministic actions**: The visualization uses deterministic actions (no exploration)
3. **Multiple episodes**: Run several episodes to see consistent performance
4. **Record videos**: Save videos to share your results or analyze performance later

## 🎉 Enjoy Watching Your Agents!

Your RL agents are now ready to show off their skills! The visualization will open a window showing your agent interacting with the environment in real-time, just like the examples on the Gymnasium website.

Happy visualizing! 🎮🤖
