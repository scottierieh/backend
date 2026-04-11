"""
Reinforcement Learning FastAPI Endpoint
Train agents in simulated environments OR from uploaded CSV data
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

router = APIRouter()


class RLResponse(BaseModel):
    """Response model for RL training"""
    success: bool
    mode: str
    final_reward: float
    avg_reward: float
    episodes_trained: int
    convergence_episode: Optional[int]
    policy_info: Dict[str, Any]
    environment_info: Dict[str, Any]
    training_stats: Dict[str, Any]
    data_sample: Optional[Dict[str, Any]]
    plots: Dict[str, Optional[str]]
    interpretation: Dict[str, Any]


# ============================================================================
# SIMULATION ENVIRONMENTS
# ============================================================================

class GridWorld:
    """Simple Grid World environment (5x5)"""
    def __init__(self, size: int = 5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.start_state = 0
        self.goal_state = self.n_states - 1
        self.obstacles = [7, 11, 12, 17]
        self.reset()
    
    def reset(self):
        self.state = self.start_state
        return self.state
    
    def step(self, action):
        row, col = divmod(self.state, self.size)
        
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(self.size - 1, col + 1)
        
        next_state = row * self.size + col
        
        if next_state in self.obstacles:
            next_state = self.state
            reward = -10
            done = False
        elif next_state == self.goal_state:
            reward = 100
            done = True
        else:
            reward = -1
            done = False
        
        self.state = next_state
        return next_state, reward, done


class FrozenLake:
    """Frozen Lake environment (4x4)"""
    def __init__(self):
        self.size = 4
        self.n_states = 16
        self.n_actions = 4
        self.start_state = 0
        self.goal_state = 15
        self.holes = [5, 7, 11, 12]
        self.reset()
    
    def reset(self):
        self.state = self.start_state
        return self.state
    
    def step(self, action):
        row, col = divmod(self.state, self.size)
        
        # Slippery ice
        if np.random.random() < 0.33:
            action = np.random.randint(4)
        
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(self.size - 1, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(self.size - 1, col + 1)
        
        next_state = row * self.size + col
        
        if next_state in self.holes:
            reward = -100
            done = True
        elif next_state == self.goal_state:
            reward = 100
            done = True
        else:
            reward = -1
            done = False
        
        self.state = next_state
        return next_state, reward, done


class CliffWalking:
    """Cliff Walking environment (4x12)"""
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.n_states = 48
        self.n_actions = 4
        self.start_state = 36
        self.goal_state = 47
        self.cliff = list(range(37, 47))
        self.reset()
    
    def reset(self):
        self.state = self.start_state
        return self.state
    
    def step(self, action):
        row, col = divmod(self.state, self.cols)
        
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(self.rows - 1, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(self.cols - 1, col + 1)
        
        next_state = row * self.cols + col
        
        if next_state in self.cliff:
            reward = -100
            done = False
            next_state = self.start_state
        elif next_state == self.goal_state:
            reward = 0
            done = True
        else:
            reward = -1
            done = False
        
        self.state = next_state
        return next_state, reward, done


def create_environment(env_name: str):
    """Factory function"""
    if env_name == "grid_world":
        return GridWorld()
    elif env_name == "frozen_lake":
        return FrozenLake()
    elif env_name == "cliff_walking":
        return CliffWalking()
    else:
        raise ValueError(f"Unknown environment: {env_name}")


# ============================================================================
# TRAINING ALGORITHMS
# ============================================================================

def epsilon_greedy_policy(Q, state, epsilon, n_actions):
    """Epsilon-greedy action selection"""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[state])


def train_q_learning(env, episodes, lr, gamma, epsilon, epsilon_decay):
    """Q-Learning algorithm"""
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_per_episode = []
    steps_per_episode = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 200:
            action = epsilon_greedy_policy(Q, state, epsilon, env.n_actions)
            next_state, reward, done = env.step(action)
            
            best_next_action = np.argmax(Q[next_state])
            Q[state, action] += lr * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])
            
            state = next_state
            total_reward += reward
            steps += 1
        
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        epsilon *= epsilon_decay
    
    return Q, rewards_per_episode, steps_per_episode


def train_sarsa(env, episodes, lr, gamma, epsilon, epsilon_decay):
    """SARSA algorithm"""
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_per_episode = []
    steps_per_episode = []
    
    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy_policy(Q, state, epsilon, env.n_actions)
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 200:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon, env.n_actions)
            
            Q[state, action] += lr * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1
        
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        epsilon *= epsilon_decay
    
    return Q, rewards_per_episode, steps_per_episode


def train_expected_sarsa(env, episodes, lr, gamma, epsilon, epsilon_decay):
    """Expected SARSA algorithm"""
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_per_episode = []
    steps_per_episode = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 200:
            action = epsilon_greedy_policy(Q, state, epsilon, env.n_actions)
            next_state, reward, done = env.step(action)
            
            policy_probs = np.ones(env.n_actions) * epsilon / env.n_actions
            best_action = np.argmax(Q[next_state])
            policy_probs[best_action] += 1 - epsilon
            expected_q = np.sum(policy_probs * Q[next_state])
            
            Q[state, action] += lr * (reward + gamma * expected_q - Q[state, action])
            
            state = next_state
            total_reward += reward
            steps += 1
        
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        epsilon *= epsilon_decay
    
    return Q, rewards_per_episode, steps_per_episode


def train_from_csv(df, algorithm, lr, gamma, epsilon):
    """Train from CSV data"""
    # Validate
    required = ['state', 'action', 'reward', 'next_state', 'done']
    if not all(col in df.columns for col in required):
        raise ValueError(f"CSV must have columns: {required}")
    
    # Get dimensions
    n_states = max(df['state'].max(), df['next_state'].max()) + 1
    n_actions = df['action'].max() + 1
    Q = np.zeros((n_states, n_actions))
    
    # Train from each experience
    for _, row in df.iterrows():
        s = int(row['state'])
        a = int(row['action'])
        r = float(row['reward'])
        s_next = int(row['next_state'])
        done = bool(row['done'])
        
        if algorithm == "q_learning":
            target = r + (0 if done else gamma * np.max(Q[s_next]))
        elif algorithm == "sarsa":
            a_next = np.argmax(Q[s_next])
            target = r + (0 if done else gamma * Q[s_next, a_next])
        else:  # expected_sarsa
            policy_probs = np.ones(n_actions) * epsilon / n_actions
            policy_probs[np.argmax(Q[s_next])] += 1 - epsilon
            target = r + (0 if done else gamma * np.sum(policy_probs * Q[s_next]))
        
        Q[s, a] += lr * (target - Q[s, a])
    
    # Extract episodes
    rewards, steps = [], []
    ep_r, ep_s = 0, 0
    
    for _, row in df.iterrows():
        ep_r += row['reward']
        ep_s += 1
        if row['done']:
            rewards.append(ep_r)
            steps.append(ep_s)
            ep_r, ep_s = 0, 0
    
    if ep_s > 0:
        rewards.append(ep_r)
        steps.append(ep_s)
    
    return Q, rewards, steps, n_states, n_actions


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def create_learning_curve(rewards, steps, algorithm: str) -> str:
    """Create learning curve visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    window = min(50, len(rewards) // 10)
    if window > 1:
        smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        episodes = range(len(smoothed_rewards))
    else:
        smoothed_rewards = rewards
        episodes = range(len(rewards))
    
    ax1.plot(episodes, smoothed_rewards, linewidth=2, label='Smoothed Reward')
    ax1.axhline(np.mean(rewards[-100:]), color='red', linestyle='--', 
                linewidth=2, label=f'Final Avg: {np.mean(rewards[-100:]):.1f}')
    ax1.set_xlabel('Episode', fontsize=12, weight='bold')
    ax1.set_ylabel('Total Reward', fontsize=12, weight='bold')
    ax1.set_title(f'{algorithm} - Reward Progression', fontsize=13, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if window > 1:
        smoothed_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
    else:
        smoothed_steps = steps
    
    ax2.plot(episodes, smoothed_steps, linewidth=2, color='orange', label='Steps')
    ax2.set_xlabel('Episode', fontsize=12, weight='bold')
    ax2.set_ylabel('Steps per Episode', fontsize=12, weight='bold')
    ax2.set_title('Episode Length', fontsize=13, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def create_reward_distribution(rewards: List[float]) -> str:
    """Create reward distribution histogram"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(rewards, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(rewards), color='red', linestyle='--', 
              linewidth=2.5, label=f'Mean: {np.mean(rewards):.1f}')
    ax.axvline(np.median(rewards), color='orange', linestyle=':',
              linewidth=2.5, label=f'Median: {np.median(rewards):.1f}')
    
    ax.set_xlabel('Total Reward', fontsize=12, weight='bold')
    ax.set_ylabel('Frequency', fontsize=12, weight='bold')
    ax.set_title('Reward Distribution', fontsize=13, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def create_policy_visualization(Q, env_type: str, n_states: int, n_actions: int) -> str:
    """Visualize learned policy"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if env_type == "grid_world":
        size = 5
        obstacles = [7, 11, 12, 17]
        start, goal = 0, 24
    elif env_type == "frozen_lake":
        size = 4
        obstacles = [5, 7, 11, 12]
        start, goal = 0, 15
    elif env_type == "cliff_walking":
        rows, cols = 4, 12
        value_grid = np.zeros((rows, cols))
        for state in range(rows * cols):
            row, col = divmod(state, cols)
            value_grid[row, col] = np.max(Q[state])
        
        im = ax.imshow(value_grid, cmap='YlGnBu', aspect='auto')
        plt.colorbar(im, ax=ax, label='State Value')
        
        arrow_symbols = ['↑', '↓', '←', '→']
        for row in range(rows):
            for col in range(cols):
                state = row * cols + col
                if state == 36:
                    ax.text(col, row, 'S', ha='center', va='center', 
                           fontsize=20, weight='bold', color='blue')
                elif state == 47:
                    ax.text(col, row, 'G', ha='center', va='center',
                           fontsize=20, weight='bold', color='green')
                elif state in range(37, 47):
                    ax.text(col, row, 'C', ha='center', va='center',
                           fontsize=20, weight='bold', color='red')
                else:
                    action = int(np.argmax(Q[state]))
                    ax.text(col, row, arrow_symbols[action], ha='center', va='center',
                           fontsize=16, weight='bold')
        
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_title('Learned Policy (Cliff Walking)', fontsize=14, weight='bold', pad=20)
        ax.grid(True, linewidth=2, color='black', alpha=0.3)
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return img_str
    else:
        # Generic grid (for CSV data)
        size = int(np.sqrt(n_states))
        obstacles = []
        start, goal = 0, n_states - 1
    
    # For grid world / frozen lake / generic
    value_grid = np.zeros((size, size))
    for state in range(min(size * size, n_states)):
        row, col = divmod(state, size)
        value_grid[row, col] = np.max(Q[state])
    
    im = ax.imshow(value_grid, cmap='YlGnBu')
    plt.colorbar(im, ax=ax, label='State Value')
    
    arrow_symbols = ['↑', '↓', '←', '→']
    for row in range(size):
        for col in range(size):
            state = row * size + col
            if state >= n_states:
                continue
            if state == goal:
                ax.text(col, row, 'G', ha='center', va='center',
                       fontsize=20, weight='bold', color='green')
            elif state in obstacles:
                ax.text(col, row, 'X', ha='center', va='center',
                       fontsize=20, weight='bold', color='red')
            elif state == start:
                ax.text(col, row, 'S', ha='center', va='center',
                       fontsize=20, weight='bold', color='blue')
            else:
                action = int(np.argmax(Q[state]))
                if action < 4:
                    ax.text(col, row, arrow_symbols[action], ha='center', va='center',
                           fontsize=16, weight='bold')
    
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    title = 'Learned Policy (Grid World)' if env_type == "grid_world" else \
            'Learned Policy (Frozen Lake)' if env_type == "frozen_lake" else \
            'Learned Policy (Custom Environment)'
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    ax.grid(True, linewidth=2, color='black', alpha=0.3)
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/reinforcement-learning")
async def train_rl_agent(
    mode: str = Form(...),  # "simulation" or "offline"
    algorithm: str = Form(...),
    learning_rate: float = Form(0.1),
    discount_factor: float = Form(0.95),
    epsilon: float = Form(0.1),
    epsilon_decay: float = Form(0.995),
    episodes: int = Form(500),
    environment: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Train RL agent from simulation OR CSV data
    
    Mode: "simulation" - train in Grid World, Frozen Lake, or Cliff Walking
    Mode: "offline" - train from CSV (columns: state, action, reward, next_state, done)
    """
    try:
        if mode == "simulation":
            if not environment:
                raise HTTPException(400, "Environment required for simulation mode")
            
            # Create environment
            env = create_environment(environment)
            
            # Train
            if algorithm == "q_learning":
                Q, rewards, steps = train_q_learning(env, episodes, learning_rate, discount_factor, epsilon, epsilon_decay)
            elif algorithm == "sarsa":
                Q, rewards, steps = train_sarsa(env, episodes, learning_rate, discount_factor, epsilon, epsilon_decay)
            else:
                Q, rewards, steps = train_expected_sarsa(env, episodes, learning_rate, discount_factor, epsilon, epsilon_decay)
            
            # Environment info
            env_descriptions = {
                "grid_world": "5×5 grid with obstacles. Goal: reach bottom-right corner.",
                "frozen_lake": "4×4 slippery ice grid with holes. Stochastic environment.",
                "cliff_walking": "4×12 grid with cliff. High penalty for falling."
            }
            
            env_info = {
                "name": environment,
                "description": env_descriptions[environment],
                "n_states": env.n_states,
                "n_actions": env.n_actions,
                "start_state": env.start_state,
                "goal_state": env.goal_state
            }
            
            data_sample = None
            env_type = environment
            
        elif mode == "offline":
            if not file:
                raise HTTPException(400, "CSV file required for offline mode")
            
            # Read CSV
            df = pd.read_csv(file.file)
            
            # Train
            Q, rewards, steps, n_states, n_actions = train_from_csv(df, algorithm, learning_rate, discount_factor, epsilon)
            
            # Environment info
            env_info = {
                "name": "Custom (CSV)",
                "description": f"Learned from {len(df)} experiences",
                "n_states": int(n_states),
                "n_actions": int(n_actions),
                "start_state": 0,
                "goal_state": int(n_states - 1)
            }
            
            # Data sample
            data_sample = {
                "sample_experiences": df.head(10).to_dict('records'),
                "total_experiences": len(df),
                "n_episodes": len(rewards)
            }
            
            env_type = "custom"
        
        else:
            raise HTTPException(400, "Mode must be 'simulation' or 'offline'")
        
        # Calculate stats
        final_reward = float(np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards))
        avg_reward = float(np.mean(rewards))
        
        # Find convergence
        convergence_episode = None
        window = 50
        if len(rewards) > window:
            for i in range(window, len(rewards)):
                if np.std(rewards[i-window:i]) < 10:
                    convergence_episode = i
                    break
        
        # Generate plots
        plots = {
            "learning_curve": create_learning_curve(rewards, steps, algorithm.upper()),
            "policy": create_policy_visualization(Q, env_type, env_info["n_states"], env_info["n_actions"]),
            "reward_distribution": create_reward_distribution(rewards)
        }
        
        # Generate interpretation
        key_insights = []
        recommendations = []
        
        improvement = final_reward - np.mean(rewards[:min(100, len(rewards))])
        if improvement > 0:
            key_insights.append({
                "title": f"Successful Learning",
                "description": f"Agent improved reward by {improvement:.1f}. Final average: {final_reward:.1f}",
                "status": "positive"
            })
        else:
            key_insights.append({
                "title": "Limited Improvement",
                "description": f"Agent showed minimal improvement. Consider adjusting hyperparameters.",
                "status": "warning"
            })
        
        recent_std = np.std(rewards[-50:])
        if recent_std < 10:
            key_insights.append({
                "title": "Policy Converged",
                "description": f"Reward variance stabilized (std: {recent_std:.2f}).",
                "status": "positive"
            })
        
        if mode == "simulation":
            recommendations.append(f"Trained for {episodes} episodes in {environment}")
        else:
            recommendations.append(f"Trained from {data_sample['total_experiences']} experiences")
        
        recommendations.append(f"Average steps per episode: {np.mean(steps):.1f}")
        
        return RLResponse(
            success=True,
            mode=mode,
            final_reward=final_reward,
            avg_reward=avg_reward,
            episodes_trained=len(rewards),
            convergence_episode=convergence_episode,
            policy_info={
                "n_states": env_info["n_states"],
                "n_actions": env_info["n_actions"],
                "max_q_value": float(np.max(Q)),
                "min_q_value": float(np.min(Q))
            },
            environment_info=env_info,
            training_stats={
                "algorithm": algorithm,
                "learning_rate": learning_rate,
                "discount_factor": discount_factor,
                "initial_epsilon": epsilon,
                "epsilon_decay": epsilon_decay,
                "best_reward": float(np.max(rewards)),
                "worst_reward": float(np.min(rewards)),
                "avg_steps": float(np.mean(steps))
            },
            data_sample=data_sample,
            plots=plots,
            interpretation={
                "key_insights": key_insights,
                "recommendations": recommendations
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")
