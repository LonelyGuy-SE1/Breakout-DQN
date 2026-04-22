import os
import torch
import numpy as np
import gymnasium as gym
import ale_py
import signal

from Preprocessing.prep_pipeline import AtariPipeline
from Memory.buffer import LazyReplayBuffer
from Agent.dqn_agent import DQNAgent

MAX_FRAMES = 5000000
BATCH_SIZE = 32
BUFFER_CAPACITY = 250000
SYNC_TARGET_FRAMES = 10000 
SAVE_INTERVAL = 50000 

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 5000000

WEIGHTS_PATH = "/content/drive/MyDrive/dqn_brain.pth"

SHUTDOWN_REQUESTED = False

def secure_shutdown(sig, frame):
    global SHUTDOWN_REQUESTED
    if not SHUTDOWN_REQUESTED:
        print("\n[CRITICAL] Kill Signal Detected. Locking process for atomic save...")
        SHUTDOWN_REQUESTED = True
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

signal.signal(signal.SIGINT, secure_shutdown)
signal.signal(signal.SIGTERM, secure_shutdown)

def save_checkpoint(frame_idx, agent, path):
    temp_path = path + ".tmp"
    checkpoint = {
        'frame_idx': frame_idx,
        'policy_state_dict': agent.policy_net.state_dict(),
        'target_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict()
    }
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path) 

def train():
    print("Starting Orchestrator!")
    gym.register_envs(ale_py)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on hardware: {device}")

    env = gym.make("ALE/Breakout-v5")
    pipeline = AtariPipeline(stack_size=4, screen_size=84)
    buffer = LazyReplayBuffer(capacity=BUFFER_CAPACITY)
    
    agent = DQNAgent(state_shape=(4,84,84), num_actions=4)
    
    frame_idx = 0
    
    if os.path.exists(WEIGHTS_PATH):
        try:
            print(f"Loading existing checkpoint from {WEIGHTS_PATH}...")
            checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
            
            agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            agent.target_net.load_state_dict(checkpoint['target_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            frame_idx = checkpoint['frame_idx']
            
            print(f"Resumed at Frame {frame_idx}. Epsilon clock restored.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")
            frame_idx = 0
    
    episode_reward = 0
    raw_obs, _ = env.reset()
    state = pipeline.reset(raw_obs)
    
    print("Starting training loop...")
    
    while frame_idx < MAX_FRAMES:
        epsilon = max(EPSILON_END, EPSILON_START - frame_idx * (EPSILON_START - EPSILON_END) / EPSILON_DECAY)
        action = agent.act(state, epsilon)
        
        raw_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = pipeline.step(raw_obs)
        
        buffer.add(next_state[-1], action, reward, done)
        state = next_state
        episode_reward += reward
        frame_idx += 1
        
        if frame_idx > BATCH_SIZE:
            agent.learn(buffer.sample(BATCH_SIZE))
            
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            agent.update_target_network()
            print(f"Frame {frame_idx} | Syncing | Epsilon: {epsilon:.2f}")
            
        if frame_idx % SAVE_INTERVAL == 0:
            save_checkpoint(frame_idx, agent, WEIGHTS_PATH)
            print(f"--- ATOMIC CHECKPOINT SECURED AT FRAME {frame_idx} ---")
            
        if done: 
            raw_obs, _ = env.reset()
            state = pipeline.reset(raw_obs)
            episode_reward = 0

        if SHUTDOWN_REQUESTED:
            break
    
    print("\nExecuting Final Core Dump...")
    save_checkpoint(frame_idx, agent, WEIGHTS_PATH)
    print(f"Universe saved at Frame {frame_idx}. Process Terminated.")
    env.close()

if __name__ == "__main__":
    train()