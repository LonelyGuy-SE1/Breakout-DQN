import torch
import numpy as np
import gymnasium as gym
import ale_py

from Preprocessing.prep_pipeline import AtariPipeline
from Memory.buffer import LazyReplayBuffer
from Agent.dqn_agent import DQNAgent

#modify as desired
MAX_FRAMES=5000000
BATCH_SIZE=32
BUFFER_CAPACITY=250000
SYNC_TARGET_FRAMES=10000 # target network clones policy network

EPSILON_START=1.0
EPSILON_END=0.01
EPSILON_DECAY=5000000 #frame at which epsilon hits the minimum

def train():
    print("Starting Orchestratior!")
    gym.register_envs(ale_py)
    env=gym.make("ALE/Breakout-v5", render_mode="human")
    pipeline=AtariPipeline(stack_size=4,screen_size=84)
    buffer=LazyReplayBuffer(capacity=BUFFER_CAPACITY)
    agent=DQNAgent(state_shape=(4,84,84), num_actions=4)
    
    frame_idx=0
    episode_reward=0
    
    raw_obs, _ = env.reset()
    state=pipeline.reset(raw_obs)
    
    print("Starting training!")
    
    while frame_idx<MAX_FRAMES:
        
        epsilon=max(EPSILON_END,EPSILON_START-frame_idx*(EPSILON_START-EPSILON_END)/EPSILON_DECAY)
        action=agent.act(state,epsilon)
        
        raw_obs, reward, terminated, truncated, _ =env.step(action)
        done = terminated or truncated
        next_state=pipeline.step(raw_obs)
        
        buffer.add(next_state[-1],action, reward, done)
        
        state=next_state
        episode_reward+=reward
        frame_idx+=1
        
        if frame_idx>BATCH_SIZE:
            batch=buffer.sample(BATCH_SIZE)
            loss=agent.learn(batch)
            
        if frame_idx % SYNC_TARGET_FRAMES==0:
            agent.update_target_network()
            print(f"Frame {frame_idx} | Syncing Target Network | Epsilon: {epsilon:.2f} | Loss: {loss:.4f}")
            
        if done: 
            print(f"Game Over | Frames Survived: {frame_idx} | Score: {episode_reward}")
            raw_obs, _ = env.reset()
            state = pipeline.reset(raw_obs)
            episode_reward = 0
            
    print("Training Complete!")
        
    print("Saving Weights...")
    torch.save(agent.policy_net.state_dict(), "dqn_brain.pth")
    print("Weights Saved!")
    env.close()
        
if __name__=="__main__":
    train()
                


