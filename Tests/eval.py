import gymnasium as gym
import torch
import ale_py
from Preprocessing.prep_pipeline import AtariPipeline
from Agent.network import DQN

def deploy_agent():
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    pipeline = AtariPipeline(stack_size=4, screen_size=84)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brain = DQN(input_channels=4, num_actions=4).to(device)
    
    print("Loading weights...")
    #brain.load_state_dict(torch.load("dqn_brain.pth", map_location=device)) - use this line if the other does not work
    checkpoint = torch.load("dqn_brain.pth", map_location=device)
    brain.load_state_dict(checkpoint['policy_state_dict'])
    brain.eval() 
    
    raw_obs, _ = env.reset()
    state = pipeline.reset(raw_obs)
    done = False
    
    print("Agent Deployed!")
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            q_values = brain(state_tensor)
            action = q_values.argmax(dim=1).item()
            
        raw_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = pipeline.step(raw_obs)

    env.close()

if __name__ == "__main__":
    deploy_agent()