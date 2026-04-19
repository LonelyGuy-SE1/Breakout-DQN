import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from Agent.network import DQN

class DQNAgent:
    def __init__(self, state_shape=(4,84,84), num_actions=4, lr=1e-4, gamma=0.99):
        
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions=num_actions
        self.gamma=gamma
        
        # twin brain.. active network and target network
        self.policy_net=DQN(input_channels=state_shape[0], num_actions=num_actions).to(self.device)
        self.target_net=DQN(input_channels=state_shape[0], num_actions=num_actions).to(self.device)
        
        #freezing target network to match starting policy
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        #adam optimizer
        self.optimizer=optim.Adam(self.policy_net.parameters(),lr=lr)
        
    def act(self, state, epsilon): 
        
        if random.random()<epsilon:
            return random.randrange(self.num_actions)
        
        state_tensor=torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values=self.policy_net(state_tensor)
            best_action=q_values.argmax(dim=1).item()
            
        return best_action
    
    def learn(self, experiences):
        #bellman eqn
        
        states, actions, rewards, next_states, dones = experiences
        states=torch.FloatTensor(states).to(self.device)
        next_states=torch.FloatTensor(next_states).to(self.device)
        # unsqueeze to convert row matrix to column so bellman math can be done
        actions=torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards=torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones=torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        current_q=self.policy_net(states).gather(1,actions)
        
        #reality
        with torch.no_grad():
            max_next_q=self.target_net(next_states).max(1)[0].unsqueeze(1)
            expected_q=rewards+(self.gamma*max_next_q*(1-dones))
        
        #Huber loss
        loss=nn.functional.smooth_l1_loss(current_q,expected_q)
        
        #backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        
        
                
        
        
        
        
        