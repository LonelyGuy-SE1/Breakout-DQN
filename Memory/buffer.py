import numpy as np
import random

class LazyReplayBuffer:
    def __init__(self,capacity, frame_shape=(84,84),history_length=4):
        self.capacity=capacity
        self.history_length=history_length
        self.pos=0
        self.full=False
        #memory allocation
        self.frames=np.zeros((capacity, *frame_shape), dtype=np.uint8)
        self.actions=np.zeros(capacity, dtype=np.uint8)
        self.rewards=np.zeros(capacity, dtype=np.float32)
        #for edge case of death while taping together frames when needed
        self.dones=np.zeros(capacity, dtype=np.bool_)
        
    def add(self, frame, action, reward, done):
        self.frames[self.pos]=(frame*255).astype(np.uint8)
        self.actions[self.pos]=action
        self.rewards[self.pos]=reward
        self.dones[self.pos]=done
        #circular pointer logic
        self.pos=(self.pos+1) % self.capacity
        if self.pos==0:
            self.full=True
    
    def _get_stack(self, index):
        stack=np.zeros((self.history_length,84,84),dtype=np.float32)
        
        for i in range(self.history_length):
            curr_idx=(index-i)%self.capacity
            
            if not self.dones[curr_idx] and curr_idx>=self.pos:
                break
            
            stack[self.history_length-1-i]=self.frames[curr_idx].astype(np.float32)/255
            
            #edge case of death
            if i<self.history_length-1 and self.dones[curr_idx]:
                break
        
        return stack
    
    def sample(self, batch_size):
        max_idx=self.capacity if self.full else self.pos
        indices = np.random.randint(0,max_idx, size=batch_size)
        
        states=np.zeros((batch_size, self.history_length,84,84), dtype=np.float32)
        next_states=np.zeros((batch_size, self.history_length,84,84), dtype=np.float32)
        actions=np.zeros(batch_size, dtype=np.int64)
        rewards=np.zeros(batch_size, dtype=np.float32)
        dones=np.zeros(batch_size, dtype=np.bool_)
        
        for i, idx in enumerate(indices):
            states[i]=self._get_stack(idx)
            next_idx=(idx+1) % self.capacity
            next_states[i]=self._get_stack(next_idx)
            
            actions[i]=self.actions[idx]
            rewards[i]=self.rewards[idx]
            dones[i]=self.dones[idx]
            
        return states, actions, rewards, next_states, dones
            
                
            
