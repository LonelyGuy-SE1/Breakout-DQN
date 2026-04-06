#buffer of frames
#lazybuffer will handle frames while training and this will be used while playing the game post training

import numpy as np

class TemporalProcessor:
    def __init__(self, num_frames=4, size=84):
        self.num_frames=num_frames
        self.size=size
        self.frames=np.zeros((num_frames, size, size), dtype=np.float32)
    
    def reset(self, initial_frame):
        for i in range(self.num_frames):
            self.frames[i]=initial_frame
        return self.frames
    
    def step(self, new_frame):
        self.frames[:-1]=self.frames[1:]
        self.frames[-1]=new_frame
        return self.frames