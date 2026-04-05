from .spatial import SpatialProcessor
from .temporal import TemporalProcessor


class AtariPipeline:
    def __init__(self, stack_size=4, screen_size=84):
        self.spatial_processor = SpatialProcessor(size=screen_size)
        self.temporal_processor = TemporalProcessor(num_frames=stack_size, size=screen_size)
    
    def reset(self, frame):
        processed_frame=self.spatial_processor.process(frame)
        state_stack = self.temporal_processor.reset(processed_frame)
        return state_stack
    
    def step(self, raw_frame):
        processed_frame=self.spatial_processor.process(raw_frame)
        state_stack = self.temporal_processor.step(processed_frame)
        return state_stack