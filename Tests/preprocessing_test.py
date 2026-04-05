import gymnasium as gym
import numpy as np
import ale_py
from Preprocessing.prep_pipeline import AtariPipeline

gym.register_envs(ale_py)

def save_as_pgm(filename, image_matrix):
    img_uint8 = (image_matrix * 255).astype(np.uint8)
    height, width = img_uint8.shape
    with open(filename, 'w') as f:
        f.write(f"P2\n{width} {height}\n255\n")
        for row in img_uint8:
            f.write(" ".join(str(pixel) for pixel in row) + "\n")

def test_full_factory():
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    pipeline = AtariPipeline(stack_size=4, screen_size=84)
    
    raw_obs, _ = env.reset()
    
    raw_obs, _, _, _, _ = env.step(1) 
    
    state_stack = pipeline.reset(raw_obs)
    print("--- PIPELINE RESET TEST ---")
    print(f"Stack Shape: {state_stack.shape} (Target: 4, 84, 84)")
    print(f"Data Type: {state_stack.dtype} (Target: float32)")
    
    for _ in range(3):
        raw_obs, _, _, _, _ = env.step(2) 
        state_stack = pipeline.step(raw_obs)
    
    print("\n--- PIPELINE STEP TEST ---")
    print(f"Final Stack Shape: {state_stack.shape}")
    print(f"Max Val: {state_stack.max():.2f} (Target: <= 1.0)")
    
    latest_frame = state_stack[-1]
    save_as_pgm("test_pipeline_output.pgm", latest_frame)
    print("\nVerification file saved as 'test_pipeline_output.pgm'.")
    
    env.close()

if __name__ == "__main__":
    test_full_factory()