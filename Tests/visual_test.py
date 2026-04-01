# the objective of this file is to visualize the env/action space, etc.

import time

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5", render_mode="human")
observation, info = env.reset()

print("Observation space:", observation.shape)
print("Action space:", env.action_space)
print("Action meanings:", env.unwrapped.get_action_meanings())

for _ in range(200):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

    time.sleep(0.02)

env.close()