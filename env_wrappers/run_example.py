import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time

from env.firefighting_env import FireFightingEnv
from env_wrappers.observation_wrapper import SimpleObservationWrapper
from env_wrappers.reward_wrapper import TimePenaltyRewardWrapper

"""
This file shows you how you can use our custom created observation and reward wrappers to step through the FireFightingEnv()
"""
# 1. Start with your base environment
base_env = FireFightingEnv()
obs, info = base_env.reset()
print(f"Initial Observation (from base_env): {obs}")

# 2. Apply the observation wrapper first
#    This creates an environment that speaks in "simple observations"
obs_wrapped_env = SimpleObservationWrapper(base_env)

# 3. Apply the reward wrapper on top of the already-wrapped environment
#    This adds the time penalty logic on top of the simple observation logic
final_env = TimePenaltyRewardWrapper(obs_wrapped_env)

# Now, `final_env` has both modifications!
# - When you call final_env.step(), you will get back the simplified observation
# - AND the reward will have the time penalty applied.

# --- Example Usage ---
obs, info = final_env.reset()
print(f"Initial Observation (from final_env): {obs}")
print(f"Observation Space (from final_env): {final_env.observation_space}")
print(f"Info: {info}")
print("-" * 30)

# Take a step
action = final_env.action_space.sample() # Use a random action
observation, reward, terminated, truncated, info = final_env.step(action)

print(f"Action taken: {action}")
print(f"Next Observation: {observation}")
print(f"Reward Received: {reward}")
print(f"Info: {info}")


