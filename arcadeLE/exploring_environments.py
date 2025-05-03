import gymnasium
import ale_py
from atariari.benchmark.wrapper import AtariARIWrapper

"""
First test with reward wrappers, can be ignored for now
"""

class CustomRewardWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        new_reward = self.modify_reward(reward, obs, info, terminated)

        return obs, new_reward, terminated, info

    def modify_reward(self, reward, obs, info, terminated):
        return reward

gymnasium.register_envs(ale_py)

env = gymnasium.make("Pong-v4", render_mode="human")
env = CustomRewardWrapper(env)
env = AtariARIWrapper(env)
env.reset()
output_file = open("output.txt", "w")
terminated = False
truncated = False

while terminated == False and truncated == False:
    action = env.action_space.sample()

    output_file.write(str(action) + '\n')

    obs, reward, terminated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close() 
output_file.close()