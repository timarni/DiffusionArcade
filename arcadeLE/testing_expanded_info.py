import gymnasium
import ale_py
from atariari.benchmark.wrapper import AtariARIWrapper
from reward_wrapper import RewardWrapper
from constants import *

def main():   
    gymnasium.register_envs(ale_py)

    env = gymnasium.make("Pong-v4", render_mode="human")
    env = AtariARIWrapper(env)
    env = RewardWrapper(env)

    ##### UNCOMMENT AND CHANGE render_mode="rgb_array" TO SAVE VIDEO OF RUN
#     env = gymnasium.wrappers.RecordVideo(
#     env,
#     episode_trigger=lambda num: True,
#     video_folder="./videos",
#     name_prefix="video-",
# )

    obs = env.reset()
    total_reward = 0

    obs, reward, terminated, truncated, old_info = env.step(env.action_space.sample())

    for _ in range(1000):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
    
    env.close()

    with open("info.out", "w") as f:
        for key, value in info.items():
            if key != 'labels':
                f.write(f"{key}: {value}\n")
        for key, value in info['labels'].items():
            f.write(f"{key}: {value}\n")
        f.write(f'total reward: {total_reward}\n')

if __name__ == '__main__':
    main()