import gymnasium as gym
import numpy as np
from constants import *

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.old_info = None  # Stores previous step info
        self.old_direction = None
        self.mid_position = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.old_info = None  # Clear on reset
        self.mid_position = info.get('labels', None)['player_y'] # Get the starting point of the paddle as the mid point
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward = self.modify_reward(obs, info, self.old_info, terminated)

        self.old_info = info.copy()

        return obs, reward, terminated, truncated, info

    def modify_reward(self, obs, info, old_info, terminated):
        reward = 0

        if old_info == None:
            return reward

        # Get the labels if they exist
        labels = info.get('labels', None)
        old_labels = old_info.get('labels', None)

        # Check if anything reward-worthy happened
        if self.scored(labels, old_labels):
            reward += SCORE_REWARD
        # if self.opponent_scored(labels, old_labels):
        #     return -SCORE_REWARD
        if self.returned_ball(labels, old_labels):
            reward += BOUNCE_REWARD
        # if self.big_movement(labels, old_labels):
        #     return BIG_MOVE_REWARD
        if labels['player_score'] == 21:
            reward += WIN_REWARD

        # if labels['player_y'] != old_labels['player_y']:
        #     reward += MOVE_REWARD 

        if np.abs(labels['player_y']-self.mid_position) > 40:
            reward += AT_THE_EDGE
        
        return reward

    """
    Checks if the ball was rebounded from the player side
    """
    def returned_ball(self, labels, old_labels):
        if labels == None or old_labels == None:
            return False
    
        ball_dirn = int(old_labels['ball_x']) - int(labels['ball_x'])

        if old_labels['ball_x'] != 0:
            if self.old_direction < 0 and ball_dirn > 0:
                self.old_direction = ball_dirn
                return True

        self.old_direction = ball_dirn
        return False
        
    """
    Checks if the player scored
    """
    def scored(self, labels, old_labels):
        if labels == None or old_labels == None:
            return False

        if labels['player_score'] > old_labels['player_score']:
            return True
        else:
            return False
        
    """
    Checks if the player moved a lot, trying to smoothen the movements
    """
    def big_movement(self, labels, old_labels):
        if np.abs(labels['player_y'] - old_labels['player_y']) > 15:
            return True
        else:
            return False

    def opponent_scored(self, labels, old_labels):
        if labels['enemy_score'] > old_labels['enemy_score']:
            return True
        else:
            return False