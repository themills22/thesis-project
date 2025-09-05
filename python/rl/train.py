import numpy as np
import gymnasium as gym
import python.rl.environments
import rl_zoo3.train as train

from stable_baselines3 import td3
from stable_baselines3.common.noise import NormalActionNoise

def main():
    train.train()
    # environment = gym.make('EllipseSystemEnv-v0', max_episode_steps=1000, dimension=10, action_limit=0.05)
    # policy_kwargs = {
    #     'net_arch' : {
    #         'pi' : [1000, 1000],
    #         'qf' : [1000, 1000]
    #     }
    # }
    # action_noise = NormalActionNoise(np.zeros(environment.unwrapped.total_dimension), np.full(environment.unwrapped.total_dimension, 0.05))
    # model = td3.TD3('MlpPolicy', environment, action_noise=action_noise, policy_kwargs=policy_kwargs)
    # model.learn(20)
    
if __name__ == '__main__':
    main()