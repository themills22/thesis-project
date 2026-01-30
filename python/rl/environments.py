import gymnasium as gym
import numpy as np
import python.approximating.approximator as ap

from gymnasium import spaces

class EllipseSystemEnv(gym.Env):
    
    def __init__(self, dimension, perturb, point_count, matrix_count, render_mode=None, action_limit=1.0):
        self.dimension = dimension
        self.total_dimension = (dimension, dimension, dimension)
        self.perturb = perturb
        self.point_count = point_count
        self.matrix_count = matrix_count
        self.render_mode = render_mode
        
        self.observation_space = spaces.Box(-1, 1, self.total_dimension)
        self.action_space = spaces.Box(-action_limit, action_limit, self.total_dimension)
        
        self._agent_location = np.zeros(self.total_dimension, dtype=np.float32)
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._agent_location = self.np_random.uniform(-1, 1, self.total_dimension).astype(np.float32)
        return self._agent_location, {}
    
    def step(self, action):
        self._agent_location += action
        self._agent_location
        self._agent_location = np.clip(self._agent_location, -1, 1, out=self._agent_location)
        
        reward = 0
        terminated = False
        try:
            scaled_system, scaled_solutions = ap.scale_system(self._agent_location)
            reward = ap.approximate(self.dimension, self.perturb, self.point_count, self.matrix_count, self.np_random, scaled_system, scaled_solutions)
            # reward = self.np_random.normal(0, 1) # TODO: this is where the approximator goes
        except ValueError:
            reward = -100
            terminated = True
        return self._agent_location, reward, terminated, False, {}
    
gym.register(id="EllipseSystemEnv-v0", entry_point=EllipseSystemEnv, max_episode_steps=20)