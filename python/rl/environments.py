import gymnasium as gym
import math
import networkx as nx
import numpy as np
import python.approximating.approximator as ap
import time

from gymnasium import spaces
from python.rl.power_flow_matrices import PowerFlowMatrices

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
        self._agent_location = np.clip(self._agent_location, -1, 1, out=self._agent_location)
        
        reward = 0
        terminated = False
        try:
            scaled_system, scaled_solutions = ap.scale_system(self._agent_location)
            reward = ap.approximate(self.dimension, self.perturb, self.point_count, self.matrix_count, self.np_random, scaled_system, scaled_solutions)
        except ValueError:
            reward = -100
            terminated = True
        return self._agent_location, reward, terminated, False, {}
    
    
class PowerFlowSystemEnv(gym.Env):
    
    def __init__(self, graph_path, perturb, point_count, matrix_count, action_limit, render_mode=None, size=None):
        self.graph_path = graph_path
        if size is not None:
            self.graph = self._create_graph_matrix(size)
        else:
            self.graph = nx.read_adjlist(self.graph_path)
        
        self.sorted_edges = sorted([edge for edge in self.graph.edges])
        self.matrices = PowerFlowMatrices(self.graph_size, self.sorted_edges)
        self.perturb = perturb
        self.point_count = point_count
        self.matrix_count = matrix_count
        self.render_mode = render_mode

        self.observation_space = spaces.Box(-1, 1, (len(self.sorted_edges),))
        self.action_space = spaces.Box(-action_limit, action_limit, (len(self.sorted_edges),))
        
        self._agent_location = np.zeros(self.observation_space.shape, dtype=np.float32)
        
    def _create_graph_matrix(self, n):
        p = n * math.log2(n)
        p /= (n * (n - 1))
        graph = nx.fast_gnp_random_graph(n, p)
        nx.write_adjlist(graph, self.graph_path)
        return graph 
    
    @property
    def graph_size(self):
        return len(self.graph)
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._agent_location = self.np_random.uniform(-1, 1, self.observation_space.shape).astype(np.float32)
        self.matrices.update(self._agent_location)
        return self._agent_location, {}
    
    def step(self, action):
        self._agent_location = np.clip(self._agent_location + action, -1, 1)
        self.matrices.update(self._agent_location)
        
        reward = 0
        terminated = False
        try:
            scaled_system, scaled_solutions = ap.scale_system(self.matrices.matrix_systems)
            reward = ap.approximate(self.matrices.system_size, self.perturb, self.point_count, self.matrix_count, self.np_random, scaled_system, scaled_solutions)
        except ValueError as e:
            print(e)
            reward = -100
            terminated = True
        return self._agent_location, reward, terminated, False, {}
    
gym.register(id="EllipseSystemEnv-v0", entry_point=EllipseSystemEnv, max_episode_steps=20)
gym.register(id="PowerFlowSystemEnv-v0", entry_point=PowerFlowSystemEnv, max_episode_steps=20)