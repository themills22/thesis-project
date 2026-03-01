import gymnasium as gym
import math
import networkx as nx
import numpy as np
import python.approximating.approximator as ap
import time

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
        self.matrix_systems = self._create_matrix_systems()
        self._decomposed_systems = np.array(self.matrix_systems)
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
    
    def _create_matrix_systems(self):
        system_size = 2 *  self.graph_size
        matrix_systems = np.zeros((system_size, system_size, system_size), dtype=np.float32)
        matrix_systems[0, 0, 0] = 1
        
        for i in range(self.graph_size, system_size):
            matrix_systems[i, i - self.graph_size, i - self.graph_size] = 1
            matrix_systems[i, i, i] = 1
            
        return matrix_systems
    
    def _update_observation(self, new_location):
        def set_value(i, j, value):
            if i != 0:
                self.matrix_systems[i, i + self.graph_size, j] = value
                self.matrix_systems[i, j + self.graph_size, i] = value
                self.matrix_systems[i, i, j + self.graph_size] = value
                self.matrix_systems[i, j, i + self.graph_size] = value
                
        new_location = np.clip(new_location, -1, 1)
        for value, edge in zip(new_location, self.sorted_edges):
            i, j = edge
            set_value(i, j, value)
            set_value(j, i, value)
            
        return new_location
    
    
    @property
    def graph_size(self):
        return len(self.graph)
    
    @property
    def matrix_size(self):
        return len(self.matrix_systems)
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._agent_location = self._update_observation(self.np_random.uniform(-1, 1, self.observation_space.shape).astype(np.float32))
        return self._agent_location, {}
    
    def step(self, action):
        self._agent_location = self._update_observation(self._agent_location + action)
        
        reward = 0
        terminated = False
        try:
            scaled_system, scaled_solutions = ap.scale_system(self.matrix_systems)
            reward = ap.approximate(len(self.matrix_systems), self.perturb, self.point_count, self.matrix_count, self.np_random, scaled_system, scaled_solutions)
        except ValueError as e:
            print(e)
            reward = -100
            terminated = True
        return self._agent_location, reward, terminated, False, {}
    
gym.register(id="EllipseSystemEnv-v0", entry_point=EllipseSystemEnv, max_episode_steps=20)
gym.register(id="PowerFlowSystemEnv-v0", entry_point=PowerFlowSystemEnv, max_episode_steps=20)