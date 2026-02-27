import gymnasium as gym
import math
import networkx as nx
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
            self.graph_size, self.graph_matrix = self._create_graph_matrix(size)
        else:
            self.graph_size, self.graph_matrix = self._read_graph(graph_path)
        
        self.matrix_systems, self.matrix_indices = self._create_matrix_systems()
        self.perturb = perturb
        self.point_count = point_count
        self.matrix_count = matrix_count
        self.render_mode = render_mode

        self.space_matrix = np.triu(self.graph_matrix)
        self.observation_space = spaces.Box(-1 * self.space_matrix, 1 * self.space_matrix)
        self.action_space = spaces.Box(-action_limit * self.space_matrix, action_limit * self.space_matrix)
        
        self._agent_location = np.zeros(self.graph_matrix.shape, dtype=np.float32)
        
    def _create_graph_matrix(self, n):
        p = n * math.log2(n)
        p /= (n * (n - 1))
        graph = nx.fast_gnp_random_graph(n, p)
        nx.write_adjlist(graph, self.graph_path)
        return n, nx.to_numpy_array(graph).astype(np.float32)
    
    def _read_graph(self, file):
        graph = nx.read_adjlist(file)
        return len(graph.nodes), nx.to_numpy_array(graph).astype(np.float32)
    
    def _create_matrix_systems(self):
        system_size = 2 * self.graph_size
        matrix_systems = np.zeros((system_size, system_size, system_size), dtype=np.float32)
        matrix_systems[0, 0, 0] = 1
        
        for i in range(self.graph_size, system_size):
            matrix_systems[i, i - self.graph_size, i - self.graph_size] = 1
            matrix_systems[i, i, i] = 1
            
        matrix_indices = {}
        for i in range(1, self.graph_size):
            observation_indices1 = []
            observation_indices2 = []
            matrix_indices1 = []
            matrix_indices2 = []
            for j in range(0, self.graph_size):
                if i == j or self.graph_matrix[i, j] == 0:
                    continue
                observation_indices1.append(i)
                observation_indices2.append(j)
                
                matrix_indices1.append(i)
                matrix_indices2.append(j + self.graph_size)
                
                observation_indices1.append(j)
                observation_indices2.append(i)
                
                matrix_indices1.append(j)
                matrix_indices2.append(i + self.graph_size)
            
            matrix_indices[i] = np.array(observation_indices1, dtype=np.int32), np.array(observation_indices2, dtype=np.int32), \
                np.array(matrix_indices1, dtype=np.int32), np.array(matrix_indices2, dtype=np.int32)
            
        return matrix_systems, matrix_indices
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._agent_location = self.np_random.uniform(-1, 1, self.space_matrix.shape).astype(np.float32) * self.space_matrix
        return self._agent_location, {}
    
    def step(self, action):
        self._agent_location += action
        self._agent_location = np.clip(self._agent_location, -1, 1, out=self._agent_location)
        
        location = self._agent_location + self._agent_location.T
        for i in range(1, self.graph_size):
            observation_indices1, observation_indices2, matrix_indices1, matrix_indices2 = self.matrix_indices[i]
            self.matrix_systems[i, matrix_indices1, matrix_indices2] = location[observation_indices1, observation_indices2]
            self.matrix_systems[i, matrix_indices2, matrix_indices1] = location[observation_indices2, observation_indices1]
        
        reward = 0
        terminated = False
        try:
            decomposed_matrices = np.linalg.cholesky(self.matrix_systems)
            scaled_system, scaled_solutions = ap.scale_system(decomposed_matrices)
            reward = ap.approximate(len(self.matrix_systems), self.perturb, self.point_count, self.matrix_count, self.np_random, scaled_system, scaled_solutions)
        except ValueError:
            reward = -100
            terminated = True
        return self._agent_location, reward, terminated, False, {}
    
gym.register(id="EllipseSystemEnv-v0", entry_point=EllipseSystemEnv, max_episode_steps=20)
gym.register(id="PowerFlowSystemEnv-v0", entry_point=PowerFlowSystemEnv, max_episode_steps=20)