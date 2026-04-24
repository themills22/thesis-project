import gymnasium as gym
import networkx as nx
import numpy as np
import python.approximating.approximator as ap

from python.approximating.mpi_approximator import Coordinator, Settings, CoordinatorException
from mpi4py import MPI
from gymnasium import spaces
from python.rl.power_flow_matrices import PowerFlowMatrices

class EllipseSystemEnv(gym.Env):
    
    def __init__(self, dimension, perturb, point_count, matrix_count, render_mode=None, action_limit=1.0):
        self.render_mode = render_mode
        self.approximation_settings = Settings(None, point_count, matrix_count, dimension, perturb)
        self.coordinator = Coordinator(MPI.COMM_WORLD, self.approximation_settings)
        
        self.observation_space = spaces.Box(-1, 1, self.approximation_settings.total_dimension)
        self.action_space = spaces.Box(-action_limit, action_limit, self.approximation_settings.total_dimension)
        
        self._agent_location = np.zeros(self.approximation_settings.total_dimension, dtype=np.float32)
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.coordinator.reset(self.np_random)
        self._agent_location = self.np_random.uniform(-1, 1, self.approximation_settings.total_dimension).astype(np.float32)
        return self._agent_location, {}
    
    def step(self, action):
        self._agent_location += action
        self._agent_location = np.clip(self._agent_location, -1, 1, out=self._agent_location)
        
        reward = 0
        terminated = False
        try:
            scaled_system, scaled_solutions = ap.scale_system(self._agent_location)
            reward = self.coordinator.approximate(scaled_system, scaled_solutions)
        except ValueError:
            reward = -100
            terminated = True
        except CoordinatorException:
            reward = -100
            terminated = True
        return self._agent_location, reward, terminated, False, {}
    
    def close(self):
        self.coordinator.close()
        super().close()    
    
class PowerFlowSystemEnv(gym.Env):
    
    def __init__(self, graph_path, perturb, point_count, matrix_count, action_limit, render_mode=None):
        self.graph_path = graph_path
        self.graph = nx.read_adjlist(self.graph_path)
        self.sorted_edges = sorted([edge for edge in self.graph.edges])
        self.matrices = PowerFlowMatrices(self.graph_size, self.sorted_edges)
        self.approximation_settings = Settings(None, point_count, matrix_count, self.matrices.system_size, perturb)
        self.coordinator = Coordinator(MPI.COMM_WORLD, self.approximation_settings)
        self.render_mode = render_mode

        self.observation_space = spaces.Box(-1, 1, (len(self.sorted_edges),))
        self.action_space = spaces.Box(-action_limit, action_limit, (len(self.sorted_edges),))
        
        self._agent_location = np.zeros(self.observation_space.shape, dtype=np.float32)
    
    @property
    def graph_size(self):
        return len(self.graph)
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.coordinator.reset(self.np_random)
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
            reward = self.coordinator.approximate(scaled_system, scaled_solutions)
        except ValueError:
            reward = -100
            terminated = True
        except CoordinatorException:
            reward = -100
            terminated = True
        return self._agent_location, reward, terminated, False, {}
    
    def close(self):
        self.coordinator.close()
        super().close()    
    
gym.register(id="EllipseSystemEnv-v0", entry_point=EllipseSystemEnv)
gym.register(id="PowerFlowSystemEnv-v0", entry_point=PowerFlowSystemEnv)