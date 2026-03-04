import numpy as np

class PowerFlowMatrices:
    def __init__(self, graph_size, sorted_edges):
        self.graph_size = graph_size
        self.sorted_edges = sorted_edges
        
        self.system_size = 2 * graph_size
        self.matrix_systems = np.zeros((self.system_size, self.system_size, self.system_size), dtype=np.float32)
        self.matrix_systems[0, 0, 0] = 1
        
        for i in range(self.graph_size, self.system_size):
            self.matrix_systems[i, i - self.graph_size, i - self.graph_size] = 1
            self.matrix_systems[i, i, i] = 1
    
    def update(self, new_location):
        def set_value(i, j, value):
            if i != 0:
                self.matrix_systems[i, i + self.graph_size, j] = value
                self.matrix_systems[i, j + self.graph_size, i] = value
                self.matrix_systems[i, i, j + self.graph_size] = value
                self.matrix_systems[i, j, i + self.graph_size] = value
                
        for value, edge in zip(new_location, self.sorted_edges):
            i, j = edge
            set_value(i, j, value)
            set_value(j, i, value)