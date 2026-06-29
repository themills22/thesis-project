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
                self.matrix_systems[i, j + self.graph_size, i] = -value
                self.matrix_systems[i, i, j + self.graph_size] = -value
                self.matrix_systems[i, j, i + self.graph_size] = value

        for value, edge in zip(new_location, self.sorted_edges):
            i, j = edge
            i, j = int(i), int(j)
            value /= 2
            set_value(i, j, value)
            set_value(j, i, value)

    @staticmethod
    def _weighted_sum(matrices, weights):
        weights = np.asarray(weights, dtype=np.float64)
        return np.tensordot(weights, matrices, axes=(0, 0))

    @staticmethod
    def _make_psd(matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        matrix = (matrix + matrix.T) / 2.0

        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        min_eigenvalue = float(np.min(eigenvalues))
        shift = max(1e-6, -min_eigenvalue + 1e-6)
        psd_matrix = matrix + shift * np.eye(matrix.shape[0], dtype=np.float64)
        return psd_matrix.astype(np.float32)

    def to_psd_system(self):
        """Build a positive-definite system consistent with the paper's construction.

        The raw system stores the $Q_k$ blocks in the first half of the stack and the
        $S_k$ blocks in the second half. Following the paper, the first half of the
        transformed system is built from the $S_k$ blocks, while the second half uses
        a combination of the $S_k$ and $Q_k$ blocks plus a minimal diagonal shift so
        that each resulting matrix is positive definite.
        """
        q_matrices = np.asarray(self.matrix_systems[:self.graph_size], dtype=np.float64)
        s_matrices = np.asarray(self.matrix_systems[self.graph_size:], dtype=np.float64)

        psd_system = np.zeros_like(self.matrix_systems, dtype=np.float32)

        for k in range(self.graph_size):
            alpha = np.full((self.graph_size,), 1.0, dtype=np.float64)
            alpha[k] += 0.25
            psd_system[k] = self._make_psd(self._weighted_sum(s_matrices, alpha))

        for k in range(self.graph_size):
            beta = np.full((self.graph_size,), 0.25, dtype=np.float64)
            gamma = np.zeros((self.graph_size,), dtype=np.float64)
            gamma[k] = 1.0
            base_matrix = self._weighted_sum(s_matrices, beta) + self._weighted_sum(q_matrices, gamma)
            psd_system[self.graph_size + k] = self._make_psd(base_matrix)

        return psd_system