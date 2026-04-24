import python.approximating.approximator as ap


class CoordinatorException(Exception):
    def __init__(self, rank: int):
        self.rank = rank
        super().__init__(f"Exception from process {rank}")


class Settings:
    def __init__(self, rng, point_count, matrix_count, dimension, perturb):
        self.rng = rng
        self.point_count = point_count
        self.matrix_count = matrix_count
        self.dimension = dimension
        self.total_dimension = (dimension, dimension, dimension)
        self.perturb = perturb

class Coordinator:
    def __init__(self, comm, settings):
        self.comm = comm
        self.settings = settings
        self.worker_settings = []
        
    def _split(self, rng):
        num_processes = self.comm.Get_size()
        chunk_size = self.settings.matrix_count // num_processes
        remainder = self.settings.matrix_count % num_processes
        
        generators = rng.spawn(num_processes)
        worker_settings = []
        for i in range(num_processes):
            matrix_count_i = chunk_size + (1 if i < remainder else 0)
            worker_settings.append(Settings(generators[i], self.settings.point_count, matrix_count_i, self.settings.dimension, self.settings.perturb))
        return worker_settings
    
    def _approximate(self, settings, scaled_system, scaled_solutions):
        try:
            approximation = ap.approximate(settings.dimension, settings.perturb, settings.point_count, settings.matrix_count,
                                        settings.rng, scaled_system, scaled_solutions)
            return approximation * (settings.point_count * settings.matrix_count)
        except Exception as e:
            return e

    def reset(self, rng):
        self.worker_settings = self._split(rng)
        for i in range(1, self.comm.Get_size()):
            self.comm.send('reset', dest=i, tag=1)
            self.comm.send(self.worker_settings[i], dest=i, tag=2)
    
    def approximate(self, scaled_system, scaled_solutions):
        for i in range(1, self.comm.Get_size()):
            self.comm.send('approximate', dest=i, tag=1)
        self.comm.bcast((scaled_system, scaled_solutions), root=0)
        coordinator_settings = self.worker_settings[0]
        approximation = self._approximate(coordinator_settings, scaled_system, scaled_solutions)
        all_approximations = self.comm.gather(approximation, root=0)
        for i, approximation in zip(range(self.comm.Get_size()), all_approximations):
            if isinstance(approximation, Exception):
                raise CoordinatorException(i) from approximation
        return sum(all_approximations) / (self.settings.point_count * self.settings.matrix_count)
    
    def close(self):
        for i in range(1, self.comm.Get_size()):
            self.comm.send('close', dest=i, tag=1)
    
class Worker:
    def __init__(self, comm):
        self.comm = comm
        
    def _approximate(self, settings, scaled_system, scaled_solutions):
        try:
            approximation = ap.approximate(settings.dimension, settings.perturb, settings.point_count, settings.matrix_count,
                                        settings.rng, scaled_system, scaled_solutions)
            return approximation * (settings.point_count * settings.matrix_count)
        except Exception as e:
            return e
        
    def work(self):
        settings = None
        while True:
            command = self.comm.recv(source=0, tag=1)
            if command == 'reset':
                settings = self.comm.recv(source=0, tag=2)
            elif command == 'approximate':
                scaled_system, scaled_solutions = self.comm.bcast(None, root=0)
                approximation = self._approximate(settings, scaled_system, scaled_solutions)
                self.comm.gather(approximation, root=0)                
            elif command == 'close':
                break