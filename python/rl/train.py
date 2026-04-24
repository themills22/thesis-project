import python.rl.environments
import rl_zoo3.train as train

from mpi4py import MPI
from python.approximating.approximate import Worker

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        train.train()
    else:
        worker = Worker(comm)
        worker.work()