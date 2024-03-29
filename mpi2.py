from mpi4py import MPI
import numpy as np
import time


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_elements_list = [10, 1000, 10000000]

    for num_elements in num_elements_list:
        comm.Barrier()

        if rank == 0:
            start_time = time.time()
            data = np.arange(num_elements, dtype=np.int64)
            counts = [num_elements // size + (1 if i < (num_elements % size) else 0) for i in range(size)]
            displacements = [sum(counts[:i]) for i in range(size)]
        else:
            data = None
            counts = None
            displacements = None
            start_time = 0

        start_time = comm.bcast(start_time, root=0)
        counts = comm.bcast(counts, root=0)
        displacements = comm.bcast(displacements, root=0)

        chunk_size = counts[rank]
        chunk = np.zeros(chunk_size, dtype=np.int64)

        comm.Scatterv([data, counts, displacements, MPI.INT64_T], chunk, root=0)

        local_sum = np.sum(chunk)

        total_sum = comm.reduce(local_sum, root=0, op=MPI.SUM)

        if rank == 0:
            end_time = time.time()
            print(f"Total sum of {num_elements} elements: {total_sum}")
            print(f"Execution time: {end_time - start_time} seconds\n")


if __name__ == '__main__':
    main()