from mpi4py import MPI
import numpy as np
import time


def solve_gaussian_elimination(A, b, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    n = len(b)

    # Forward elimination
    for i in range(n):
        # Root process normalizes leader-row
        if i % size == rank:
            pivot_row = A[i] / A[i, i]
            pivot_val = b[i] / A[i, i]
            comm.Bcast(pivot_row, root=rank)
            comm.bcast(pivot_val, root=rank)

            # Update rows
            for j in range(i + 1, n):
                if j % size == rank:
                    A[j] -= pivot_row * A[j, i]
                    b[j] -= pivot_val * A[j, i]
        else:
            pivot_row = np.empty(n, dtype=np.float64)
            pivot_val = None
            comm.Bcast(pivot_row, root=i % size)
            pivot_val = comm.bcast(pivot_val, root=i % size)

            # Update rows
            for j in range(i + 1, n):
                if j % size == rank:
                    A[j] -= pivot_row * A[j, i]
                    b[j] -= pivot_val * A[j, i]

    # Backward substitution
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        if i % size == rank:
            x[i] = b[i] / A[i, i]
        x[i] = comm.bcast(x[i], root=i % size)

        for j in range(i):
            if j % size == rank:
                b[j] -= A[j, i] * x[i]

    return x


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Matrix sizes for the experiment
    n_values = [10, 100, 500, 1000, 2000]

    for n in n_values:
        if rank == 0:
            # Create a random system of linear equations
            A = np.random.rand(n, n)
            b = np.random.rand(n)
            start_time = time.time()
        else:
            A = np.empty((n, n), dtype=np.float64)
            b = np.empty(n, dtype=np.float64)

        comm.Bcast(A, root=0)
        comm.Bcast(b, root=0)

        x = solve_gaussian_elimination(A, b, comm)

        if rank == 0:
            end_time = time.time()
            print(f"System size: {n}, number of processes: {size}")
            print(f"Solving time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    main()