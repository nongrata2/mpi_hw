from mpi4py import MPI
import numpy as np
import time


def matrix_multiply(A, B, C, comm, root=0):
    rank = comm.Get_rank()
    size = comm.Get_size()

    A_rows, A_cols = A.shape
    B_rows, B_cols = B.shape

    # проверка совместности входных матриц
    assert A_cols == B_rows, "Wrong sizes of matrices"

    # проверка, правильных ли размеров матрица C
    C_rows, C_cols = C.shape
    assert C_rows == A_rows and C_cols == B_cols, "Result matrix has wrong size"

    rows_per_process = A_rows // size
    extra_rows = A_rows % size
    if rank < extra_rows:
        start_row = rank * (rows_per_process + 1)
        end_row = start_row + rows_per_process + 1
    else:
        start_row = rank * rows_per_process + extra_rows
        end_row = start_row + rows_per_process

    # рассчитываем подматрицу A для текущего процесса
    local_A = A[start_row:end_row, :]

    # выполняем локальное умножение матриц
    local_C = np.matmul(local_A, B)

    # собираем результаты умножения подматриц в матрицу C
    if rank == 0:
        C[start_row:end_row, :] = local_C
        for i in range(1, size):
            if i < extra_rows:
                proc_rows = rows_per_process + 1
            else:
                proc_rows = rows_per_process
            proc_start_row = i * rows_per_process + min(i, extra_rows)
            C[proc_start_row:proc_start_row + proc_rows, :] = comm.recv(source=i, tag=10)
    else:
        comm.send(local_C, dest=root, tag=10)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # матрицы для умножения
    matrix_size_list = [(10, 10), (100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    for size in matrix_size_list:
        A = np.random.rand(size[0], size[1])
        B = np.random.rand(size[1], size[0])
        C = np.empty((size[0], size[0]), dtype=np.float64)

        if rank == 0:
            start_time = time.time()

        matrix_multiply(A, B, C, comm)

        if rank == 0:
            end_time = time.time()
            print(f"Multiplication of matrices size {size} took {end_time - start_time} seconds")
            print("\n")


if __name__ == "__main__":
    main()
