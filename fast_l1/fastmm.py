from numba import cuda, float32
import torch as ch
import numpy as np
from numba.cuda import as_cuda_array, synchronize

TB = 16

@cuda.jit
def copy_buffers(B, B_buff, indices):
    x, y = cuda.grid(2)
    if y > indices.shape[0] or x > B.shape[0]:
        return
    B_buff[x, y] =  B[x, indices[y]]

@cuda.jit
def copy_back(C, C_buff, indices):
    x, y = cuda.grid(2)
    if y > indices.shape[0] or x > C.shape[0]:
        return
    C[x, indices[y]] =  C_buff[x, y]

def extract_columns(B, B_buff, indices):
    grid_size = (
            (B.shape[0] - 1) // TB + 1,
            (len(indices) - 1) // TB + 1,
    )
    copy_buffers[grid_size, (TB, TB)](as_cuda_array(B), as_cuda_array(B_buff), as_cuda_array(indices))

def write_columns(C, C_buff, indices):
    grid_size = (
            (C.shape[0] - 1) // TB + 1,
            (len(indices) - 1) // TB + 1,
    )
    copy_back[grid_size, (TB, TB)](as_cuda_array(C), as_cuda_array(C_buff), as_cuda_array(indices))

def selective_matmul(A, B, C, B_buff, C_buff, indices):
    indices = ch.where(indices)[0]
    extract_columns(B, B_buff, indices)
    ch.matmul(A, B[:, :len(indices)], out=C_buff[:, :len(indices)])
    write_columns(C, C_buff, indices)

def selective_admm(inp, A, B, C, inp_buff, B_buff, C_buff, indices):
    indices = ch.where(indices)[0]
    extract_columns(B, B_buff, indices)
    extract_columns(inp, inp_buff, indices)
    n = len(indices)
    ch.admm(inp_buff[:, :n], A, B[:, :n], out=C_buff[:, :n])
    write_columns(C, C_buff, indices)