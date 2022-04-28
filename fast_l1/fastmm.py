import torch as ch


def extract_columns(B, B_buff, indices):
    B_buff[:, :len(indices)] = B[:, indices]


def write_columns(C, C_buff, indices):
    C.index_copy_(1, indices, C_buff[:, :len(indices)])


def selective_matmul(A, B, C, B_buff, C_buff, indices):
    indices = ch.where(indices)[0]
    extract_columns(B, B_buff, indices)
    ch.matmul(A, B_buff[:, :len(indices)], out=C_buff[:, :len(indices)])
    write_columns(C, C_buff, indices)


def selective_addmm(inp, A, B, C, inp_buff, B_buff, C_buff, indices, beta):
    indices = ch.where(indices)[0]
    extract_columns(B, B_buff, indices)
    extract_columns(inp, inp_buff, indices)
    n = len(indices)
    ch.addmm(input=inp_buff[:, :n], mat1=A,
             mat2=B_buff[:, :n], out=C_buff[:, :n], beta=beta)
    write_columns(C, C_buff, indices)
