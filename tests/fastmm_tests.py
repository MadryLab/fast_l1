from fast_l1.fastmm import selective_addmm
import torch as ch

X = ch.randn(10, 1000)
X_buf = ch.zeros_like(X)

A = ch.randn(10, 10_000)
B = ch.randn(10_000, 1000)
C = ch.zeros(10, 1000)

B_buf = ch.zeros_like(B)
C_buf = ch.zeros_like(C)

inds = (ch.rand(1000) > 0.5)

selective_addmm(X, A, B, C, X_buf, B_buf, C_buf, inds, -1)
print(C)

C[:] = 0
ch.addmm(X, A, B, out=C, beta=-1.)
print(C)
