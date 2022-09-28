"""
clustering evaluation measures library
"""
import numpy as np
import math

def ARI(labels1, labels2):

    dim1 = labels1.ndim
    dim2 = labels2.ndim
    n1 = labels1.shape[0]
    n2 = labels2.shape[0]
    if dim1 > 1 or dim2 > 1 or n1 != n2:
        print('Error in ARI!\n')
        return -1

    N = n1

    min1 = min(labels1)
    min2 = min(labels2)
    # make labels start from 0 if they are started from 1
    if min1 == 1:
        labels1 = labels1 - 1
    if min2 == 1:
        labels2 = labels2 - 1

    K1 = max(labels1) + 1
    K2 = max(labels2) + 1
    m1 = np.unique(labels1)
    m2 = np.unique(labels2)

    # if empty clusters, just fix it
    if m1.shape[0] != K1:
        for i in range(m1.shape[0]):
            labels1[labels1 == m1[i]] = i # start from 1 not zero

    if m2.shape[0] != K2:
        for i in range(m2.shape[0]):
            labels2[labels2 == m2[i]] = i # start from 1 not zero

    K1 = m1.shape[0]
    K2 = m2.shape[0]

    # Contingency matrix
    cntg_mat = np.zeros([K2, K1], dtype=float)
    for i in range(N):
        cntg_mat[labels2[i]][labels1[i]] = cntg_mat[labels2[i]][labels1[i]] + 1

    ni = np.sum((np.sum(cntg_mat, axis=1))**2).astype(float)
    nj = np.sum((np.sum(cntg_mat, axis=0))**2).astype(float)

    t1 = float(N) * (N - 1) / 2
    t2 = np.sum(cntg_mat**2) # sum(nij ^ 2)
    t3 = 0.5 * (ni + nj)

    # Expected value of RI
    nc = (float(N) * (N*N + 1) - (N + 1) * ni - (N + 1) * nj + (2 * (ni * nj)) / N) / (2 * float(N - 1))
    p1 = N * (N*N + 1)
    p2 = (N + 1) * ni
    p3 = (N + 1) * nj
    p4 = (2 * (ni * nj)) / N
    p5 = (2 * (N - 1))

    q1 = p1 - p2
    q2 = q1 - p3
    q3 = q2 + p4
    q4 = q3 / p5


    A = t1 + t2 - t3; # agreements
    RI = A / t1

    if t1 == nc:
        ARI = 0
    else:
        ARI = (A - nc) / (t1 - nc)

    return ARI
