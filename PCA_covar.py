# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:32:15 2020

@author: Linh
"""

"""
This program will run principal component analysis using the covariance method.
"""

from scipy.linalg import svd
import numpy as np
from numpy import linalg

def covar(Y):
    X = np.array(Y)
    m, n = X.shape
    tol = 0.9 # set tolerance to 90%

    for i in range(n): # mean center the X matrix
        X[:,i] = np.mean(X[:,i]) - X[:,i]

    A = np.matmul(X.T, X) # Store X.T * X into A
    vals, vecs = np.linalg.eigh(A) # Find eigenvalues and eigenvectors of A

    eigSum = np.sum(vals) # initialize sum of eigenvalues

    i = -1 # start index from right to left
    g = vals[-1] / eigSum # initialize g to adjust for starting from right to left instead of left to right

    while g < tol: # calculates which eigenvalues to use
        g += (vals[i]) / eigSum
        i -= 1

    T = np.zeros_like(A, dtype=float) # initialize matrix of representative data set
    T = np.matmul(X, vecs[:,i:][:,::-1]) # multiplies X and V_j
    return T