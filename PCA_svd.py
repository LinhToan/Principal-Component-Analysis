# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:35:09 2020

@author: Hiro
"""

"""
This program will run principal component analysis using the SVD method.
"""

from scipy.linalg import svd
import numpy as np
from numpy import linalg
import math as m

def SVD(Y):
    X = np.array(Y)
    m, n = X.shape
    k = min(m,n)
    tol = 0.9 # set tolerance to 90%

    for i in range(n): # mean center the X matrix
        X[:,i] = X[:,i] - np.mean(X[:,i])

    U, sig, VT = svd(X) # run SVD on X
    A = np.matmul(X.T, X) # Store X.T * X into A
    E = np.linalg.eigh(A) # Find eigenvalues and eigenvectors of A
    vec = E[1] # stores the eigenvectors of A

    sigMat = np.zeros_like(X, dtype=float) # initialize sigma matrix

    sigMat[:len(sig), :len(sig)] = np.diag(sig) # create sigma matrix

    # Create a square submatrix of sigMat. We don't need to check for square
    # dimensions because this if-else will force the matrix to be square.
    if k == m:
        sigMat = sigMat[:,0:m]
    else:
        sigMat = sigMat[0:n,:]

    sigSum = 0 # initialize sum of singular values

    for i in range(k):
        sigSum += (np.sum(sigMat[i]) ** 2) # calculate sum of square of singular values

    g = 0 # initialize g_j variable
    i = 0 # intialize counter sigMatrix index

    while g < tol:
        g += (np.sum(sigMat[i]) ** 2) / sigSum
        i += 1

    T = np.zeros_like(X, dtype=float) # initialize matrix of representative data set
    T = np.matmul(X, vec[:,:i]) # multiplies X and V_j
    return T