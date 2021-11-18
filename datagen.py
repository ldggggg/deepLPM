#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:56:03 2021

@author: marco
"""
import numpy as np
import random

N = 500
Adj = np.zeros((N,N))
Z = np.random.choice([0,1], size = N)
Pi = np.matrix([[0.9, 0.2],[0.2, 0.9]])

# ix = [(row, col) for row in range(N) for col in range(N)]
# for row, col in random.sample(ix, int(round(0.5*len(ix)))):
#     u = np.random.uniform()
#     if u < Pi[Z[row], Z[col]]:
#         Adj[row, col] = 1.
#         Adj[row, col] = 1.

for idx in range(N-1):
    for idy in range(idx,N):
        u = np.random.uniform()
        if u < Pi[Z[idx], Z[idy]]:
            Adj[idx, idy] = 1.
            Adj[idy, idx] = 1.

row, col = np.diag_indices_from(Adj)
Adj[row, col] = 1.

np.savetxt('Adj3.txt', Adj)
