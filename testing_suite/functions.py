import time
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random as rand
from itertools import product
from scipy.optimize import curve_fit


def beta(G): # G[j'th point, nth coord]
    return np.array(
        [
            2*G[:,0] - 3*G[:,0]**3 / (2*math.pi*(G[:,1] + G[:,0])**3), 
            -G[:,0]**2*(7*G[:,1] + G[:,0]) / (4*math.pi*(G[:,1] + G[:,0])**3)
        ]
    ) # beta[jth point, nth function] [!] this line is specifically 2D


def F_fp(g): # Calculate c_fp[n',mu]*g[n'] (n' means (gn, 1))
    c_fp = np.array(
        [
            [172 - 137*43**0.5], 
            [215 + 44*43**0.5], 
            [49*(1003*43**0.5 - 989)/(288*math.pi)]
        ]
    ) # N+1 x N-M coefficients c_fp[n',mu] [!] this line is specifically 2D
    return np.matmul(
        np.append(g, [1]),
        c_fp
    ) # F_fp [!] this line is specifically 2D


def psi(G, N_cp, sigma, N, CP_transpose): # G[j'th point, nth coord]
    # divide(1, (1 + 1/sigma**2 * matmul(1, square([g,...,g]^T - CP^T)))) for all g in G
    return np.array(
        [
            np.divide(
                np.ones(N_cp),
                (np.ones(N_cp) + (1/sigma**2)*np.matmul(
                    np.ones(N),
                    np.square(
                        np.tile(g, (N_cp,1)).transpose() - CP_transpose
                    )
                )
                )
            ) for g in G
        ]
    ) # psi[jth point,ith function]


def dpsi(G, N_cp, sigma, N, CP_transpose, CP): # G[j'th point, nth coord]
    # multiply(-2/sigma**2 * ([g,...,g]^T - CP^T) for all g in G, (tile(square(psi(CP)),(N,1,1)), axes=[1,0,2])^T)
    return np.multiply(
        np.array(
            [
                (
                    (-2/sigma**2) * (np.tile(g, (N_cp,1)).transpose() - CP_transpose)
                ) for g in G
            ]
        ), 
        np.transpose(
            np.tile(np.square(psi(CP, N_cp, sigma, N, CP_transpose)),(N,1,1)), 
            axes=[1,0,2]
        )
    ) # dpsi[jth point, nth derivative,ith function]