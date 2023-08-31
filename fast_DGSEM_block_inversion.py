#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 14:07:32 2023

@author: jruel
"""

"""

    "Maximum principle preserving time implicit DGSEM
    for linear scalar conservation laws"
    R.Milani, F.Renac, J.Ruel

    Appendix: Fast DGSEM block inversion

"""

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

p = 2  # Polynomial order

# -----------------------------------------------------------------------------
# Gauss-Lobatto points and weights / d_min
# -----------------------------------------------------------------------------

lobatto_points = [
    [],
    [-1, 1],
    [-1, 0, 1],
    [-1, -np.sqrt(1 / 5), np.sqrt(1 / 5), 1],
    [-1, -np.sqrt(3 / 7), 0, np.sqrt(3 / 7), 1],
    [
        -1,
        -np.sqrt(1 / 3 + 2 * np.sqrt(7) / 21),
        -np.sqrt(1 / 3 - 2 * np.sqrt(7) / 21),
        np.sqrt(1 / 3 - 2 * np.sqrt(7) / 21),
        np.sqrt(1 / 3 + 2 * np.sqrt(7) / 21),
        1,
    ],
]

lobatto_weights = [
    [],
    [1, 1],
    [1 / 3, 4 / 3, 1 / 3],
    [1 / 6, 5 / 6, 5 / 6, 1 / 6],
    [0.1, 49 / 90, 32 / 45, 49 / 90, 0.1],
    [
        1 / 15,
        (14 - np.sqrt(7)) / 30,
        (14 + np.sqrt(7)) / 30,
        (14 + np.sqrt(7)) / 30,
        (14 - np.sqrt(7)) / 30,
        1 / 15,
    ],
]

d_min = [None, 8, 24, 24 * (1 + np.sqrt(5)), 198.6, 428.8]

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def lagrange_derivative(l, xi, pts):
    """Derivative of the l-th Lagrange polynomial at the point xi"""
    n = len(pts)
    c = 1
    for i in range(0, n):
        if i != l:
            c *= pts[l] - pts[i]
    p = 0
    for i in range(0, n):
        if i != l:
            t = 1
            for j in range(0, n):
                if j != l and j != i:
                    t *= xi - pts[j]
            p += t
    return p / c


def D_matrix(p):
    """Discrete derivative matrix"""
    D = np.zeros((p + 1, p + 1))
    for k in range(0, p + 1):
        for l in range(0, p + 1):
            D[k][l] = lagrange_derivative(l, lobatto_points[p][k], lobatto_points[p])
    return D


def L_matrix(p):
    A = np.zeros((p + 1, p + 1))
    A[-1, -1] = 1
    return np.transpose(D_matrix(p)) - 1 / lobatto_weights[p][p] * A


# -----------------------------------------------------------------------------
# Diagonalization of L
# -----------------------------------------------------------------------------

L = L_matrix(p)

# Computation of L eigenvalues and eigenvectors with numpy
[eigValNp, eigVecNp] = np.linalg.eig(L)

# "Analytical" computation of L eigenvalues and eigenvectors
D = D_matrix(p)

coeff = np.zeros(p + 2)
coeff[0] = lobatto_weights[p][p]
for i in range(1, p + 2):
    coeff[i] = np.linalg.matrix_power(D, i - 1)[p, p]

Psi = np.roots(coeff)  # Eigenvalues

R = np.ones((p + 1, p + 1), dtype="complex_")  # Right eigenvectors matrix
for i in range(p + 1):
    for k in range(p):
        R[k, i] = (
            -1
            / lobatto_weights[p][p]
            * sum(
                Psi[i] ** (-l - 1) * np.linalg.matrix_power(D, l)[p, k]
                for l in range(0, p + 1)
            )
        )

print("L eigenval.with numpy: {}".format(eigValNp))
print("Semi-analytical L eigenval.: {}\n".format(Psi))

# -----------------------------------------------------------------------------
# Inversion of L2d
# -----------------------------------------------------------------------------

lbd_x = 1
lbd_y = 1
lbd = lbd_x + lbd_y

I = np.eye(p + 1)
L1d = I - 2 * lbd * L

L2d = lbd_x / lbd * np.kron(I, L1d) + lbd_y / lbd * np.kron(L1d, I)

# Diagonal block inversion with numpy

M = 1 / 2 * np.diag(lobatto_weights[p])

L2d_numpyInv = np.dot(np.linalg.inv(np.kron(M, M)), np.linalg.inv(L2d))

# Explicit diagonal block inversion

Psi_lbd = I - 2 * lbd * np.diag(Psi)

Psi2d = lbd_x / lbd * np.kron(I, Psi_lbd) + lbd_y / lbd * np.kron(Psi_lbd, I)

invR = np.linalg.inv(R)
invMR = np.dot(np.linalg.inv(M), R)

L2d_explInv = np.dot(
    np.kron(invMR, invMR), np.dot(np.linalg.inv(Psi2d), np.kron(invR, invR))
)

print(
    "Verification of diag. block inv. (difference inf. norm between the two methods): {}\n".format(
        np.linalg.norm(L2d_numpyInv - L2d_explInv)
    )
)

# -----------------------------------------------------------------------------
# Inversion de L2dv
# -----------------------------------------------------------------------------

L2d0 = L2d + 2 * d_min[p] * lbd * np.kron(I, I)
Uv = np.concatenate(
    (
        lbd_x * np.kron(I, np.array(lobatto_weights[p]).reshape((p + 1, 1))),
        lbd_y * np.kron(np.array(lobatto_weights[p]).reshape((p + 1, 1)), I),
    ),
    axis=1,
)
Vv = np.concatenate(
    (np.kron(I, np.ones((p + 1, 1))), np.kron(np.ones((p + 1, 1)), I)), axis=1
)

L2dV = L2d0 - np.dot(Uv, np.transpose(Vv))

# Diagonal block inversion with numpy

L2dV_numpyInv = np.dot(np.linalg.inv(np.kron(M, M)), np.linalg.inv(L2dV))

# Explicit diagonal block inversion

invL2d0 = np.dot(
    np.kron(R, R),
    np.dot(
        np.linalg.inv(Psi2d + 2 * d_min[p] * lbd * np.kron(I, I)), np.kron(invR, invR)
    ),
)

invROmega = np.dot(np.linalg.inv(R), np.array(lobatto_weights[p]).reshape((p + 1, 1)))

Z = np.dot(
    np.kron(R, R),
    np.dot(
        np.linalg.inv(Psi2d + 2 * d_min[p] * lbd * np.kron(I, I)),
        np.concatenate(
            (np.kron(lbd_x * invR, invROmega), lbd_y * np.kron(invROmega, invR)), axis=1
        ),
    ),
)

# TODO: check the line below
# L2dV_explInv = np.dot(np.linalg.inv(np.kron(M, M)), np.dot(np.kron(I, I) + \
#                np.dot(Z, np.dot(np.linalg.inv(np.eye(2 * p + 2) + np.dot(np.transpose(Vv), Z)), np.transpose(Vv))), invL2d0))

L2dV_explInv = np.dot(
    np.linalg.inv(np.kron(M, M)),
    np.dot(np.linalg.inv(np.kron(I, I) - np.dot(Z, np.transpose(Vv))), invL2d0),
)

print(
    "Verification of diag. block inv. with graph visc. (difference inf. norm between the two methods): {}".format(
        np.linalg.norm(L2dV_numpyInv - L2dV_explInv)
    )
)
