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

import math

import numpy as np

# -----------------------------------------------------------------------------
# Gauss-Lobatto points and weights / d_min
# -----------------------------------------------------------------------------

_1OV3 = 1.0 / 3.0
_1OV6 = 0.5 * _1OV3
_SQRT1OV5 = math.sqrt(0.2)
_SQRT3OV7 = math.sqrt(3.0 / 7.0)
_SQRT7 = math.sqrt(7.0)
_2SQRT7OV21 = 2.0 * _SQRT7 / 21.0
LOBATTO_POINTS_BY_ORDER = {
    1: np.array([-1.0, 1.0]),
    2: np.array([-1.0, 0.0, 1.0]),
    3: np.array([-1.0, -_SQRT1OV5, +_SQRT1OV5, 1.0]),
    4: np.array([-1.0, -_SQRT3OV7, 0.0, +_SQRT3OV7, 1.0]),
    5: np.array(
        [
            -1,
            -math.sqrt(_1OV3 + _2SQRT7OV21),
            -math.sqrt(_1OV3 - _2SQRT7OV21),
            +math.sqrt(_1OV3 - _2SQRT7OV21),
            +math.sqrt(_1OV3 + _2SQRT7OV21),
            1,
        ],
    ),
}

_49OV90 = 49.0 / 90.0
LOBATTO_WEIGHTS_BY_ORDER = {
    1: np.array([1, 1]),
    2: np.array([_1OV3, 4.0 * _1OV3, _1OV3]),
    3: np.array([_1OV6, 5.0 * _1OV6, 5.0 * _1OV6, _1OV6]),
    4: np.array([0.1, _49OV90, 32.0 / 45.0, _49OV90, 0.1]),
    5: np.array(
        [
            1.0 / 15.0,
            (14.0 - _SQRT7) / 30.0,
            (14.0 + _SQRT7) / 30.0,
            (14.0 + _SQRT7) / 30.0,
            (14.0 - _SQRT7) / 30.0,
            1.0 / 15.0,
        ]
    ),
}

d_min_BY_ORDER = {1: 8, 2: 24, 3: 24 * (1 + np.sqrt(5)), 4: 198.6, 5: 428.8}

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def lagrange_derivative(l, xi, pts):
    """Derivative of the l-th Lagrange polynomial at the point xi

    l: int
        Index of the desired polynomial
    xi: float
        Point at which the derivative is computed
    pts: np.array
        Lagrangian points
    """
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


def diagonal_matrix_multiply(A_diag, B):
    """Fast way to compute A.B whenever A is diagonal.
    Taken from https://stackoverflow.com/a/44388621/12152457

    A_diag: np.ndarray
        Diagonal of first matrix
    B: np.ndarray
        Second matrix
    invert_A: bool
        Whether to invert A before multiplying

    Return: np.ndarray
        Result of the multiplication
    """
    return A_diag[:, None] * B


def diagonal_auto_kron(A_diag):
    """Let A be a diagonal matrix and A_diag its diagonal, this function compute
    np.diag(np.kron(A, A)) taking advantage of the structure of the matrix
    """
    # Vector-vector product, then flatten the resulting matrix
    return np.multiply(A_diag.reshape((A_diag.shape[0], -1)), A_diag).flatten()


def D_matrix(p):
    """Discrete derivative matrix

    p: int
        Polynomial order

    Return:
        The matrix
    """
    points = LOBATTO_POINTS_BY_ORDER[p]
    D = np.zeros((p + 1, p + 1))
    for k in range(0, p + 1):
        for l in range(0, p + 1):
            D[k][l] = lagrange_derivative(l, points[k], points)
    return D


def L_matrix(D):
    """Compute matrix L using relying on matrix D

    D: np.ndarray
        Derivative matrix

    Return:
        The matrix
    """
    p = D.shape[0] - 1
    A = np.zeros_like(D)
    A[-1, -1] = 1.0
    return D.T - A / LOBATTO_WEIGHTS_BY_ORDER[p][p]


def L2d_matrix(D, lambda_x, lambda_y):
    """Compute the matrix for 2D DGSEM systems

    D: np.ndarray
        Derivative matrix
    lambda_x, lambda_y: float
        Celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        The 2D matrix
    """
    lbd = lambda_x + lambda_y

    I = np.eye(D.shape[0])
    L = L_matrix(D)
    L1d = I - 2 * lbd * L

    L2d = lambda_x / lbd * np.kron(I, L1d) + lambda_y / lbd * np.kron(L1d, I)
    return L2d


def R_matrix(D, psi):
    """Given the eigenvalues, compute the right-eigen-matrix

    D: np.ndarray
        Derivative matrix
    psi: np.ndarray
        Eigenvalues

    Return:
        The right-eigen-matrix
    """
    p = D.shape[0] - 1
    ov_w_p = 1.0 / LOBATTO_WEIGHTS_BY_ORDER[p][p]
    D_powers = np.stack([np.linalg.matrix_power(D, i) for i in range(0, p + 1)])
    Psi_powers = np.stack([psi ** (-l - 1) for l in range(p + 1)], axis=1)
    # Psi_powers = 1. / np.stack([psi ** (l + 1) for l in range (p+1)])
    R = np.ones_like(D, dtype="complex_")  # Right eigenvectors matrix
    R[:-1, :] = -np.dot(Psi_powers, D_powers[:, p, :-1]).T * ov_w_p
    # for i in range(p + 1):
    #    Psi_i_powers = np.array([psi[i] ** (-l - 1) for l in range(0, p + 1)])
    #    R[:-1, i] = -np.dot(Psi_i_powers, D_powers[:, p, :-1]) * ov_w_p
    # #    for k in range(p):
    # #        R[k, i] = -np.dot(Psi_i_powers, D_powers[:, p, k]) * ov_w_p
    return R


def eigen_2D(Psi, lambda_x, lambda_y):
    """Given the 1D ones, compute the 2D eigen-values

    Psi: np.ndarray
        1D eigenvalues
    lambda_x, lambda_y: float
        Celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        np.ndarray with containing the 2D eigenvalues
    """
    lbd = lambda_x + lambda_y

    I = np.eye(Psi.shape[0])
    # Psi_lbd = I - 2 * lbd * np.diag(Psi)
    Psi_lbd = np.diag(1.0 - 2.0 * lbd * Psi)
    Psi2d = lambda_x / lbd * np.kron(I, Psi_lbd) + lambda_y / lbd * np.kron(Psi_lbd, I)
    return Psi2d

def eigen_2D_diag(Psi, lambda_x, lambda_y):
    """Given the 1D ones, compute the 2D eigen-values

    Psi: np.ndarray
        1D eigenvalues
    lambda_x, lambda_y: float
        Celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        np.ndarray with containing the 2D eigenvalues
    """
    return 1. - 2. * np.array([lambda_x*p_i + lambda_y*p_j for p_j in Psi for p_i in Psi])


def eigen_L_numpy(D):
    """Using matrix D, compute matrix L, then compute its eigen-values using standard
    numpy functions

    D: np.ndarray
        Derivative matrix

    Return:
        The eigenvalues
    """
    L = L_matrix(D)
    eigValNp, _ = np.linalg.eig(L)
    return eigValNp


def eigen_L_analytical(D):
    """Using matrix D, compute matrix L, then compute its eigen-values using standard
    numpy functions

    D: np.ndarray
        Derivative matrix

    Return:
        The eigenvalues
    """
    p = D.shape[0] - 1

    # Storing powers of D: D_powers[i,:,:] is the i-th power of D
    D_powers = np.stack([np.linalg.matrix_power(D, i) for i in range(0, p + 1)])

    coeff = np.zeros(p + 2)
    coeff[0] = LOBATTO_WEIGHTS_BY_ORDER[p][p]
    coeff[1:] = D_powers[:, p, p]

    # Eigenvalues
    Psi = np.roots(coeff)
    return Psi


def L2d_inversion_numpy(D, M, lambda_x, lambda_y):
    """Invert 2D systems resulting from DGSEM problems with numpy function

    D: np.ndarray
        Derivative matrix
    M: np.ndarray
        M matrix
    lambda_x, lambda_y: float
        Celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        The inverse of the 2D matrix
    """
    L2d = L2d_matrix(D, lambda_x, lambda_y)

    return diagonal_matrix_multiply(
        np.reciprocal(diagonal_auto_kron(np.diag(M))), np.linalg.inv(L2d)
    )


def L2d_inversion_analytical(
    D,
    M,
    lambda_x,
    lambda_y,
    Psi=None,
    #R=None,
    #invR=None,
    #invMR=None,
    iR2d=None,
    iMR2d=None,
):
    """Invert 2D systems resulting from DGSEM problems with analytical method

    D: np.ndarray
        Derivative matrix
    M: np.ndarray
        M matrix
    lambda_x, lambda_y: float
        Celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        The inverse of the 2D matrix
    """
    Psi = Psi if Psi is not None else eigen_L_analytical(D)
    Psi2d_diag = eigen_2D_diag(Psi, lambda_x, lambda_y)

    if(iR2d is None or iMR2d is None):
        R = R_matrix(D, Psi)
        invR = np.linalg.inv(R)
        invMR = diagonal_matrix_multiply(np.reciprocal(np.diag(M)), R)
    
    iMR2d = iMR2d if iMR2d is not None else np.kron(invMR, invMR)
    iR2d = iR2d if iR2d is not None else np.kron(invR, invR)

    return np.dot(
        iMR2d,
        diagonal_matrix_multiply(np.reciprocal(Psi2d_diag), iR2d),
    )


def L2d_inversion_viscosity_numpy(
    D, 
    M, 
    lambda_x, 
    lambda_y,
    VvT=None,
    IOmega=None,
    OmegaI=None,
    ):
    """Invert 2D systems resulting from DGSEM problems with graph-viscosity with
    standard `numpy` functions

    p: int
        Polynomial order
    D: np.ndarray
        Derivative matrix
    M: np.ndarray
        Mass matrix
    lambda_x, lambda_y: float
        Ration between celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        The inverse of the matrix
    """
    p = D.shape[0] - 1
    lobatto_weights = LOBATTO_WEIGHTS_BY_ORDER[p]
    d_min = d_min_BY_ORDER[p]

    lbd = lambda_x + lambda_y

    I = np.eye(p + 1)

    # L2d0 = L2d_matrix(D, lambda_x, lambda_y) + 2 * d_min * lbd * np.kron(I, I)
    L2d0 = L2d_matrix(D, lambda_x, lambda_y)
    L2d0[np.diag_indices_from(L2d0)] += 2 * d_min * lbd
    
    
    if(IOmega is None or OmegaI is None):
        Omega = lobatto_weights.reshape((p + 1, 1))
    IOmega = IOmega if IOmega is not None else np.kron(I, Omega)
    OmegaI = OmegaI if OmegaI is not None else np.kron(Omega, I)
    
    Uv = np.concatenate(
        (
            lambda_x * IOmega,
            lambda_y * OmegaI,
        ),
        axis=1,
    )
    #Vv = np.concatenate(
    #    (np.kron(I, np.ones((p + 1, 1))), np.kron(np.ones((p + 1, 1)), I)), axis=1
    #)

    if(VvT is None):
        VvT = np.transpose(
            np.concatenate(
             (np.kron(I, np.ones((p + 1, 1))), np.kron(np.ones((p + 1, 1)), I)), axis=1
            )
        )

    L2dV = L2d0 - np.dot(Uv, VvT)

    # M is diagonal, hence we M<kron>M is diagonal
    invMkM_diag = np.reciprocal(diagonal_auto_kron(np.diag(M)))
    # Diagonal block inversion with numpy
    L2dV_numpyInv = diagonal_matrix_multiply(invMkM_diag, np.linalg.inv(L2dV))
    return L2dV_numpyInv


def L2d_inversion_viscosity_analytical(
    D,
    M,
    lambda_x,
    lambda_y,
    Psi=None,
    R2d=None,
    iR2d=None,
    VvT=None,
    invRROmega=None,
    invROmegaR=None,
):
    """Invert 2D systems resulting from DGSEM problems with graph-viscosity with
    analytical formula

    p: int
        Polynomial order
    D: np.ndarray
        Derivative matrix
    M: np.ndarray
        Mass matrix
    lambda_x, lambda_y: float
        Ration between celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        The inverse of the matrix
    """
    p = D.shape[0] - 1
    d_min = d_min_BY_ORDER[p]

    Psi = Psi if Psi is not None else eigen_L_analytical(D)
    Psi2d_diag = eigen_2D_diag(Psi, lambda_x, lambda_y)

    if(R2d is None or iR2d is None):
        R = R_matrix(D, Psi)
        invR = np.linalg.inv(R)
    R2d = R2d if R2d is not None else np.kron(R, R)
    iR2d = iR2d if iR2d is not None else np.kron(invR, invR)
    
    if(invRROmega is None or invROmegaR is None):
        invROmega = np.dot(invR, LOBATTO_WEIGHTS_BY_ORDER[p].reshape((p + 1, 1)))
    invRROmega = invRROmega if invRROmega is not None else np.kron(invR, invROmega)
    invROmegaR = invROmegaR if invROmegaR is not None else np.kron(invROmega, invR)

    lbd = lambda_x + lambda_y

    if(VvT is None):
        I = np.eye(p + 1)
        VvT = np.transpose(
            np.concatenate(
             (np.kron(I, np.ones((p + 1, 1))), np.kron(np.ones((p + 1, 1)), I)), axis=1
            )
        )
    
    # Explicit diagonal block inversion
    # invPsi2d = np.linalg.inv(Psi2d + 2 * d_min * lbd * np.kron(I, I))
    invdiagPsi = np.reciprocal(Psi2d_diag + 2 * d_min * lbd)
    invL2d0 = np.dot(
        R2d, diagonal_matrix_multiply(invdiagPsi, iR2d)
    )

    Z = np.dot(
        R2d,
        diagonal_matrix_multiply(
            invdiagPsi,
            np.concatenate(
                (
                    lambda_x * invRROmega,
                    lambda_y * invROmegaR,
                ),
                axis=1,
            ),
        ),
    )
    
    #omega = LOBATTO_WEIGHTS_BY_ORDER[p].reshape((p + 1, 1))
    #Z = np.dot(
    #    invL2d0,
    #    np.concatenate(
    #        (
    #            np.kron(lambda_x * I, omega),
    #            np.kron(omega, lambda_y * I),
    #        ),
    #        axis=1,
    #    ),
    #)

    # M is diagonal, hence we M<kron>M is diagonal
    invMkM_diag = np.reciprocal(diagonal_auto_kron(np.diag(M)))
    
    L2dV_explInv = diagonal_matrix_multiply(
        invMkM_diag,
        np.dot(
            np.eye((p+1)*(p+1))
            + np.dot(
                Z,
                np.dot(
                    np.linalg.inv(np.eye(2 * p + 2) - np.dot(VvT, Z)), 
                    VvT,
                ),
            ),
            invL2d0,
        ),
    )

    return L2dV_explInv


def compare_eigenvalues_computation(p):
    """Compare the computation of the eigenvalues of the L matrix with numpy function
    and analytical formula

    p: int
        Polynomial order

    Return:
        The eigenvalues
    """
    # Main matrix with derivatives
    D = D_matrix(p)

    # Computation of L eigenvalues and eigenvectors with numpy
    eigValNp = eigen_L_numpy(D)

    # "Analytical" computation of L eigenvalues and eigenvectors
    Psi = eigen_L_analytical(D)

    # print("L eigenval.with numpy: {}".format(eigValNp))
    # print("Semi-analytical L eigenval.: {}\n".format(Psi))
    print(
        "Verification of L eigenvalues (difference inf. norm between the two methods): {}\n".format(
            np.linalg.norm(eigValNp - Psi)
        )
    )
    return Psi


def compare_2D_inversion(p, lambda_x, lambda_y):
    """Compare standard and fast methods to invert 2D systems resulting from DGSEM problems

    p: int
        Polynomial order
    lambda_x, lambda_y: float
        Ration between celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        The inverse of the 2D matrix
    """

    lobatto_weights = LOBATTO_WEIGHTS_BY_ORDER[p]

    # Main matrices
    D = D_matrix(p)
    M = 0.5 * np.diag(lobatto_weights)

    # Diagonal block inversion with numpy
    L2d_numpyInv = L2d_inversion_numpy(D, M, lambda_x, lambda_y)

    # Explicit diagonal block inversion
    L2d_explInv = L2d_inversion_analytical(D, M, lambda_x, lambda_y)

    print(
        "Verification of diag. block inv. (difference inf. norm between the two methods): {}\n".format(
            np.linalg.norm(L2d_numpyInv - L2d_explInv)
        )
    )
    return L2d_numpyInv


def compare_2D_inversion_viscosity(p, lambda_x, lambda_y):
    """Compare standard and fast methods to invert 2D systems resulting from DGSEM
    problems with graph-viscosity

    p: int
        Polynomial order
    lambda_x, lambda_y: float
        Ration between celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        The inverse of the matrix
    """

    lobatto_weights = LOBATTO_WEIGHTS_BY_ORDER[p]

    # Main matrices
    D = D_matrix(p)
    M = 0.5 * np.diag(lobatto_weights)

    L2dV_numpyInv = L2d_inversion_viscosity_numpy(D, M, lambda_x, lambda_y)
    L2dV_explInv = L2d_inversion_viscosity_analytical(D, M, lambda_x, lambda_y)

    print(
        "Verification of diag. block inv. with graph visc. (difference inf. norm between the two methods): {}".format(
            np.linalg.norm(L2dV_numpyInv - L2dV_explInv)
        )
    )
    return L2dV_numpyInv


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------------------

    # Polynomial order
    p = 2

    # Celerity*time step over spatial grid size
    lambda_x, lambda_y = 1.0, 1.0

    # -----------------------------------------------------------------------------
    # Diagonalization of L
    # -----------------------------------------------------------------------------
    compare_eigenvalues_computation(p)

    # -----------------------------------------------------------------------------
    # Inversion of L2d
    # -----------------------------------------------------------------------------

    compare_2D_inversion(p, lambda_x, lambda_y)

    # -----------------------------------------------------------------------------
    # Inversion de L2dv
    # -----------------------------------------------------------------------------

    compare_2D_inversion_viscosity(p, lambda_x, lambda_y)
