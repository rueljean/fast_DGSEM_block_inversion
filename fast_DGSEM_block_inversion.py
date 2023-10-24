#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides several tools to invert dense diagonal blocks resulting from
hyperbolic problems discretized with DGSEM. All the details are in "Maximum principle
preserving time implicit DGSEM for linear scalar hyperbolic conservation laws", by
R. Milani, F. Renac, J. Ruel [MRR], and, more specifically in "Appendix B: Inversion of
diagonal blocks". Indeed, to better document our code, we refer to equations there,
e.g., see [MRR, (B.1)].

One of the goals of the module is also to check the exactness of the original inversion
strategies with respect to standard algebraic tools (i.e., `numpy`): see the `__main__`
function and functions called therein.
"""

import math

import numpy as np

# -----------------------------------------------------------------------------
# Gauss-Lobatto points and weights / d_min
# -----------------------------------------------------------------------------

_1OV3 = 1.0 / 3.0
_1OV6 = 0.5 * _1OV3
_SQRT1OV5 = math.sqrt(0.2)
_SQRT5OV3 = math.sqrt(5.0 / 3.0)
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
            -1.0,
            -math.sqrt(_1OV3 + _2SQRT7OV21),
            -math.sqrt(_1OV3 - _2SQRT7OV21),
            +math.sqrt(_1OV3 - _2SQRT7OV21),
            +math.sqrt(_1OV3 + _2SQRT7OV21),
            1.0,
        ],
    ),
    6: np.array(
        [
            -1.0,
            -math.sqrt((5.0 + 2.0 * _SQRT5OV3) / 11.0),
            -math.sqrt((5.0 - 2.0 * _SQRT5OV3) / 11.0),
            0.0,
            +math.sqrt((5.0 - 2.0 * _SQRT5OV3) / 11.0),
            +math.sqrt((5.0 + 2.0 * _SQRT5OV3) / 11.0),
            1.0,
        ],
    ),
}

_49OV90 = 49.0 / 90.0
_SQRT15 = math.sqrt(15.0)
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
    6: np.array(
        [
            1.0 / 21.0,
            (124 - 7.0 * _SQRT15) / 350.0,
            (124 + 7.0 * _SQRT15) / 350.0,
            256.0 / 525.0,
            (124 + 7.0 * _SQRT15) / 350.0,
            (124 - 7.0 * _SQRT15) / 350.0,
            1.0 / 21.0,
        ]
    ),
}

d_min_BY_ORDER = {1: 8, 2: 24, 3: 24 * (1 + np.sqrt(5)), 4: 198.6, 5: 428.8, 6: 820.8}

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def lagrange_derivative(l, xi, pts):
    """Derivative of the l-th Lagrange polynomial at the point xi

    Reference:
        RHS of [MRR, (B.1)]

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

    Return: np.ndarray
        Result of the multiplication
    """
    return A_diag[:, None] * B


def diagonal_auto_kron(A_diag):
    """Let A be a diagonal matrix and A_diag its diagonal, this function compute
    np.diag(np.kron(A, A)) taking advantage of the structure of the matrix

    A_diag: np.ndarray
        Diagonal of the matrix

    Return: np.ndarray
        Equivalent of np.diag(np.kron(A, A))
    """
    # Vector-vector product, then flatten the resulting matrix
    return np.multiply(A_diag.reshape((A_diag.shape[0], -1)), A_diag).flatten()


def diagonal_add(A, val=1.0):
    """Add `val` to the diagonal of `A`. Hence, `diagonal_add(A, 1)` is equivalent to
    adding the identity matrix to `A`

    A: np.ndarray
        Matrix
    val: float or np.ndarray of the same size of np.diag(A), default: 1.0
        Value(s) to add to the diagonal of A

    Return: np.ndarray
        Matrix A modified to have `val` added to its diagonal
    """
    ret = A.copy()
    ret[np.diag_indices_from(ret)] += val
    return ret

def diagonal_solve(A_diag, b, inverted=False):
    """Solve problem A.x=b when A is diagonal

    A_diag: np.ndarray
        Diagonal of the matrix composing the LHS of the system
    b: np.ndarray
        RHS of the system
    inverted: bool, default: False
        Whether `A_diag` is already inverted

    Return: np.ndarray
        Solution of A.x=b
    """
    return diagonal_matrix_multiply(A_diag if inverted else np.reciprocal(A_diag), b)


def I_kron_mat(A, dim_I=None):
    """Equivalent to `np.kron(I, A)`

    A: np.ndarray
        Matrix
    dim_I: int or None
        Dimension of I. If None, `A.shape[0]`

    Return: np.ndarray
        `np.kron(I, A)`
    """
    dim_I = dim_I if dim_I is not None else A.shape[0]
    ret = np.zeros((dim_I * A.shape[0], dim_I * A.shape[1]))
    for i in range(dim_I):
        ret[
            i * A.shape[0] : (i + 1) * A.shape[0], i * A.shape[1] : (i + 1) * A.shape[1]
        ] = A
    return ret


def D_matrix(p):
    """Discrete derivative matrix

    Reference:
        [MRR, (B.1)]

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

    Reference:
        [MRR, (B.3)]

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

    Reference:
        [MRR, (B.7a)]

    D: np.ndarray
        Derivative matrix [MRR, (B.1)]
    lambda_x, lambda_y: float
        Celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        The 2D matrix
    """
    lbd = lambda_x + lambda_y

    I = np.eye(D.shape[0])
    L1d = diagonal_add(-2 * lbd * L_matrix(D), 1.0)  # [MRR, B.3)

    L2d = lambda_x / lbd * I_kron_mat(L1d) + lambda_y / lbd * np.kron(L1d, I)
    return L2d


def R_matrix(D, psi):
    """Given the eigenvalues, compute the right-eigen-matrix

    Reference:
        [MRR, (B.4-5)]

    D: np.ndarray
        Derivative matrix [MRR, (B.1)]
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

    Reference:
        Although not explicitly detailed, one can grasp the intent of this function by
        referring to [MRR, (B.7b,8b)]

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
    Psi2d = lambda_x / lbd * I_kron_mat(Psi_lbd) + lambda_y / lbd * np.kron(Psi_lbd, I)
    return Psi2d


def eigen_2D_diag(Psi, lambda_x, lambda_y):
    """Given the 1D ones, compute the 2D eigen-values

    Reference:
        Although not explicitly detailed, one can grasp the intent of this function by
        referring to [MRR, (B.7b,8b)]

    Psi: np.ndarray
        1D eigenvalues
    lambda_x, lambda_y: float
        Celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        np.ndarray with containing the 2D eigenvalues
    """
    return 1.0 - 2.0 * np.array(
        [lambda_x * p_i + lambda_y * p_j for p_j in Psi for p_i in Psi]
    )


def eigen_L_numpy(D):
    """Using matrix D, compute matrix L, then compute its eigen-values using standard
    numpy functions

    Reference:
        Solution of [MRR, (B.4)] using `numpy` tools

    D: np.ndarray
        Derivative matrix [MRR, (B.1)]

    Return:
        The eigenvalues
    """
    L = L_matrix(D)
    eigValNp, _ = np.linalg.eig(L)
    return eigValNp


def eigen_L_analytical(D):
    """Using matrix D, compute matrix L, then compute its eigen-values using standard
    numpy functions

    Reference:
        Solution of [MRR, (B.4)] using [MRR, (B.5)]

    D: np.ndarray
        Derivative matrix [MRR, (B.1)]

    Return:
        The eigenvalues
    """
    p = D.shape[0] - 1

    # Storing powers of D: D_powers[i,:,:] is the i-th power of D
    # TODO: Since we use only one entry of these matrices, could we find a more
    # efficient way instead of computing the full matrix (since it's a power,
    # keeping the last row of the matrix should be sufficient even to compute the whole
    # series)?
    D_powers = np.stack([np.linalg.matrix_power(D, i) for i in range(0, p + 1)])

    coeff = np.zeros(p + 2)
    coeff[0] = LOBATTO_WEIGHTS_BY_ORDER[p][p]
    coeff[1:] = D_powers[:, p, p]

    # Eigenvalues
    Psi = np.roots(coeff)
    return Psi


def L2d_inversion_numpy(D, M, lambda_x, lambda_y):
    """Invert 2D systems resulting from DGSEM problems with numpy function

    Reference:
        Solve [MRR, (B.6)] using `numpy` tools

    D: np.ndarray
        Derivative matrix [MRR, (B.1)]
    M: np.ndarray
        Mass matrix [MRR, (B.2)]
    lambda_x, lambda_y: float
        Celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        The inverse of the 2D matrix
    """
    L2d = L2d_matrix(D, lambda_x, lambda_y)  # [MRR, (B.7a)]

    return diagonal_solve(diagonal_auto_kron(np.diag(M)), np.linalg.inv(L2d))


def L2d_inversion_analytical(
    D,
    M,
    lambda_x,
    lambda_y,
    Psi=None,
    iR2d=None,
    iMR2d=None,
):
    """Invert 2D systems resulting from DGSEM problems with analytical method

    Reference:
        Solve [MRR, (B.6)] using [MRR, (B.8)]

    D: np.ndarray
        Derivative matrix [MRR, (B.1)]
    M: np.ndarray
        Mass matrix [MRR, (B.2)]
    lambda_x, lambda_y: float
        Celerity*time step over spatial grid size in, respectively, in x
        and y direction
    Psi: np.ndarray or None
        Eigenvalues, see [MRR, (B.4)]
    iR2d: np.ndarray or None
        Kronecker product of the inverse of the right-eigenvector matrix by itself
    iMR2d: np.ndarray or None
        Kronecker product of the product of the inverse of the mass matrix by the
        right-eigenvector matrix by itself

    Return:
        The inverse of the 2D matrix
    """
    Psi = Psi if Psi is not None else eigen_L_analytical(D)
    Psi2d_diag = eigen_2D_diag(Psi, lambda_x, lambda_y)

    if iR2d is None or iMR2d is None:
        R = R_matrix(D, Psi)
        invR = np.linalg.inv(R)
        invMR = diagonal_solve(np.diag(M), R)

    iMR2d = iMR2d if iMR2d is not None else np.kron(invMR, invMR)
    iR2d = iR2d if iR2d is not None else np.kron(invR, invR)

    return np.dot(iMR2d, diagonal_solve(Psi2d_diag, iR2d))


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

    Reference:
        Solve [MRR, (B.10)] using `numpy` tools

    p: int
        Polynomial order
    D: np.ndarray
        Derivative matrix [MRR, (B.1)]
    M: np.ndarray
        Mass matrix [MRR, (B.2)]
    lambda_x, lambda_y: float
        Ratio between celerity*time step over spatial grid size in, respectively, in x
        and y direction
    VvT: np.array or None
        Matrix with sparse structure for 2D problems [MRR, (B.9d)]
    IOmega: np.array or None
        Kronecker product of identity and Lobatto weights, first element of RHS of
        [MRR, (B.9c)]
    OmegaI: np.array or None
        Kronecker product of Lobatto weights and identity, second element of RHS of
        [MRR, (B.9c)]

    Return:
        The inverse of the matrix
    """
    p = D.shape[0] - 1
    lobatto_weights = LOBATTO_WEIGHTS_BY_ORDER[p]
    d_min = d_min_BY_ORDER[p]

    lbd = lambda_x + lambda_y

    I = np.eye(p + 1)

    # [MRR, (B.9b)]
    # L2d0 = L2d_matrix(D, lambda_x, lambda_y) + 2 * d_min * lbd * np.kron(I, I)
    L2d0 = diagonal_add(L2d_matrix(D, lambda_x, lambda_y), 2 * d_min * lbd)

    if IOmega is None or OmegaI is None:
        Omega = lobatto_weights.reshape((p + 1, 1))
    IOmega = IOmega if IOmega is not None else I_kron_mat(Omega)
    OmegaI = OmegaI if OmegaI is not None else np.kron(Omega, I)

    # [MRR, (B.9c)]
    Uv = 2 * d_min * np.concatenate((lambda_x * IOmega, lambda_y * OmegaI), axis=1)
    # Vv = np.concatenate(
    #    (np.kron(I, np.ones((p + 1, 1))), np.kron(np.ones((p + 1, 1)), I)), axis=1
    # )

    if VvT is None:
        VvT = np.transpose(
            np.concatenate(
                (I_kron_mat(np.ones((p + 1, 1))), np.kron(np.ones((p + 1, 1)), I)),
                axis=1,
            )
        )

    # [MRR, (B.9a)]
    L2dV = L2d0 - np.dot(Uv, VvT)

    # M is diagonal, hence we M<kron>M is diagonal
    return diagonal_solve(diagonal_auto_kron(np.diag(M)), np.linalg.inv(L2dV))


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

    Reference:
        Solve [MRR, (B.10)] using [MRR, (B.11)]

    p: int
        Polynomial order
    D: np.ndarray
        Derivative matrix [MRR, (B.1)]
    M: np.ndarray
        Mass matrix [MRR, (B.2)]
    lambda_x, lambda_y: float
        Ratio between celerity*time step over spatial grid size in, respectively, in x
        and y direction
    Psi: np.ndarray or None
        Eigenvalues, see [MRR, (B.4)]
    iR2d: np.ndarray or None
        Kronecker product of the inverse of the right-eigenvector matrix by itself
    iMR2d: np.ndarray or None
        Kronecker product of the product of the inverse of the mass matrix by the
        right-eigenvector matrix by itself
    VvT: np.array or None
        Matrix with sparse structure for 2D problems [MRR, (B.9d)]
    invRROmega: np.array or None
        kron(R^{-1}, R^{-1}.Omega), R being the right-eigenvector matrix and Omega the
        Lobatto weights
    invROmegaR: np.array or None
        kron(R^{-1}.Omega, R^{-1}), R being the right-eigenvector matrix and Omega the
        Lobatto weights

    Return:
        The inverse of the matrix
    """
    p = D.shape[0] - 1
    d_min = d_min_BY_ORDER[p]

    Psi = Psi if Psi is not None else eigen_L_analytical(D)
    Psi2d_diag = eigen_2D_diag(Psi, lambda_x, lambda_y)

    if R2d is None or iR2d is None:
        R = R_matrix(D, Psi)
        invR = np.linalg.inv(R)
    R2d = R2d if R2d is not None else np.kron(R, R)
    iR2d = iR2d if iR2d is not None else np.kron(invR, invR)

    if invRROmega is None or invROmegaR is None:
        invROmega = np.dot(invR, LOBATTO_WEIGHTS_BY_ORDER[p].reshape((p + 1, 1)))
    invRROmega = invRROmega if invRROmega is not None else np.kron(invR, invROmega)
    invROmegaR = invROmegaR if invROmegaR is not None else np.kron(invROmega, invR)

    lbd = lambda_x + lambda_y

    if VvT is None:
        I = np.eye(p + 1)
        VvT = np.transpose(
            np.concatenate(
                (I_kron_mat(np.ones((p + 1, 1))), np.kron(np.ones((p + 1, 1)), I)),
                axis=1,
            )
        )

    # Central element of RHS of [MRR, (B.8b)]
    # Explicit diagonal block inversion
    # invPsi2d = np.linalg.inv(Psi2d + 2 * d_min * lbd * np.kron(I, I))
    invdiagPsi = np.reciprocal(Psi2d_diag + 2 * d_min * lbd)

    # See [MRR, (B.9b)]
    invL2d0 = np.dot(R2d, diagonal_solve(invdiagPsi, iR2d, True))

    # [MRR, (B.11b)]
    Z = (
        2
        * d_min
        * np.dot(
            R2d,
            diagonal_solve(
                invdiagPsi,
                np.concatenate((lambda_x * invRROmega, lambda_y * invROmegaR), axis=1),
                True,
            ),
        )
    )

    # omega = LOBATTO_WEIGHTS_BY_ORDER[p].reshape((p + 1, 1))
    # Z = np.dot(
    #    invL2d0,
    #    np.concatenate(
    #        (
    #            np.kron(lambda_x * I, omega),
    #            np.kron(omega, lambda_y * I),
    #        ),
    #        axis=1,
    #    ),
    # )

    # M is diagonal, hence we M<kron>M is diagonal, [MRR, (B.11d)]
    L2dV_explInv = diagonal_solve(
        diagonal_auto_kron(np.diag(M)),
        np.dot(
            diagonal_add(
                np.dot(
                    Z,
                    # [MRR, (B.11c)]
                    np.linalg.solve(diagonal_add(-np.dot(VvT, Z), 1.0), VvT),
                ),
                1.0,
            ),
            invL2d0,
        ),
    )

    return L2dV_explInv


def compare_eigenvalues_computation(p):
    """Compare the computation of the eigenvalues of the L matrix with numpy function
    and analytical formula

    Reference:
        Compare methods for solving [MRR, (B.4)]

    p: int
        Polynomial order

    Return:
        The eigenvalues
    """
    # Main matrix with derivatives
    D = D_matrix(p)  # [MRR, B.1)

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

    Reference:
        Compare methods for solving [MRR, (B.6)]

    p: int
        Polynomial order
    lambda_x, lambda_y: float
        Ratio between celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        The inverse of the 2D matrix
    """

    lobatto_weights = LOBATTO_WEIGHTS_BY_ORDER[p]

    # Main matrices
    D = D_matrix(p)  # [MRR, B.1)
    M = 0.5 * np.diag(lobatto_weights)  # [MRR, B.2)

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

    Reference:
        Compare methods for solving [MRR, (B.10)]

    p: int
        Polynomial order
    lambda_x, lambda_y: float
        Ratio between celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        The inverse of the matrix
    """

    lobatto_weights = LOBATTO_WEIGHTS_BY_ORDER[p]

    # Main matrices
    D = D_matrix(p)  # [MRR, B.1)
    M = 0.5 * np.diag(lobatto_weights)  # [MRR, B.2)

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
