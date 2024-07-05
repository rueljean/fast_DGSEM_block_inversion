#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides several tools to invert dense diagonal blocks resulting from
hyperbolic problems discretized with DGSEM. All the details are in "Maximum principle
preserving time implicit DGSEM for linear scalar hyperbolic conservation laws", by
R. Milani, F. Renac, J. Ruel [MRR]. To better document our code, we refer to equations
there, e.g., see [MRR, (B.1)].

One of the goals of the module is also to check the exactness of the original inversion
strategies with respect to standard algebraic tools (i.e., `numpy`): see the `__main__`
function and functions called therein.

The module also proposes a further improved method using the ideas from R. Lynch, J.
Rice, and D. Thomas, "Direct Solution of Partial Difference Equations by Tensor Product
Methods", Numerische Mathematik, 6 (1964), pp. 185–199 [LRT].
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
        RHS of [MRR, (6)]

    l: int
        Index of the desired polynomial
    xi: float
        Point at which the derivative is computed
    pts: np.array
        Lagrangian points
    """
    n = len(pts)
    c = 1.0
    for i in range(n):
        if i != l:
            c *= pts[l] - pts[i]
    p = 0.0
    for i in range(n):
        if i != l:
            t = 1.0
            for j in range(n):
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


def diagonal_matrix_multiply_right(A, B_diag):
    """Fast way to compute A.B whenever B is diagonal.
    Adapted from https://stackoverflow.com/a/44388621/12152457

    A: np.ndarray
        First matrix
    B_diag: np.ndarray
        Diagonal of second matrix

    Return: np.ndarray
        Result of the multiplication
    """
    return A * B_diag[None, :]


def diagonal_auto_kron(A_diag):
    """Let A be a diagonal matrix and A_diag its diagonal, this function computes
    np.diag(np.kron(A, A)) taking advantage of the structure of the matrix

    A_diag: np.ndarray
        Diagonal of the matrix

    Return: np.ndarray
        Equivalent of np.diag(np.kron(A, A))
    """
    # Vector-vector product, then flatten the resulting matrix
    # return np.multiply(A_diag.reshape((A_diag.shape[0], -1)), A_diag).flatten()
    return np.outer(A_diag, A_diag).flatten()


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
    A_inv = A_diag if inverted else np.reciprocal(A_diag)
    if len(b.shape) == 1:
        return A_inv * b
    return diagonal_matrix_multiply(A_inv, b)


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


def mass_matrix(p):
    """Compute the mass matrix of a 1D DGSEM discretization.

    Reference:
        [MRR, (17)]

    p: int
        Polynomial order
    """
    return 0.5 * np.diag(LOBATTO_WEIGHTS_BY_ORDER[p])


def mass_matrix_diag(p):
    """Compute the diagonal of the mass matrix of a 1D DGSEM discretization.

    Reference:
        [MRR, (17)]

    p: int
        Polynomial order
    """
    return 0.5 * np.asarray(LOBATTO_WEIGHTS_BY_ORDER[p])


def D_matrix(p):
    """Discrete derivative matrix

    Reference:
        [MRR, (6)]

    p: int
        Polynomial order

    Return:
        The matrix
    """
    points = LOBATTO_POINTS_BY_ORDER[p]
    D = np.empty((p + 1, p + 1))
    for k in range(p + 1):
        for l in range(p + 1):
            D[k][l] = lagrange_derivative(l, points[k], points)
    return D


def L_matrix(D):
    """Compute matrix L relying on matrix D

    Reference:
        Second term of [MRR, (24)]

    D: np.ndarray
        Derivative matrix

    Return:
        The matrix
    """
    p = D.shape[0] - 1
    L = D.T.copy()
    L[-1, -1] -= 1.0 / LOBATTO_WEIGHTS_BY_ORDER[p][p]
    return L


def L2d_matrix(D, lambda_x, lambda_y):
    """Compute the matrix for 2D DGSEM systems

    Reference:
        First term of [MRR, (42)]

    D: np.ndarray
        Derivative matrix [MRR, (6)]
    lambda_x, lambda_y: float
        Ratio between celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        The 2D matrix
    """
    lbd = lambda_x + lambda_y

    I = np.eye(D.shape[0])
    L1d = diagonal_add(-2 * lbd * L_matrix(D), 1.0)  # [MRR, (24)]

    L2d = lambda_x / lbd * I_kron_mat(L1d) + lambda_y / lbd * np.kron(L1d, I)
    return L2d


def R_matrix(D, psi):
    """Given the eigenvalues, compute the right-eigen-matrix

    Reference:
        [MRR, (38-40)]

    D: np.ndarray
        Derivative matrix [MRR, (6)]
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
        referring to [MRR, (42,43)]

    Psi: np.ndarray
        1D eigenvalues
    lambda_x, lambda_y: float
        Ratio between celerity*time step over spatial grid size in, respectively, in x
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


def eigen_2D_diag(Psi, lambda_x, lambda_y, as_array=True):
    """Given the 1D ones, compute the 2D eigen-values

    Reference:
        Although not explicitly detailed, one can grasp the intent of this function by
        referring to [MRR, (42,43)]

    Psi: np.ndarray
        1D eigenvalues
    lambda_x, lambda_y: float
        Ratio between celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        np.ndarray with containing the 2D eigenvalues
    """
    return 1.0 - 2.0 * np.array(
        [lambda_x * p_i + lambda_y * p_j for p_j in Psi for p_i in Psi]
    )


def fast_diagonalization_method(P, Q, L_inv, rhs, invP=None, invQ=None):
    r"""Solve a problem of the form
    ```math
    (I \kron A + B \kron I) x = rhs
    ```
    following section 3 of Lynch et al, "Direct Solution of Partial Difference Equations
    by Tensor Product Methods", Numerische Mathematik, 6 (1964), pp. 185–199. More
    particularly, this function performs eq. (3.9)
    ```math
    x = [(P \kron Q).L.(P^{-1} \ kron Q^{-1})] rhs
    ```
    following steps (3.12)-(3.15).

    Notes:
        The rhs should already be in matrix form.
        The main point on which it relies is the following identity:
        ```math
        (A \kron B) u = A.U.B^T
        ```
        where `U` is a convenient matrix form of `u`

    P: np.ndarray
        Diagonalization matrix for `A`
    Q: np.ndarray
        Diagonalization matrix for `B`
    L_inv: np.ndarray
        Reciprocal of the eigenvalues matrix. It should already be in the final form
        combining both sets of eigenvalues.
    rhs: np.ndarray
        Right-hand side of the system (in matrix form)
    invP: np.ndarray or None. Default: None
        Inverse of `P`. If not provided, computed
    invQ: np.ndarray or None. Default: None
        Inverse of `Q`. If not provided, computed

    Return:
        np.ndarray with the solution
    """
    if invP is None:
        invP = np.linalg.inv(P)
    if invQ is None:
        invQ = np.linalg.inv(Q)
    # We use in the actual code below a concise formula, but for the sake of the
    # example, we give here the step-by-step computation of the original paper
    # R = rhs @ invQ.T  # [LRT, (3.12)]
    # S = L_diag_inv * (invP @ R)  # [LRT, (3.13)]
    # T = S @ Q.T  # [LRT, (3.14)]
    # U = P @ T  # [LRT, (3.15)]
    return P @ np.matmul(L_inv * (invP @ (rhs @ invQ.T)), Q.T)


def eigen_L_numpy(D):
    """Using matrix D, compute matrix L, then compute its eigen-values using standard
    numpy functions

    Reference:
        Perform [MRR, (38)] using `numpy` tools

    D: np.ndarray
        Derivative matrix [MRR, (6)]

    Return:
        The eigenvalues
    """
    L = L_matrix(D)
    eigValNp, _ = np.linalg.eig(L)
    return eigValNp


def eigen_L_analytical(D):
    """Using matrix D, compute the eigen-values of matrix L using an analytical formula

    Reference:
        Perform [MRR, (38)] using [MRR, (40)]

    D: np.ndarray
        Derivative matrix [MRR, (6)]

    Return:
        The eigenvalues
    """
    p = D.shape[0] - 1
    coeff = np.empty(p + 2)
    coeff[0] = LOBATTO_WEIGHTS_BY_ORDER[p][p]
    coeff[1] = 1  # First is identity
    coeff[2] = D[p, p]  # Second is matrix itself
    prod = D.copy()
    for i in range(3, len(coeff)):
        prod = prod @ D
        coeff[i] = prod[p, p]
    # Alternative
    # Storing powers of D: D_powers[i,:,:] is the i-th power of D
    # TODO: Since we use only one entry of these matrices, could we find a more
    # efficient way instead of computing the full matrix (since it's a power,
    # keeping the last row of the matrix should be sufficient even to compute the whole
    # series)?
    # D_powers = np.stack([np.linalg.matrix_power(D, i) for i in range(0, p + 1)])
    # coeff[1:] = D_powers[:, p, p]

    # Eigenvalues
    Psi = np.roots(coeff)
    return Psi


def L2d_inversion_numpy(D, M_diag, lambda_x, lambda_y):
    """Invert 2D systems resulting from DGSEM problems with numpy function

    Reference:
        Compute LHS of [MRR, (43)] (see also [MRR, (B.1)]) using `numpy` tools

    D: np.ndarray
        Derivative matrix [MRR, (6)]
    M_diag: np.ndarray
        Diagonal of the mass matrix, see [MRR, (17)]
    lambda_x, lambda_y: float
        Ratio between celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        The inverse of the 2D matrix
    """
    L2d = L2d_matrix(D, lambda_x, lambda_y)  # First term of [MRR, (42)]

    return diagonal_solve(diagonal_auto_kron(M_diag), np.linalg.inv(L2d))


def L2d_inversion_analytical(
    D,
    M_diag,
    lambda_x,
    lambda_y,
    Psi=None,
    iR2d=None,
    iMR2d=None,
):
    """Invert 2D systems resulting from DGSEM problems with analytical method

    Reference:
        [MRR, (43)] (see also [MRR, (B.1)])

    D: np.ndarray
        Derivative matrix [MRR, (6)]
    M_diag: np.ndarray
        Diagonal of the mass matrix, see [MRR, (17)]
    lambda_x, lambda_y: float
        Ratio between celerity*time step over spatial grid size in, respectively, in x
        and y direction
    Psi: np.ndarray or None
        Eigenvalues, see [MRR, (38)]
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
        invMR = diagonal_solve(M_diag, R)

    iMR2d = iMR2d if iMR2d is not None else np.kron(invMR, invMR)
    iR2d = iR2d if iR2d is not None else np.kron(invR, invR)

    return iMR2d @ diagonal_solve(Psi2d_diag, iR2d)


def L2d_inversion_analytical_FDM(
    D,
    M_diag,
    lambda_x,
    lambda_y,
    rhs,
    Psi=None,
    R=None,
    invR=None,
    invM_R=None,
):
    """Invert 2D systems resulting from DGSEM problems with analytical method using Fast
    Diagonalization Method of [LRT, Sect. 3].

    Reference:
        [MRR, (43)] (see also [MRR, (B.1)])

    D: np.ndarray
        Derivative matrix [MRR, (6)]
    M_diag: np.ndarray
        Diagonal of the mass matrix, see [MRR, (17)]
    lambda_x, lambda_y: float
        Ratio between celerity*time step over spatial grid size in, respectively, in x
        and y direction
    rhs: np.ndarray
        Right-hand side of the problem (in vector storage)
    Psi: np.ndarray or None
        Eigenvalues, see [MRR, (38)]
    R: np.ndarray or None
        Right-eigenvector matrix
    invR: np.ndarray or None
        Inverse of the right-eigenvector matrix
    invM_R: np.ndarray or None
        Product of the inverse of the mass matrix and the right-eigenvector matrix

    Return:
        The solution of the problem (in vector storage)
    """
    Psi = Psi if Psi is not None else eigen_L_analytical(D)
    Psi2d_diag_inv = np.reciprocal(eigen_2D_diag(Psi, lambda_x, lambda_y)).reshape(
        D.shape, order="F"
    )

    if R is None:
        R = R_matrix(D, Psi)
        invR = np.linalg.inv(R)
        invM_R = diagonal_solve(M_diag, R)
    else:
        if invR is None:
            invR = np.linalg.inv(R)
        if invM_R is None:
            invM_R = diagonal_solve(M_diag, R)

    # METHOD 1: Usual FDM than solve (M \kron M)
    # See first term of [MRR, (42)]
    # L2d_inv = fast_diagonalization_method(R, R, Psi2d_diag_inv, rhs, invR, invR)
    # With FDM, we have seen that (P \kron Q).a can be obtained with P.A.Q^T where A is
    # the matrix form of a.
    # Here, we have to solve for (M \kron M), with M diagonal. Hence, it's equivalent to
    # a left multiplication by (M^{-1} \kron M^{-1}). Hence, M^{-1}.A.M^{-T}
    # M_diag_inv = np.reciprocal(M_diag)
    # return diagonal_matrix_multiply(
    #     M_diag_inv, diagonal_matrix_multiply_right(L2d_inv, M_diag_inv)
    # )

    # METHOD 2: include the inversion by M in the FDM, however we tricked it, since
    # what we pass as P^{-1} is not exactly the inverse of what we pass as P.
    return fast_diagonalization_method(
        invM_R, invM_R, Psi2d_diag_inv, rhs.reshape(D.shape, order="F"), invR, invR
    ).flatten(order="F")


def L2d_inversion_viscosity_numpy(
    D,
    M2d_invdiag,
    lambda_x,
    lambda_y,
    VvT=None,
    IOmega=None,
    OmegaI=None,
):
    """Invert 2D systems resulting from DGSEM problems with graph-viscosity with
    standard `numpy` functions

    Reference:
        Invert LHS of [MRR, (B.2)] using `numpy` tools

    p: int
        Polynomial order
    D: np.ndarray
        Derivative matrix [MRR, (6)]
    M2d_invdiag: np.ndarray
        Reciprocal of the diagonal of the 2D mass matrix `M<kron>M`, where M is the 1D
        mass matrix defined in  [MRR, (17)]
    lambda_x, lambda_y: float
        Ratio between celerity*time step over spatial grid size in, respectively, in x
        and y direction
    VvT: np.array or None
        Matrix with sparse structure for 2D problems, see rightmost equation of
        [MRR, (46)]
    IOmega: np.array or None
        Kronecker product of identity and Lobatto weights, first element of RHS of
        central equation of [MRR, (46)]
    OmegaI: np.array or None
        Kronecker product of Lobatto weights and identity, second element of RHS of
        central equation of [MRR, (46)]

    Return:
        The inverse of the matrix
    """
    p = D.shape[0] - 1
    lobatto_weights = LOBATTO_WEIGHTS_BY_ORDER[p]
    d_min = d_min_BY_ORDER[p]

    lbd = lambda_x + lambda_y

    I = np.eye(p + 1)

    # [MRR, (46)]
    # L2d0 = L2d_matrix(D, lambda_x, lambda_y) + 2 * d_min * lbd * np.kron(I, I)
    L2d0 = diagonal_add(L2d_matrix(D, lambda_x, lambda_y), 2 * d_min * lbd)

    if IOmega is None or OmegaI is None:
        Omega = lobatto_weights.reshape((p + 1, 1))
    IOmega = IOmega if IOmega is not None else I_kron_mat(Omega)
    OmegaI = OmegaI if OmegaI is not None else np.kron(Omega, I)

    # Central part of [MRR, (46)]
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

    # [MRR, (45)]
    L2dV = L2d0 - np.matmul(Uv, VvT)

    # M is diagonal, hence we M<kron>M is diagonal
    return diagonal_solve(M2d_invdiag, np.linalg.inv(L2dV), True)


def L2d_inversion_viscosity_analytical(
    D,
    M2d_invdiag,
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
        Invert LHS of [MRR, (B.2)] using [MRR, Algorithm 1]

    p: int
        Polynomial order
    D: np.ndarray
        Derivative matrix [MRR, (6)]
    M2d_invdiag: np.ndarray
        Reciprocal of the diagonal of the 2D mass matrix `M<kron>M`, where M is the 1D
        mass matrix defined in  [MRR, (17)]
    lambda_x, lambda_y: float
        Ratio between celerity*time step over spatial grid size in, respectively, in x
        and y direction
    Psi: np.ndarray or None
        Eigenvalues, see [MRR, (38)]
    R2d: np.ndarray or None
        Kronecker product of the right-eigenvector matrix by itself
    iR2d: np.ndarray or None
        Kronecker product of the inverse of the right-eigenvector matrix by itself
    VvT: np.array or None
        Matrix with sparse structure for 2D problems, see rightmost equation of
        [MRR, (46)]
    invRROmega: np.array or None
        `kron(R^{-1}, R^{-1}.Omega)`, R being the right-eigenvector matrix and Omega
        the Lobatto weights
    invROmegaR: np.array or None
        `kron(R^{-1}.Omega, R^{-1})`, R being the right-eigenvector matrix and Omega
        the Lobatto weights

    Return:
        The inverse of the matrix
    """
    p = D.shape[0] - 1
    d_min = d_min_BY_ORDER[p]

    lbd = lambda_x + lambda_y

    Psi = Psi if Psi is not None else eigen_L_analytical(D)
    # Inverse of the central element of RHS of leftmost equation of [MRR, (46)]
    # Explicit diagonal block inversion
    # invPsi2d = np.linalg.inv(Psi2d + 2 * d_min * lbd * np.kron(I, I))
    invdiagPsi = np.reciprocal(eigen_2D_diag(Psi, lambda_x, lambda_y) + 2 * d_min * lbd)

    if R2d is None or iR2d is None:
        R = R_matrix(D, Psi)
        invR = np.linalg.inv(R)
    R2d = R2d if R2d is not None else np.kron(R, R)
    iR2d = iR2d if iR2d is not None else np.kron(invR, invR)

    if invRROmega is None or invROmegaR is None:
        invROmega = np.dot(invR, LOBATTO_WEIGHTS_BY_ORDER[p].reshape((p + 1, 1)))
    invRROmega = invRROmega if invRROmega is not None else np.kron(invR, invROmega)
    invROmegaR = invROmegaR if invROmegaR is not None else np.kron(invROmega, invR)

    if VvT is None:
        I = np.eye(p + 1)
        VvT = np.transpose(
            np.concatenate(
                (I_kron_mat(np.ones((p + 1, 1))), np.kron(np.ones((p + 1, 1)), I)),
                axis=1,
            )
        )

    # Inverse of leftmost equation of [MRR, (46)], that is, [MRR, Algorithm 1 - step 1]
    invL2d0 = R2d @ diagonal_solve(invdiagPsi, iR2d, True)

    # [MRR, Algorithm 1 - step 2]
    Z = (
        2
        * d_min
        * np.matmul(
            R2d,
            diagonal_solve(
                invdiagPsi,
                np.concatenate((lambda_x * invRROmega, lambda_y * invROmegaR), axis=1),
                True,
            ),
        )
    )

    # omega = LOBATTO_WEIGHTS_BY_ORDER[p].reshape((p + 1, 1))
    # Z = np.matmul(
    #    invL2d0,
    #    np.concatenate(
    #        (
    #            np.kron(lambda_x * I, omega),
    #            np.kron(omega, lambda_y * I),
    #        ),
    #        axis=1,
    #    ),
    # )

    # M is diagonal, hence M<kron>M is diagonal, [MRR, Algorithm 1 - step 4]
    L2dV_explInv = diagonal_solve(
        M2d_invdiag,
        np.matmul(
            #                [MRR, Algorithm 1 - step 3]
            diagonal_add(Z @ np.linalg.solve(diagonal_add(-VvT @ Z, 1.0), VvT), 1.0),
            invL2d0,
        ),
        True,
    )

    return L2dV_explInv


def L2d_inversion_viscosity_analytical_FDM(
    D,
    M2d_invdiag,
    lambda_x,
    lambda_y,
    rhs,
    Psi=None,
    R=None,
    invR=None,
    VvT=None,
    IOmega=None,
    OmegaI=None,
):
    """Invert 2D systems resulting from DGSEM problems with graph-viscosity with
    analytical method using Fast Diagonalization Method of [LRT, Sect. 3].

    Reference:
        Solve [MRR, (B.2)]

    p: int
        Polynomial order
    D: np.ndarray
        Derivative matrix [MRR, (6)]
    M2d_invdiag: np.ndarray
        Reciprocal of the diagonal of the 2D mass matrix `M<kron>M`, where M is the 1D
        mass matrix defined in  [MRR, (17)]
    lambda_x, lambda_y: float
        Ratio between celerity*time step over spatial grid size in, respectively, in x
        and y direction
    rhs: np.ndarray
        Right-hand side of the problem (in vector storage)
    Psi: np.ndarray or None
        Eigenvalues, see [MRR, (38)]
    R: np.ndarray or None
        Right-eigenvector matrix
    invR: np.ndarray or None
        Inverse of the right-eigenvector matrix
    VvT: np.array or None
        Matrix with sparse structure for 2D problems, see rightmost equation of
        [MRR, (46)]
    IOmega: np.array or None
        Kronecker product of identity and Lobatto weights, first element of RHS of
        central equation of [MRR, (46)]
    OmegaI: np.array or None
        Kronecker product of Lobatto weights and identity, second element of RHS of
        central equation of [MRR, (46)]

    Return:
        The solution of the problem (in vector storage)
    """
    p = D.shape[0] - 1
    d_min = d_min_BY_ORDER[p]

    lbd = lambda_x + lambda_y

    Psi = Psi if Psi is not None else eigen_L_analytical(D)
    # Inverse of the central element of RHS of leftmost equation of [MRR, (46)]
    # Explicit diagonal block inversion
    # invPsi2d = np.linalg.inv(Psi2d + 2 * d_min * lbd * np.kron(I, I))
    invdiagPsi = np.reciprocal(
        eigen_2D_diag(Psi, lambda_x, lambda_y) + 2 * d_min * lbd
    ).reshape(D.shape, order="F")

    if R is None:
        R = R_matrix(D, Psi)
        invR = np.linalg.inv(R)
    elif invR is None:
        invR = np.linalg.inv(R)

    if IOmega is None or OmegaI is None:
        Omega = np.asarray(LOBATTO_WEIGHTS_BY_ORDER[p]).reshape((p + 1, 1))
    IOmega = IOmega if IOmega is not None else I_kron_mat(Omega)
    OmegaI = OmegaI if OmegaI is not None else np.kron(Omega, np.eye(*D.shape))

    # Central part of [MRR, (46)]
    Uv = 2 * d_min * np.concatenate((lambda_x * IOmega, lambda_y * OmegaI), axis=1)
    # Vv = np.concatenate(
    #    (np.kron(I, np.ones((p + 1, 1))), np.kron(np.ones((p + 1, 1)), I)), axis=1
    # )

    if VvT is None:
        VvT = np.transpose(
            np.concatenate(
                (
                    I_kron_mat(np.ones((p + 1, 1))),
                    np.kron(np.ones((p + 1, 1)), np.eye(*D.shape)),
                ),
                axis=1,
            )
        )
    # Solving Steps 1 and 2 [MRR, Algorithm 1] together
    Zy = fast_diagonalization_method(
        R,
        R,
        invdiagPsi,
        np.append(
            # Transpose to make the dimensions match. Final dimension (p+2)x(p+1)x(p+1)
            Uv.reshape((*D.shape, -1), order="F").transpose((2, 0, 1)),
            rhs.reshape((1, *D.shape), order="F"),
            axis=0,
        ),
        invR,
        invR,
    )
    # Some concerns about copying memory here
    y = Zy[-1, :, :].flatten(order="F")
    # Transpose back
    Z = Zy[:-1, :, :].transpose((1, 2, 0)).reshape(Uv.shape, order="F")
    # [MRR, Algorithm 1 - step 4]
    return diagonal_solve(
        M2d_invdiag,
        #        [MRR, Algorithm 1 - step 3]
        y + (Z @ np.linalg.solve(diagonal_add(-VvT @ Z, 1.0), VvT @ y)),
        True,
    )


def get_random_rhs_2D_inversion(D, M_diag, lambda_x, lambda_y):
    r"""Get a rhs for the problem [MRR, (B.1)], `L_{2d}(M \ kron M)x=rhs`. The idea is
    to build the matrix, generate a random `x` and compute the rhs accordingly.

    D: np.ndarray
        Derivative matrix [MRR, (6)]
    M_diag: np.ndarray
        Diagonal of the mass matrix, see [MRR, (17)]
    lambda_x, lambda_y: float
        Ratio between celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        A vector to be used as rhs and the solution
    """
    # To choose a common rhs, we compute the matrix and choose a random solution
    # Building the inverse matrix
    mat = diagonal_matrix_multiply_right(
        L2d_matrix(D, lambda_x, lambda_y), diagonal_auto_kron(M_diag)
    )
    sol = np.random.rand(mat.shape[-1])
    return mat @ sol, sol


def get_random_rhs_2D_inversion_visosity(D, M_diag, lambda_x, lambda_y):
    r"""Get a rhs for the problem [MRR, (B.2)], `L^{v}_{2d}(M \ kron M)x=rhs`. The idea
    is to build the matrix, generate a random `x` and compute the rhs accordingly.

    D: np.ndarray
        Derivative matrix [MRR, (6)]
    M_diag: np.ndarray
        Diagonal of the mass matrix, see [MRR, (17)]
    lambda_x, lambda_y: float
        Ratio between celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        A vector to be used as rhs and the solution
    """
    p = D.shape[0] - 1
    d_min = d_min_BY_ORDER[p]
    # See [MRR, (45-46)]
    Omega = np.asarray(LOBATTO_WEIGHTS_BY_ORDER[p]).reshape((p + 1, 1))
    Uv = (
        2
        * d_min
        * np.concatenate(
            (lambda_x * I_kron_mat(Omega), lambda_y * np.kron(Omega, np.eye(*D.shape))),
            axis=1,
        )
    )
    Vv = np.concatenate(
        (
            I_kron_mat(np.ones((p + 1, 1))),
            np.kron(np.ones((p + 1, 1)), np.eye(*D.shape)),
        ),
        axis=1,
    )
    mat = diagonal_matrix_multiply_right(
        diagonal_add(
            L2d_matrix(D, lambda_x, lambda_y), 2 * d_min * (lambda_x + lambda_y)
        )
        - Uv @ Vv.T,
        diagonal_auto_kron(M_diag),
    )
    sol = np.random.rand(mat.shape[-1])
    return mat @ sol, sol


def compare_eigenvalues_computation(p):
    """Compare the computation of the eigenvalues of the L matrix with numpy function
    and analytical formula

    Reference:
        Compare methods for performing diagonalization [MRR, (38)]

    p: int
        Polynomial order

    Return:
        The eigenvalues
    """
    # Main matrix with derivatives
    D = D_matrix(p)  # [MRR, (6)]

    # Computation of L eigenvalues and eigenvectors with numpy
    eigValNp = eigen_L_numpy(D)

    # "Analytical" computation of L eigenvalues and eigenvectors
    Psi = eigen_L_analytical(D)

    # print("L eigenval.with numpy: {}".format(eigValNp))
    # print("Semi-analytical L eigenval.: {}\n".format(Psi))
    print(
        "Verification of L eigenvalues (norm of the difference between the two methods): {}\n".format(
            np.linalg.norm(eigValNp - Psi)
        )
    )
    return Psi


def compare_2D_inversion(p, lambda_x, lambda_y):
    """Compare standard and fast methods to invert 2D systems resulting from DGSEM
    problems.

    Reference:
        Compare methods for computing LHS of [MRR, (43)], see also [MRR, (B.1)]

    p: int
        Polynomial order
    lambda_x, lambda_y: float
        Ratio between celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        The inverse of the 2D matrix
    """

    # Main matrices
    D = D_matrix(p)  # [MRR, (6)]
    M_diag = mass_matrix_diag(p)  # [MRR, (17)]

    # Diagonal block inversion with numpy
    L2d_numpyInv = L2d_inversion_numpy(D, M_diag, lambda_x, lambda_y)

    # Explicit diagonal block inversion
    L2d_explInv = L2d_inversion_analytical(D, M_diag, lambda_x, lambda_y)

    print(
        "Verification of diag. block inv. (norm of the difference between the two methods): {}".format(
            np.linalg.norm(L2d_numpyInv - L2d_explInv)
        )
    )

    ##############################################
    # Comparison using Fast Diagonalization Method
    ##############################################
    rhs, sol_ref = get_random_rhs_2D_inversion(D, M_diag, lambda_x, lambda_y)
    sol_numpy = L2d_numpyInv @ rhs
    sol_FDM = L2d_inversion_analytical_FDM(D, M_diag, lambda_x, lambda_y, rhs)
    print("Verification of solutions:")
    print(
        "  - Norm of the difference between the two methods: {}".format(
            np.linalg.norm(sol_numpy - sol_FDM)
        )
    )
    print(
        "  - Norm of the difference wrt reference solution: {}\n".format(
            np.linalg.norm(sol_ref - sol_FDM)
        )
    )
    return L2d_numpyInv


def compare_2D_inversion_viscosity(p, lambda_x, lambda_y):
    """Compare standard and fast methods to invert 2D systems resulting from DGSEM
    problems with graph-viscosity

    Reference:
        Compare methods for inverting [MRR, (45)], see also [MRR, (B.2)]

    p: int
        Polynomial order
    lambda_x, lambda_y: float
        Ratio between celerity*time step over spatial grid size in, respectively, in x
        and y direction

    Return:
        The inverse of the matrix
    """

    # Main matrices
    D = D_matrix(p)  # [MRR, (6)]
    M_diag = mass_matrix_diag(p)
    M2d_invdiag = np.reciprocal(diagonal_auto_kron(M_diag))

    L2dV_numpyInv = L2d_inversion_viscosity_numpy(D, M2d_invdiag, lambda_x, lambda_y)
    L2dV_explInv = L2d_inversion_viscosity_analytical(
        D, M2d_invdiag, lambda_x, lambda_y
    )

    print(
        "Verification of diag. block inv. with graph visc. (norm of the difference between the two methods): {}".format(
            np.linalg.norm(L2dV_numpyInv - L2dV_explInv)
        )
    )
    ##############################################
    # Comparison using Fast Diagonalization Method
    ##############################################
    rhs, sol_ref = get_random_rhs_2D_inversion_visosity(D, M_diag, lambda_x, lambda_y)
    sol_numpy = L2dV_numpyInv @ rhs
    sol_FDM = L2d_inversion_viscosity_analytical_FDM(
        D, M2d_invdiag, lambda_x, lambda_y, rhs
    )
    print("Verification of solutions:")
    print(
        "  - Norm of the difference between the two methods: {}".format(
            np.linalg.norm(sol_numpy - sol_FDM)
        )
    )
    print(
        "  - Norm of the difference wrt reference solution: {}\n".format(
            np.linalg.norm(sol_ref - sol_FDM)
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
