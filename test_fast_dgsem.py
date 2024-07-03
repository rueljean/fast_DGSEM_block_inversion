import numpy as np
import pytest

import fast_DGSEM_block_inversion as f_dgsem


@pytest.mark.parametrize("p", [2, 3, 4, 5, 6], ids=lambda p: f"Order: {p}")
class TestMethods:
    # Default random number generator
    rng = np.random.default_rng()

    def test_eigen(self, p):
        D = f_dgsem.D_matrix(p)
        assert np.allclose(f_dgsem.eigen_L_numpy(D), f_dgsem.eigen_L_analytical(D))

    @pytest.mark.parametrize("lambda_x", [1.0, 2.0], ids=lambda l: f"Lambda x: {l}")
    @pytest.mark.parametrize("lambda_y", [1.0], ids=lambda l: f"Lambda y: {l}")
    def test_2D_inversion(self, p, lambda_x, lambda_y):
        D = f_dgsem.D_matrix(p)
        M_diag = f_dgsem.mass_matrix_diag(p)
        assert np.allclose(
            f_dgsem.L2d_inversion_numpy(D, M_diag, lambda_x, lambda_y),
            f_dgsem.L2d_inversion_analytical(D, M_diag, lambda_x, lambda_y),
        )

    def get_random_rhs(self, D, M_diag, lambda_x, lambda_y):
        # To choose a common rhs, we compute the matrix and choose a random solution
        mat = f_dgsem.diagonal_matrix_multiply_right(
            f_dgsem.L2d_matrix(D, lambda_x, lambda_y),
            f_dgsem.diagonal_auto_kron(M_diag),
        )
        return mat @ self.rng.random(mat.shape[-1])

    @pytest.mark.parametrize("lambda_x", [1.0, 2.0], ids=lambda l: f"Lambda x: {l}")
    @pytest.mark.parametrize("lambda_y", [1.0], ids=lambda l: f"Lambda y: {l}")
    def test_2D_solution(self, p, lambda_x, lambda_y):
        D = f_dgsem.D_matrix(p)
        M_diag = f_dgsem.mass_matrix_diag(p)
        rhs = self.get_random_rhs(D, M_diag, lambda_x, lambda_y)
        assert np.allclose(
            f_dgsem.L2d_inversion_numpy(D, M_diag, lambda_x, lambda_y) @ rhs,
            f_dgsem.L2d_inversion_analytical_FDM(
                D, M_diag, lambda_x, lambda_y, rhs.reshape(D.shape, order="F")
            ).flatten(order="F"),
        )

    @pytest.mark.parametrize("lambda_x", [1.0, 2.0], ids=lambda l: f"Lambda x: {l}")
    @pytest.mark.parametrize("lambda_y", [1.0], ids=lambda l: f"Lambda y: {l}")
    def test_2D_inversion_viscosity(self, p, lambda_x, lambda_y):
        D = f_dgsem.D_matrix(p)
        M2d_invdiag = np.reciprocal(
            f_dgsem.diagonal_auto_kron(f_dgsem.mass_matrix_diag(p))
        )
        assert np.allclose(
            f_dgsem.L2d_inversion_viscosity_numpy(D, M2d_invdiag, lambda_x, lambda_y),
            f_dgsem.L2d_inversion_viscosity_analytical(
                D, M2d_invdiag, lambda_x, lambda_y
            ),
        )

    def get_random_rhs_viscosity(self, D, M_diag, lambda_x, lambda_y):
        # To choose a common rhs, we compute the matrix and choose a random solution
        p = D.shape[0] - 1
        d_min = f_dgsem.d_min_BY_ORDER[p]
        # See [MRR, (46-48)]
        Omega = np.asarray(f_dgsem.LOBATTO_WEIGHTS_BY_ORDER[p]).reshape((p + 1, 1))
        Uv = (
            2
            * d_min
            * np.concatenate(
                (
                    lambda_x * f_dgsem.I_kron_mat(Omega),
                    lambda_y * np.kron(Omega, np.eye(*D.shape)),
                ),
                axis=1,
            )
        )
        Vv = np.concatenate(
            (
                f_dgsem.I_kron_mat(np.ones((p + 1, 1))),
                np.kron(np.ones((p + 1, 1)), np.eye(*D.shape)),
            ),
            axis=1,
        )
        mat = f_dgsem.diagonal_matrix_multiply_right(
            f_dgsem.diagonal_add(
                f_dgsem.L2d_matrix(D, lambda_x, lambda_y),
                2 * d_min * (lambda_x + lambda_y),
            )
            - Uv @ Vv.T,
            f_dgsem.diagonal_auto_kron(M_diag),
        )
        return mat @ self.rng.random(mat.shape[-1])

    @pytest.mark.parametrize("lambda_x", [1.0, 2.0], ids=lambda l: f"Lambda x: {l}")
    @pytest.mark.parametrize("lambda_y", [1.0], ids=lambda l: f"Lambda y: {l}")
    def test_2D_solution_viscosity(self, p, lambda_x, lambda_y):
        D = f_dgsem.D_matrix(p)
        M_diag = f_dgsem.mass_matrix_diag(p)
        M2d_invdiag = np.reciprocal(f_dgsem.diagonal_auto_kron(M_diag))
        rhs = self.get_random_rhs_viscosity(D, M_diag, lambda_x, lambda_y)
        assert np.allclose(
            f_dgsem.L2d_inversion_viscosity_numpy(D, M2d_invdiag, lambda_x, lambda_y)
            @ rhs,
            f_dgsem.L2d_inversion_viscosity_analytical_FDM(
                D, M2d_invdiag, lambda_x, lambda_y, rhs.reshape(D.shape, order="F")
            ),
        )


class TestMatrix:
    # Default dimension
    N = 5
    # Default random number generator
    rng = np.random.default_rng()

    @pytest.mark.parametrize(
        "left_prod", [True, False], ids=["Left product", "Right product"]
    )
    def test_diagonal_matrix_multiply(self, left_prod):
        diag = self.rng.random(self.N)
        A = np.diag(diag)
        B = self.rng.random((self.N, self.N))
        if left_prod:
            assert np.allclose(f_dgsem.diagonal_matrix_multiply(diag, B), A @ B)
        else:
            assert np.allclose(B @ A, f_dgsem.diagonal_matrix_multiply_right(B, diag))

    @pytest.mark.parametrize(
        "diag_already_inverted", [False, True], ids=lambda inv: f"Is inverted? {inv}"
    )
    @pytest.mark.parametrize(
        "rhs_is_matrix", [False, True], ids=lambda mat: f"Is RHS a matrix? {mat}"
    )
    def test_diagonal_solve(self, diag_already_inverted, rhs_is_matrix):
        diag = self.rng.random(self.N)
        # If diag is already inverted, then remake it normal so that solve works
        A = np.diag(np.reciprocal(diag) if diag_already_inverted else diag)
        B = self.rng.random((self.N, self.N) if rhs_is_matrix else self.N)
        assert np.allclose(
            f_dgsem.diagonal_solve(diag, B, diag_already_inverted),
            np.linalg.solve(A, B),
        )

    def test_diagonal_auto_kron(self):
        diag = self.rng.random(self.N)
        A = np.diag(diag)
        assert np.allclose(np.diag(f_dgsem.diagonal_auto_kron(diag)), np.kron(A, A))

    @pytest.mark.parametrize("val", [3.0, None], ids=lambda v: f"Added value: {v}")
    def test_diagonal_add(self, val):
        A = self.rng.random((self.N, self.N))
        d_val = val * np.ones(self.N) if val is not None else self.rng.random(self.N)
        assert np.allclose(f_dgsem.diagonal_add(A, d_val), A + np.diag(d_val))

    @pytest.mark.parametrize("shape_A", [(3, 3), (3, 4)], ids=lambda s: f"Shape: {s}")
    @pytest.mark.parametrize("dim_I", [None, 2, 5], ids=lambda d: f"Dim Identity: {d}")
    def test_I_kron_mat(self, shape_A, dim_I):
        A = self.rng.random(shape_A)
        I = np.eye(dim_I if dim_I is not None else shape_A[0])
        assert np.allclose(f_dgsem.I_kron_mat(A, dim_I), np.kron(I, A))
