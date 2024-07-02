import numpy as np
import pytest

import fast_DGSEM_block_inversion as f_dgsem


@pytest.mark.parametrize("p", [2, 3, 4, 5, 6], ids=lambda p: f"Order: {p}")
class TestMethods:
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
            assert np.allclose(f_dgsem.diagonal_matrix_multiply(diag, B), A.dot(B))
        else:
            assert np.allclose(
                B.dot(A), f_dgsem.diagonal_matrix_multiply_right(B, diag)
            )

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
