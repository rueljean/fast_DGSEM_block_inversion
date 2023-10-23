import numpy as np
import pytest

import fast_DGSEM_block_inversion as f_dgsem


@pytest.mark.parametrize("p", [2, 3, 4, 5, 6])
class TestMethods:
    def test_eigen(self, p):
        D = f_dgsem.D_matrix(p)
        assert np.allclose(f_dgsem.eigen_L_numpy(D), f_dgsem.eigen_L_analytical(D))

    @pytest.mark.parametrize("lambda_x", [1.0, 2.0])
    @pytest.mark.parametrize("lambda_y", [1.0])
    def test_2D_inversion(self, p, lambda_x, lambda_y):
        D = f_dgsem.D_matrix(p)
        M = 0.5 * np.diag(f_dgsem.LOBATTO_WEIGHTS_BY_ORDER[p])
        assert np.allclose(
            f_dgsem.L2d_inversion_numpy(D, M, lambda_x, lambda_y),
            f_dgsem.L2d_inversion_analytical(D, M, lambda_x, lambda_y),
        )

    @pytest.mark.parametrize("lambda_x", [1.0, 2.0])
    @pytest.mark.parametrize("lambda_y", [1.0])
    def test_2D_inversion_viscosity(self, p, lambda_x, lambda_y):
        D = f_dgsem.D_matrix(p)
        M = 0.5 * np.diag(f_dgsem.LOBATTO_WEIGHTS_BY_ORDER[p])
        assert np.allclose(
            f_dgsem.L2d_inversion_viscosity_numpy(D, M, lambda_x, lambda_y),
            f_dgsem.L2d_inversion_viscosity_analytical(D, M, lambda_x, lambda_y),
        )


class TestMatrix:
    # Default dimension
    N = 5
    # Default random number generator
    rng = np.random.default_rng()

    def test_diagonal_matrix_multiply(self):
        diag = self.rng.random(self.N)
        A = np.diag(diag)
        B = self.rng.random((self.N, self.N))
        assert np.allclose(f_dgsem.diagonal_matrix_multiply(diag, B), A.dot(B))

    def test_diagonal_auto_kron(self):
        diag = self.rng.random(self.N)
        A = np.diag(diag)
        assert np.allclose(np.diag(f_dgsem.diagonal_auto_kron(diag)), np.kron(A, A))

    @pytest.mark.parametrize("val", [3.0, None])
    def test_diagonal_add(self, val):
        A = self.rng.random((self.N, self.N))
        d_val = val * np.ones(self.N) if val is not None else self.rng.random(self.N)
        assert np.allclose(f_dgsem.diagonal_add(A, d_val), A + np.diag(d_val))

    @pytest.mark.parametrize("shape_A", [(3, 3), (3, 4)])
    @pytest.mark.parametrize("dim_I", [None, 2, 5])
    def test_I_kron_mat(self, shape_A, dim_I):
        A = self.rng.random(shape_A)
        I = np.eye(dim_I if dim_I is not None else shape_A[0])
        assert np.allclose(f_dgsem.I_kron_mat(A, dim_I), np.kron(I, A))
