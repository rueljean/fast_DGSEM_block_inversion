import numpy as np
import pytest

import fast_DGSEM_block_inversion as f_dgsem


@pytest.mark.parametrize("p", [2, 3, 4, 5])
@pytest.mark.parametrize("lambda_x", [1.0, 2.0])
@pytest.mark.parametrize("lambda_y", [1.0, 1.0])
class TestMethods:
    def test_eigen(self, p, lambda_x, lambda_y):
        # lambda_x/y not useful here, but we keep to have the parametrize
        D = f_dgsem.D_matrix(p)
        assert np.allclose(f_dgsem.eigen_L_numpy(D), f_dgsem.eigen_L_analytical(D))

    def test_2D_inversion(self, p, lambda_x, lambda_y):
        D = f_dgsem.D_matrix(p)
        M = 0.5 * np.diag(f_dgsem.LOBATTO_WEIGHTS_BY_ORDER[p])
        assert np.allclose(
            f_dgsem.L2d_inversion_numpy(D, M, lambda_x, lambda_y),
            f_dgsem.L2d_inversion_analytical(D, M, lambda_x, lambda_y),
        )

    def test_2D_inversion_viscosity(self, p, lambda_x, lambda_y):
        D = f_dgsem.D_matrix(p)
        M = 0.5 * np.diag(f_dgsem.LOBATTO_WEIGHTS_BY_ORDER[p])
        assert np.allclose(
            f_dgsem.L2d_inversion_viscosity_numpy(D, M, lambda_x, lambda_y),
            f_dgsem.L2d_inversion_viscosity_analytical(D, M, lambda_x, lambda_y),
        )
