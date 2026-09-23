import math
import numpy as np
from piar.physics import PIAR
from piar import chaos as C


def test_lyapunov_sum_rule_and_reference_values():
    for A, expected_sign in ((0.40, -1), (0.80, +1)):
        p = PIAR.dimensionless(-0.8, gamma=0.25, A=A, Omega=2 / 3)
        l1, l2, run = C.lyapunov(p, n_transient=200, n_periods=1500)
        assert abs((l1 + l2) + 0.25) < 1e-6          # lambda1 + lambda3 = -gamma
        assert np.sign(l1) == expected_sign


def test_poincare_determinant():
    p = PIAR.dimensionless(-0.8, gamma=0.25, A=0.8, Omega=2 / 3)
    mult, M = C.floquet_multipliers(p, 0.3, 0.1, period=1)
    assert abs(np.linalg.det(M) - math.exp(-2 * math.pi * 0.25 / (2 / 3))) < 1e-8


def test_melnikov_pendulum_limit():
    r, I0, I1 = C.melnikov_ratio(1e-12, 2 / 3)
    assert abs(r - 4 / math.pi * math.cosh(math.pi * (2 / 3) / 2)) < 1e-4
    assert abs(I0 - 8.0) < 1e-6
