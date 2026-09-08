import numpy as np

from crystal_field.crystallography.geometry import reciprocal_metric_from_parameters


def test_orthogonal_reciprocal_metric():
    g = reciprocal_metric_from_parameters(10, 20, 40, 90, 90, 90)
    np.testing.assert_allclose(g, np.diag([1 / 100, 1 / 400, 1 / 1600]), atol=1e-12)
