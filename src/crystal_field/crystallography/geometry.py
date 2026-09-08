from __future__ import annotations

import math

import numpy as np


def direct_metric_from_parameters(a, b, c, alpha_deg, beta_deg, gamma_deg):
    alpha, beta, gamma = map(math.radians, (alpha_deg, beta_deg, gamma_deg))
    return np.array([
        [a*a, a*b*math.cos(gamma), a*c*math.cos(beta)],
        [a*b*math.cos(gamma), b*b, b*c*math.cos(alpha)],
        [a*c*math.cos(beta), b*c*math.cos(alpha), c*c],
    ], dtype=np.float64)


def reciprocal_metric_from_parameters(a, b, c, alpha_deg, beta_deg, gamma_deg):
    return np.linalg.inv(direct_metric_from_parameters(a, b, c, alpha_deg, beta_deg, gamma_deg))


def cell_tuple(cell):
    return (float(cell.a), float(cell.b), float(cell.c), float(cell.alpha), float(cell.beta), float(cell.gamma))
