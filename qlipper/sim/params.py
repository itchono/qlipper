from typing import NamedTuple

from diffrax import CubicInterpolation
from jax import Array


class Params(NamedTuple):
    """
    Subset of SimConfig struct passed into the actual
    problem being solved; only contains the necessary
    information needed as the ODE level for solving

    ONLY JITTABLE TYPES ALLOWED
    """

    y_target: Array
    conv_tol: float
    w_oe: Array
    w_penalty: float
    kappa: float
    characteristic_accel: float
    epoch_jd: float
    sun_ephem: CubicInterpolation  # meters
    moon_ephem: CubicInterpolation  # meters
