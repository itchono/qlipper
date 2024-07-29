from typing import NamedTuple

from jax import Array


class Params(NamedTuple):
    """
    Subset of SimConfig struct passed into the actual
    problem being solved; only contains the necessary
    information needed as the ODE level for solving
    """

    y_target: Array
    conv_tol: float
    w_oe: Array
    w_penalty: float
    kappa: float
    characteristic_accel: float
    epoch_jd: float
    ephem_t_sample: float
    ephem_r_sample: Array
