from typing import Callable

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from qlipper.constants import MU_EARTH
from qlipper.converters import cartesian_to_mee, rot_inertial_lvlh
from qlipper.run.prebake import Params

CARTESIAN_DYN_SCALING = 1e3


def dyn_cartesian(
    t: float,
    y: ArrayLike,
    params: Params,
    steering_law: Callable[[float, Array, Params], tuple[float, float]],
    propulsion_model: Callable[[float, Array, Params, float, float], Array],
    perturbations: list[Callable[[float, Array, Params], Array]],
) -> Array:
    """
    Top level dynamics function for Carteisan state

    Parameters
    ----------
    t : float
        Time since epoch (s).
    y : ArrayLike
        State vector (Cartesian), x, y, z, vx, vy, vz.
    params : Params
        Configuration object.
    steering_law : Callable
        Control law, should be pre-baked with the ODE.
    propulsion_model : Callable
        Propulsion model, should be pre-baked with the ODE.
    perturbations : list[Callable]
        List of perturbation functions, should be pre-baked with the ODE.

    Returns
    -------
    dy_dt : Array
        Time derivative of state vector.

    """

    # Scaling
    y = y * CARTESIAN_DYN_SCALING

    # Control
    mee = cartesian_to_mee(y)
    alpha, beta = steering_law(t, mee, params)

    # Acceleration from propulsion (LVLH frame)
    acc_lvlh = propulsion_model(t, y, params, alpha, beta)

    # Perturbations
    for perturbation in perturbations:
        acc_lvlh += perturbation(t, y, params)

    # Newton's Second Law
    acc_inertial = rot_inertial_lvlh(y) @ acc_lvlh

    acc_gravity = -MU_EARTH * y[:3] / jnp.linalg.norm(y[:3]) ** 3

    dydt = jnp.concatenate([y[3:], acc_inertial + acc_gravity])
    dydt_scaled = dydt / CARTESIAN_DYN_SCALING

    return dydt_scaled
