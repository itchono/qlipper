import jax.numpy as jnp
from jax.lax import cond
from jax.typing import ArrayLike

from qlipper.constants import MU_EARTH, MU_MOON
from qlipper.converters import cartesian_to_mee, lvlh_to_steering, steering_to_lvlh
from qlipper.run.prebake import Params
from qlipper.steering.q_law import _q_law_mee


def blending_weight(t: float, y: ArrayLike, params: Params) -> float:
    """
    Blending weight for multibody guidance.

    Parameters
    ----------
    t : float
        Time.
    y : ArrayLike
        Current state vector.
    params : Params
        Simulation parameters.

    Returns
    -------
    float
        0-1 value for using the Moon vs Earth guidance law.
    """

    moon_position = params.moon_ephem.evaluate(t)[:3]

    r_rel_moon = y[:3] - moon_position
    r_rel_earth = y[:3]

    u1 = MU_EARTH / jnp.linalg.norm(r_rel_earth) ** 2
    u2 = MU_MOON / jnp.linalg.norm(r_rel_moon) ** 2

    b = u1 / (u1 + u2)  # how much to use Earth guidance

    return b


def bbq_law(t: float, y: ArrayLike, params: Params) -> tuple[float, float]:
    """
    BBQ-Law, for multibody rendezvous.

    Parameters
    ----------
    t : float
        Time.
    y : ArrayLike
        Current state vector.
    params : Params
        Simulation parameters.

    Returns
    -------
    alpha : float
        Steering angle.
    beta : float
        Steering angle.
    """

    moon_state = params.moon_ephem.evaluate(t)

    mee_of_moon = cartesian_to_mee(moon_state, MU_EARTH)
    mee_rel_moon = cartesian_to_mee(y - moon_state, MU_MOON)
    mee_rel_earth = cartesian_to_mee(y, MU_EARTH)

    # Earth guidance
    angles_earth = _q_law_mee(
        mee_rel_earth,
        mee_of_moon,
        params.w_oe,
        params.characteristic_accel,
        MU_EARTH,
    )

    # Moon guidance
    angles_moon = _q_law_mee(
        mee_rel_moon, params.y_target, params.w_oe, params.characteristic_accel, MU_MOON
    )

    # Blend
    n_earth = steering_to_lvlh(*angles_earth)
    n_moon = steering_to_lvlh(*angles_moon)

    b = blending_weight(t, y, params)

    # n = b * n_earth + (1 - b) * n_moon  # TODO: this must be SLERPED

    chosen_dir = cond(b < 0.7, lambda: n_moon, lambda: n_earth)

    return lvlh_to_steering(chosen_dir)
