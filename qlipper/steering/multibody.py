import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from qlipper.constants import MU_EARTH, MU_MOON
from qlipper.converters import cartesian_to_mee, lvlh_to_steering, steering_to_lvlh
from qlipper.run.prebake import Params
from qlipper.steering.q_law import _q_law_mee, _rq_law_mee


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


@jax.jit
def slerp(a: ArrayLike, b: ArrayLike, t: float) -> ArrayLike:
    """
    Spherical linear interpolation between two vectors.

    Parameters
    ----------
    a : ArrayLike
        First vector.
    b : ArrayLike
        Second vector.
    t : float
        Interpolation factor.
        0 = a, 1 = b.

    Returns
    -------
    ArrayLike
        Interpolated vector.
    """

    # Ensure the vectors are unit vectors
    v0 = a / jnp.linalg.norm(a)
    v1 = b / jnp.linalg.norm(b)

    # Compute the cosine of the angle between the vectors
    # Clamp dot product to avoid numerical errors leading to values slightly out of range
    dot = jnp.clip(v0 @ v1, -1.0, 1.0)

    # Compute the angle between the vectors
    theta_0 = jnp.arccos(dot)  # theta_0 is the angle between v0 and v1
    sin_theta_0 = jnp.sin(theta_0)

    # Compute the interpolation
    theta = theta_0 * t
    sin_theta = jnp.sin(theta)

    s0 = jnp.sin((1.0 - t) * theta_0) / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return s0 * v0 + s1 * v1


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
    angles_moon = _rq_law_mee(
        mee_rel_moon, params.y_target, params.w_oe, params.characteristic_accel, MU_MOON
    )

    # Blend
    n_earth = steering_to_lvlh(*angles_earth)
    n_moon = steering_to_lvlh(*angles_moon)

    b = blending_weight(t, y, params)  # 0 = Moon, 1 = Earth

    # chosen_dir = slerp(n_earth, n_moon, b)

    chosen_dir = jax.lax.cond(b < 0.7, lambda: n_moon, lambda: n_earth)

    return lvlh_to_steering(chosen_dir)
