import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from qlipper.constants import MU_EARTH, MU_MOON, R_EARTH, R_MOON
from qlipper.converters import cartesian_to_mee, lvlh_to_steering, steering_to_lvlh
from qlipper.run.prebake import Params
from qlipper.steering.q_law import (
    _q_law_mee,
    _rq_law_mee,
    approx_max_roc,
    gve_coefficients,
    periapsis_penalty,
)


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

    p = 2  # fall-off factor

    u1 = MU_EARTH / (jnp.linalg.norm(r_rel_earth) ** p)
    u2 = MU_MOON / (jnp.linalg.norm(r_rel_moon) ** p)

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


def qbbq_law(t: float, y: ArrayLike, params: Params) -> tuple[float, float]:
    """
    Q-law with blending at the Q function level, not yet working.

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

    # EARTH
    S_earth = jnp.array([1 / R_EARTH, 1, 1, 1, 1])
    d_oe_max_earth = approx_max_roc(
        mee_rel_earth, params.characteristic_accel, MU_EARTH
    )
    A_earth, _ = gve_coefficients(mee_rel_earth, MU_EARTH)

    # Augment target set
    L_curr = mee_rel_earth[5] % (2 * jnp.pi)
    L_target = mee_of_moon[5] % (2 * jnp.pi)
    delta_L = jnp.atan2(
        jnp.sin(L_target - L_curr), jnp.cos(L_target - L_curr)
    )  # [-pi, pi]
    augmentation = (
        2 / jnp.pi * jnp.arctan(delta_L) * mee_of_moon[0] * 0.1
    )  # jank, prevents the change from being too large
    target_earth = mee_of_moon.at[0].add(augmentation)

    Xi_E_earth = 2 * (mee_rel_earth[:5] - target_earth[:5]) / d_oe_max_earth

    # periapsis constraint
    P, dPdoe = jax.value_and_grad(periapsis_penalty)(mee_rel_earth, R_EARTH, 2)
    Xi_P_earth = (
        dPdoe[:5] * ((mee_rel_earth[:5] - target_earth[:5]) / d_oe_max_earth) ** 2
    )

    D_earth = A_earth[:5, :].T @ (
        params.w_oe
        * S_earth
        * (params.w_penalty * Xi_P_earth + (1 + params.w_penalty * P) * Xi_E_earth)
    )

    # MOON
    S_moon = jnp.array([1 / R_MOON, 1, 1, 1, 1])
    d_oe_max_moon = approx_max_roc(mee_rel_moon, params.characteristic_accel, MU_MOON)
    A_moon, _ = gve_coefficients(mee_rel_moon, MU_MOON)
    Xi_E_moon = 2 * (mee_rel_moon[:5] - params.y_target[:5]) / d_oe_max_moon

    # periapsis constraint
    P, dPdoe = jax.value_and_grad(periapsis_penalty)(mee_rel_moon, R_MOON, 5)
    Xi_P_moon = (
        dPdoe[:5] * ((mee_rel_moon[:5] - params.y_target[:5]) / d_oe_max_moon) ** 2
    )

    D_moon = A_moon[:5, :].T @ (
        params.w_oe
        * S_moon
        * (params.w_penalty * Xi_P_moon + (1 + params.w_penalty * P) * Xi_E_moon)
    )

    # blend
    b = blending_weight(t, y, params)  # 0 = Moon, 1 = Earth

    D = (1 - b) * D_moon + b * D_earth

    alpha = jnp.atan2(-D[0], -D[1])
    beta = jnp.atan2(-D[2], jnp.linalg.norm(D[:2]))
    return alpha, beta


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
    angles_earth = _rq_law_mee(
        mee_rel_earth,
        mee_of_moon,
        params.w_oe,
        params.characteristic_accel,
        MU_EARTH,
    )

    # Moon guidance
    MOON_W_OE = jnp.array([1, 1, 1, 1, 1])
    angles_moon = _q_law_mee(
        mee_rel_moon, params.y_target, MOON_W_OE, params.characteristic_accel, MU_MOON
    )

    # Blend
    n_earth = steering_to_lvlh(*angles_earth)
    n_moon = steering_to_lvlh(*angles_moon)

    b = blending_weight(t, y, params)  # 0 = Moon, 1 = Earth

    chosen_dir = slerp(n_moon, n_earth, b)

    return lvlh_to_steering(chosen_dir)
