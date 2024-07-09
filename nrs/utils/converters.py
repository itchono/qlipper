import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def lvlh_to_steering(dir_lvlh: ArrayLike) -> tuple[float, float]:
    """
    Convert direction vector in LVLH frame to steering angles.

    Parameters
    ----------
    dir_lvlh : ArrayLike
        Direction vector in LVLH frame.

    Returns
    -------
    alpha : float
        Steering angle in the y-x plane [rad].
    beta : float
        Steering angle towards the z-axis [rad].
    """
    beta = jnp.atan2(dir_lvlh[2], jnp.linalg.norm(dir_lvlh[:2]))
    alpha = jnp.arctan2(dir_lvlh[0], dir_lvlh[1])
    return alpha, beta


def steering_to_lvlh(alpha: float, beta: float) -> Array:
    """
    Convert steering angles to direction vector in LVLH frame.

    Parameters
    ----------
    alpha : float
        Steering angle in the y-x plane [rad].
    beta : float
        Steering angle towards the z-axis [rad].

    Returns
    -------
    dir_lvlh : Array
        Direction vector in LVLH frame.
    """
    dir_lvlh = jnp.array(
        [
            jnp.cos(beta) * jnp.sin(alpha),
            jnp.cos(beta) * jnp.cos(alpha),
            jnp.sin(beta),
        ]
    )
    return dir_lvlh
