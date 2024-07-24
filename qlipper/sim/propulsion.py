from jax import Array
from jax.typing import ArrayLike

from qlipper.configuration import SimConfig
from qlipper.utils.converters import steering_to_lvlh


def constant_thrust(
    t: float, y: ArrayLike, cfg: SimConfig, alpha: float, beta: float
) -> Array:
    """
    Constant thrust model.

    Parameters
    ----------
    t : float
        Time since epoch (s).
    y : ArrayLike
        State vector in modified equinoctial elements.
    cfg : SimConfig
        Configuration object.
    alpha : float
        Steering angle in the y-x plane [rad].
    beta : float
        Steering angle towards the z-axis [rad].

    Returns
    -------
    sc_dir_lvlh : Array
        Thrust vector in LVLH frame.
    """
    sc_dir_lvlh = steering_to_lvlh(alpha, beta)

    return cfg.characteristic_accel * sc_dir_lvlh
