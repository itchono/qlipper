from jax import Array, jit
from jax.typing import ArrayLike

from qlipper.converters import steering_to_lvlh
from qlipper.sim import Params


@jit
def constant_thrust(
    t: float, y: ArrayLike, params: Params, alpha: float, beta: float
) -> Array:
    """
    Constant thrust model; always returns
    thrust in the direction of the spacecraft.

    Output vector is in LVLH frame.

    Parameters
    ----------
    t : float
        Time since epoch (s). [Unused]
    y : ArrayLike
        State vector [Unused, can be anything]
    params : Params
        Sim parameters
    alpha : float
        Steering angle in the y-x plane [rad].
    beta : float
        Steering angle towards the z-axis [rad].

    Returns
    -------
    acc_lvlh : Array
        Thrust vector in LVLH frame.
    """
    sc_dir_lvlh = steering_to_lvlh(alpha, beta)

    return params.characteristic_accel * sc_dir_lvlh


PROPULSION_MODELS = {
    "constant_thrust": constant_thrust,
}
