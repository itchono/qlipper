import jax.numpy as jnp
from jax import Array, jit, lax
from jax.typing import ArrayLike

from qlipper.converters import rot_inertial_lvlh, steering_to_lvlh
from qlipper.sim import Params
from qlipper.sim.eclipse import simple_eclipse


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


@jit
def ideal_solar_sail(
    t: float, y: ArrayLike, params: Params, alpha: float, beta: float
) -> Array:
    """
    Ideal solar sail model. Includes a basic occlusion model.

    Output vector is in LVLH frame.

    Parameters
    ----------
    t : float
        Time since epoch (s).
    y : ArrayLike
        Cartesian state vector [m, m/s].
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

    r_spacecraft_i = y[0:3]

    sc_dir_lvlh = steering_to_lvlh(alpha, beta)
    sc_dir_i = rot_inertial_lvlh(y) @ sc_dir_lvlh
    r_sun_i = params.sun_ephem.evaluate(t)[:3]

    r_rel_sun_i = r_sun_i - r_spacecraft_i

    sunlight_dir_i = -r_rel_sun_i / jnp.linalg.norm(r_rel_sun_i)
    c_cone_ang = sc_dir_i @ sunlight_dir_i

    eclipsed = simple_eclipse(t, y, params)

    acc_lvlh = (
        params.characteristic_accel * jnp.sign(c_cone_ang) * c_cone_ang**2 * sc_dir_lvlh
    )

    return lax.cond(
        eclipsed,
        lambda _: jnp.zeros(3),
        lambda _: acc_lvlh,
        None,
    )


PROPULSION_MODELS = {
    "constant_thrust": constant_thrust,
    "ideal_solar_sail": ideal_solar_sail,
}
