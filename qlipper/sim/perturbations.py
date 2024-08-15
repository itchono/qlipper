import jax

from qlipper.constants import MU_MOON
from qlipper.converters import rot_lvlh_inertial
from qlipper.sim.params import Params


def moon_gravity(t: float, y: jax.Array, params: Params) -> jax.Array:
    """
    Gravity from the moon.

    Parameters
    ----------
    t : float
        Elapsed time (s)
    y : jax.Array
        State vector (Cartesian), x, y, z, vx, vy, vz [m, m/s]
    params : Params
        Mission parameters, including the ephemeris.

    Returns
    -------
    acc_lvlh : jax.Array
        Acceleration vector (m/s^2) in LVLH frame.

    """

    moon_position_i = params.moon_ephem.evaluate(t)[:3]

    r = y[:3] - moon_position_i

    acc_i = -MU_MOON * r / jax.numpy.linalg.norm(r) ** 3
    C_OI = rot_lvlh_inertial(y)

    return C_OI @ acc_i


def j2(t: float, y: jax.Array, params: Params) -> jax.Array:
    """
    J2 perturbation

    Parameters
    ----------
    t : float
        Elapsed time (s)
    y : jax.Array
        State vector (Cartesian), x, y, z, vx, vy, vz [m, m/s]
    params : Params
        Mission parameters, including the ephemeris.

    Returns
    -------
    acc_lvlh : jax.Array
        Acceleration vector (m/s^2) in LVLH frame.

    """
    pass
    # TODO: Implement J2 perturbation


PERTURBATIONS = {
    "moon_gravity": moon_gravity,
    "j2": j2,
}
