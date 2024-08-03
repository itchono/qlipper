import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from qlipper.constants import R_EARTH
from qlipper.sim import Params


@jax.jit
def simple_eclipse(t: ArrayLike, y: ArrayLike, params: Params) -> bool:
    """
    Check if the spacecraft is in eclipse, using
    the simple method in Curtis.

    Parameters
    ----------
    t : float
        Time.
    y : Array
        State vector (Cartesian).
    params : Params
        Simulation parameters.

    Returns
    -------
    bool
        Whether the spacecraft is in eclipse.
    """

    r_spacecraft = y[:3]
    r_sun = params.sun_ephem.evaluate(t)

    mag_r_spacecraft = jnp.linalg.norm(r_spacecraft)
    mag_r_sun = jnp.linalg.norm(r_sun)

    theta = jnp.arccos(r_spacecraft @ r_sun / (mag_r_sun * mag_r_spacecraft))

    theta_spacecraft = jnp.arccos(R_EARTH / mag_r_spacecraft)
    theta_sun = jnp.arccos(R_EARTH / mag_r_sun)

    return (theta_spacecraft + theta_sun) <= theta
