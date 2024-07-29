import jax.numpy as jnp
from jax.lax import cond
from jax.typing import ArrayLike

from qlipper.converters import (
    lvlh_to_steering,
    mee_to_cartesian,
    rot_inertial_lvlh,
    steering_to_lvlh,
)
from qlipper.run.prebake import Params
from qlipper.sim.ephemeris import interp_position
from qlipper.steering.q_law import q_law


def cone_adaptation(
    t: float, y: ArrayLike, params: Params, alpha_star: float, beta_star: float
):
    """
    Cone angle adaptation heuristic.
    """
    C_IO = rot_inertial_lvlh(y)
    C_OI = C_IO.T

    n_star_i = C_IO @ steering_to_lvlh(alpha_star, beta_star)

    r_spacecraft_i = mee_to_cartesian(y)[0:3]
    r_sun_i = interp_position(t, params.ephem_t_sample, params.ephem_r_sample)
    r_rel_sun_i = r_sun_i - r_spacecraft_i
    u_i = -r_rel_sun_i / jnp.linalg.norm(r_rel_sun_i)

    c_cone_ang = n_star_i @ u_i

    b_i = jnp.cross(u_i, jnp.cross(n_star_i, u_i))

    n_actual_i = cond(
        c_cone_ang < 0,
        lambda: n_star_i,
        lambda: cond(
            c_cone_ang < jnp.cos(params.kappa),
            lambda: jnp.cos(params.kappa) * u_i + jnp.sin(params.kappa) * b_i,
            lambda: n_star_i,
        ),
    )

    return lvlh_to_steering(C_OI @ n_actual_i)


def quail(t: float, y: ArrayLike, params: Params) -> tuple[float, float]:
    """
    Q-law using angle of incidence limits.
    """
    # first stage

    alpha_star, beta_star = q_law(t, y, params)

    # second stage
    alpha, beta = cone_adaptation(t, y, params, alpha_star, beta_star)

    return alpha, beta
