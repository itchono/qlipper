import logging
from functools import partial
from typing import Any, Callable

import jax.numpy as jnp
from diffrax import CubicInterpolation, backward_hermite_coefficients
from jax import Array, jit

from qlipper.configuration import SimConfig
from qlipper.sim import Params
from qlipper.sim.dymamics_mee import dyn_mee
from qlipper.sim.dynamics_cartesian import dyn_cartesian
from qlipper.sim.ephemeris import generate_ephem_arrays, lookup_body_id
from qlipper.sim.perturbations import PERTURBATIONS
from qlipper.sim.propulsion import PROPULSION_MODELS
from qlipper.steering import STEERING_LAWS

logger = logging.getLogger(__name__)


def prebake_sim_config(cfg: SimConfig) -> Params:
    """
    Prebake the SimConfig struct into a SimInternalConfig struct
    that can be passed into the actual problem being solved.

    TODO: make the ephemeris generation available as its
    own function, so that it can be reused in other places
    """

    # Generate ephemeris interpolant arrays
    sun = lookup_body_id("sun")
    earth = lookup_body_id("earth")
    moon = lookup_body_id("moon")

    # Heuristic: 300 samples per year
    num_ephem_samples = max(int((cfg.t_span[1] - cfg.t_span[0]) / 86400 / 365 * 300), 2)

    # TODO: only generate ephem interpolants for the bodies that are actually used

    logger.info(
        "Generating ephemeris interpolant arrays - "
        f"{num_ephem_samples} samples will be used"
    )

    # Generate sun ephemeris interpolant
    sun_t_sample, sun_state_sample = generate_ephem_arrays(
        earth, sun, cfg.epoch_jd, cfg.t_span, num_ephem_samples
    )
    sun_state_sample = sun_state_sample * 1e3  # convert from km to m
    interp_coeffs = backward_hermite_coefficients(sun_t_sample, sun_state_sample.T)
    sun_ephem = CubicInterpolation(sun_t_sample, interp_coeffs)

    # Generate moon ephemeris interpolant
    moon_t_sample, moon_state_sample = generate_ephem_arrays(
        earth, moon, cfg.epoch_jd, cfg.t_span, num_ephem_samples
    )
    moon_state_sample = moon_state_sample * 1e3  # convert from km to m
    interp_coeffs = backward_hermite_coefficients(moon_t_sample, moon_state_sample.T)
    moon_ephem = CubicInterpolation(moon_t_sample, interp_coeffs)

    return Params(
        y_target=cfg.y_target,
        conv_tol=cfg.conv_tol,
        w_oe=cfg.w_oe,
        w_penalty=cfg.w_penalty,
        pen_param=1.0,
        kappa=jnp.deg2rad(cfg.kappa),
        characteristic_accel=cfg.characteristic_accel,
        epoch_jd=cfg.epoch_jd,
        sun_ephem=sun_ephem,
        moon_ephem=moon_ephem,
    )


def prebake_ode(cfg: SimConfig) -> Callable[[float, Array, Any], Array]:
    """
    Bake a version of the ode so that it can be JIT-compiled

    Parameters
    ----------
    ode : Callable[[float, Array, Any, Any], Array]
        The original ODE function
    cfg : SimConfig
        The simulation configuration

    Returns
    -------
    baked_ode : Callable[[float, Array, Any], Array]
        The baked ODE function
    """

    steering_law = STEERING_LAWS[cfg.steering_law]
    propulsion_model = PROPULSION_MODELS[cfg.propulsion_model]

    dynamics_options: dict[str, Callable[[float, Array, Any, Any], Array]] = {
        "mee": dyn_mee,
        "cartesian": dyn_cartesian,
    }

    ode = dynamics_options[cfg.dynamics]

    baked_ode = jit(
        partial(
            ode,
            steering_law=steering_law,
            propulsion_model=propulsion_model,
            perturbations=[PERTURBATIONS[p] for p in cfg.perturbations],
        )
    )

    return baked_ode
