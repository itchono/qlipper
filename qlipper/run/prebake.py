import logging
from functools import partial
from typing import Any, Callable

import jax.numpy as jnp
from diffrax import CubicInterpolation, backward_hermite_coefficients
from jax import Array, jit

from qlipper.configuration import SimConfig
from qlipper.constants import P_SCALING
from qlipper.sim import Params
from qlipper.sim.ephemeris import generate_interpolant_arrays, lookup_body_id
from qlipper.sim.loss import norm_loss
from qlipper.sim.propulsion import PROPULSION_MODELS
from qlipper.steering import STEERING_LAWS

logger = logging.getLogger(__name__)


def prebake_sim_config(cfg: SimConfig) -> Params:
    """
    Prebake the SimConfig struct into a SimInternalConfig struct
    that can be passed into the actual problem being solved.
    """

    # Generate ephemeris interpolant arrays
    sun = lookup_body_id("sun")
    earth = lookup_body_id("earth")

    # Heuristic: 300 samples per year
    num_ephem_samples = int((cfg.t_span[1] - cfg.t_span[0]) / 86400 / 365 * 300)
    logger.info(
        "Generating ephemeris interpolant arrays - "
        f"{num_ephem_samples} samples will be used"
    )

    # Generate ephemeris interpolants
    ephem_t_sample, ephem_r_sample = generate_interpolant_arrays(
        earth, sun, cfg.epoch_jd, cfg.t_span, num_ephem_samples
    )

    # convert ephem_r_sample to m
    ephem_r_sample = ephem_r_sample * 1e3

    interp_coeffs = backward_hermite_coefficients(ephem_t_sample, ephem_r_sample.T)

    ephem_interpolant = CubicInterpolation(ephem_t_sample, interp_coeffs)

    return Params(
        y_target=cfg.y_target,
        conv_tol=cfg.conv_tol,
        w_oe=cfg.w_oe,
        w_penalty=cfg.w_penalty,
        kappa=jnp.deg2rad(cfg.kappa),
        characteristic_accel=cfg.characteristic_accel,
        epoch_jd=cfg.epoch_jd,
        sun_ephem=ephem_interpolant,
    )


def prebake_ode(
    ode: Callable[[float, Array, Any, Any], Array], cfg: SimConfig
) -> Callable[[float, Array, Any], Array]:
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

    baked_ode = jit(
        partial(
            ode,
            steering_law=steering_law,
            propulsion_model=propulsion_model,
            perturbations=[],  # TODO: eventually add perturbations
        )
    )

    return baked_ode


def termination_condition(t: float, y: Array, args: Params, **kwargs) -> bool:
    """
    Termination condition for the ODE solver.
    """
    # Check if the guidance loss is below the convergence tolerance
    loss = norm_loss(y.at[0].mul(P_SCALING), args.y_target, args.w_oe)

    return loss < args.conv_tol
