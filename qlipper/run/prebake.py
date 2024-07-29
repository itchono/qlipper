import logging
from functools import partial
from typing import Any, Callable

from jax import Array, jit

from qlipper.configuration import SimConfig
from qlipper.sim import Params
from qlipper.sim.ephemeris import generate_interpolant_arrays, lookup_body_id

logger = logging.getLogger(__name__)


def prebake_sim_config(cfg: SimConfig) -> Params:
    """
    Prebake the SimConfig struct into a SimInternalConfig struct
    that can be passed into the actual problem being solved.
    """

    # Generate ephemeris interpolant arrays
    logger.info("Generating ephemeris interpolant arrays...")

    sun = lookup_body_id("sun")
    earth = lookup_body_id("earth")

    # TODO: either add a heuristic or make this a parameter
    NUM_SAMPLES = 100

    ephem_t_sample, ephem_y_sample = generate_interpolant_arrays(
        earth, sun, cfg.epoch_jd, cfg.t_span, NUM_SAMPLES
    )

    return Params(
        y_target=cfg.y_target,
        conv_tol=cfg.conv_tol,
        w_oe=cfg.w_oe,
        w_penalty=cfg.w_penalty,
        kappa=cfg.kappa,
        characteristic_accel=cfg.characteristic_accel,
        epoch_jd=cfg.epoch_jd,
        ephem_t_sample=ephem_t_sample,
        ephem_y_sample=ephem_y_sample,
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

    baked_ode = jit(partial(ode, cfg=cfg))

    return baked_ode
