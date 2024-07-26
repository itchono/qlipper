import logging
from datetime import datetime
from json import dump
from pathlib import Path

import jax.numpy as jnp
from diffrax import RESULTS, ODETerm, SaveAt, diffeqsolve
from jax import Array

from qlipper.configuration import SimConfig
from qlipper.constants import (
    CONVERGED_BLOCK_LETTERS,
    OUTPUT_DIR,
    P_SCALING,
    QLIPPER_BLOCK_LETTERS,
)
from qlipper.sim.dymamics_mee import dyn_mee

logger = logging.getLogger(__name__)


def run_mission(cfg: SimConfig) -> tuple[Array, Array]:
    """
    Run a qlipper simulation.

    Automatically saves the configuration and results to disk.

    Parameters
    ----------
    cfg : SimConfig
        Simulation configuration.

    Returns
    -------
    y : Array, shape (6, N)
        State vector at the end of the simulation.
    t : Array, shape (N,)
        Time vector in seconds elapsed.
    """
    # Pre-run
    run_start = datetime.now()

    # create an identifier for the mission
    run_id = f"run_{run_start:%Y%m%d_%Hh%Mm%Ss}"

    # create a directory for the mission
    mission_dir = Path(OUTPUT_DIR) / cfg.name / run_id
    mission_dir.mkdir(parents=True)

    # log to file
    log_file = mission_dir / "run.log"
    # add handler to root logger
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.getLogger().handlers[0].formatter)
    logging.getLogger().addHandler(file_handler)

    logger.info(QLIPPER_BLOCK_LETTERS)
    logger.info(f"Starting mission {cfg.name} with ID {run_id}")

    # save the configuration
    with open(mission_dir / "cfg.json", "w") as f:
        dump(cfg.serialize(), f, indent=4)

    term = ODETerm(dyn_mee)

    # preprocess scaling
    y0 = cfg.y0.at[0].divide(P_SCALING)

    # Run
    solution = diffeqsolve(
        term,
        cfg.solver,
        t0=cfg.t_span[0],
        t1=cfg.t_span[1],
        y0=y0,
        dt0=1,
        args=cfg,
        max_steps=int(1e6),
        saveat=SaveAt(steps=True),
    )

    # postprocess -- get rid of NaNs and rescale
    valid_idx = jnp.isfinite(solution.ys[:, 0])
    sol_y = solution.ys.at[:, 0].mul(P_SCALING)[valid_idx]
    sol_t = solution.ts[valid_idx]

    # Post-run
    run_end = datetime.now()
    run_duration = run_end - run_start
    logger.info(f"Mission {cfg.name} with ID {run_id} completed in {run_duration}")

    match solution.result:
        case RESULTS.event_occurred:
            logger.info(CONVERGED_BLOCK_LETTERS)
        case RESULTS.successful:
            logger.info("NOT CONVERGED - reached end of integration interval")
        case _:
            logger.warning(f"NOT CONVERGED - {solution.result}")

    # Post-run saving of results
    with open(mission_dir / "vars.npy", "wb") as f:
        jnp.save(f, sol_y)
        jnp.save(f, sol_t)

    # remove file handler
    logging.getLogger().removeHandler(file_handler)

    return sol_y, sol_t
