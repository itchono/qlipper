import logging
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
from diffrax import (
    RESULTS,
    Event,
    ODETerm,
    PIDController,
    SaveAt,
    Tsit5,
    diffeqsolve,
)
from jax import Array

from qlipper.configuration import SimConfig
from qlipper.constants import (
    CONVERGED_BLOCK_LETTERS,
    OUTPUT_DIR,
    P_SCALING,
    QLIPPER_BLOCK_LETTERS,
)
from qlipper.run.prebake import (
    norm_loss,
    prebake_ode,
    prebake_sim_config,
    termination_condition,
)
from qlipper.run.recording import temp_log_to_file
from qlipper.sim.dymamics_mee import dyn_mee

logger = logging.getLogger(__name__)


def run_mission(cfg: SimConfig) -> tuple[str, Array, Array]:
    """
    Run a qlipper simulation.

    Automatically saves the configuration and results to disk.

    Parameters
    ----------
    cfg : SimConfig
        Simulation configuration.

    Returns
    -------
    run_id : str
        Unique identifier for the mission
    t : Array, shape (N,)
        Time vector in seconds elapsed.
    y : Array, shape (N, 6)
        State vector at the end of the simulation.
    """
    # Pre-run
    run_start = datetime.now()

    # create an identifier for the mission
    run_id = f"run_{run_start:%Y%m%d_%Hh%Mm%Ss}"

    # create a directory for the mission
    mission_dir = Path(OUTPUT_DIR) / cfg.name / run_id
    mission_dir.mkdir(parents=True)

    # save the configuration
    with open(mission_dir / "cfg.json", "w") as f:
        f.write(cfg.serialize())

    with temp_log_to_file(mission_dir / "run.log"):
        logger.info(f"Starting mission {cfg.name} with ID {run_id}")

        # prebake
        term = ODETerm(prebake_ode(dyn_mee, cfg))
        ode_args = prebake_sim_config(cfg)
        termination_event = Event([termination_condition])  # TODO: get working

        # preprocessing
        y0 = cfg.y0.at[0].divide(P_SCALING)  # rescale initial state

        # Run
        logger.info(QLIPPER_BLOCK_LETTERS)
        solution = diffeqsolve(
            term,
            Tsit5(),
            *cfg.t_span,
            y0=y0,
            dt0=None,
            args=ode_args,
            max_steps=int(1e6),
            stepsize_controller=PIDController(
                rtol=1e-6, atol=1e-7, pcoeff=0.3, icoeff=0.3, dcoeff=0
            ),
            event=termination_event,
            saveat=SaveAt(steps=True),
        )

        # postprocessing
        valid_idx = jnp.isfinite(solution.ys[:, 0])
        sol_y = solution.ys.at[:, 0].mul(P_SCALING)[valid_idx]
        sol_t = solution.ts[valid_idx]

        # Post-run
        run_end = datetime.now()
        run_duration = run_end - run_start
        logger.info(f"Mission {cfg.name} with ID {run_id} completed in {run_duration}")
        logger.info(f"Final Error: {norm_loss(sol_y[-1], cfg.y_target, cfg.w_oe)}")
        logger.info(
            "Stats:\n"
            + "\n".join([f"{s_k}: {s_v}" for s_k, s_v in solution.stats.items()])
        )

        match solution.result:
            case RESULTS.event_occurred:
                logger.info(CONVERGED_BLOCK_LETTERS)
            case RESULTS.successful:
                logger.info("NOT CONVERGED - reached end of integration interval")
            case _:
                logger.warning(f"NOT CONVERGED - {solution.result}")

        # Post-run saving of results
        with open(mission_dir / "vars.npz", "wb") as f:
            jnp.savez(f, y=sol_y, t=sol_t)

        logger.info(f"Run variables saved to {mission_dir.resolve()}")

    return run_id, sol_t, sol_y
