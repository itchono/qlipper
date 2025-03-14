import logging
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
from diffrax import (
    RESULTS,
    ODETerm,
    PIDController,
    SaveAt,
    Solution,
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
from qlipper.converters import batch_cartesian_to_mee, mee_to_cartesian
from qlipper.run.prebake import (
    prebake_convergence_criterion,
    prebake_ode,
    prebake_sim_config,
)
from qlipper.run.recording import temp_log_to_file
from qlipper.sim.dynamics_cartesian import CARTESIAN_DYN_SCALING
from qlipper.sim.loss import l2_loss

logger = logging.getLogger(__name__)


def preprocess_y0(cfg: SimConfig) -> Array:
    """
    Preprocess the initial state vector y0.

    Parameters
    ----------
    cfg : SimConfig
        Simulation configuration.

    Returns
    -------
    y0 : Array
        Preprocessed initial state vector.
    """
    match cfg.dynamics:
        case "cartesian":
            return mee_to_cartesian(cfg.y0) / CARTESIAN_DYN_SCALING
        case "mee":
            return cfg.y0.at[0].divide(P_SCALING)
        case _:
            raise ValueError(f"Unsupported dynamics: {cfg.dynamics}")


def extract_solution_arrays(sol: Solution, cfg: SimConfig) -> tuple[Array, Array]:
    """
    Extract the state and time arrays from a solution.

    Parameters
    ----------
    sol : Solution
        Solution object from diffeqsolve.
    cfg : SimConfig
        Simulation configuration.

    Returns
    -------
    sol_t : Array
        Time elapsed in seconds.
    sol_y : Array
        State in modified equinoctial elements.
    """
    valid_idx = jnp.isfinite(sol.ys[:, 0])
    sol_t = sol.ts[valid_idx]

    match cfg.dynamics:
        case "cartesian":
            sol_y = batch_cartesian_to_mee(sol.ys[valid_idx, :] * CARTESIAN_DYN_SCALING)
        case "mee":
            sol_y = sol.ys.at[:, 0].mul(P_SCALING)[valid_idx]

    return sol_t, sol_y


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
        term = ODETerm(prebake_ode(cfg))
        ode_args = prebake_sim_config(cfg)
        y0 = preprocess_y0(cfg)
        end_event = prebake_convergence_criterion(cfg)

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
                rtol=1e-6, atol=1e-8, pcoeff=0.3, icoeff=0.3, dcoeff=0
            ),
            event=end_event,
            saveat=SaveAt(steps=True),
        )

        # postprocessing
        sol_t, sol_y = extract_solution_arrays(solution, cfg)

        # Post-run
        match solution.result:
            case RESULTS.event_occurred:
                logger.info(CONVERGED_BLOCK_LETTERS)
            case RESULTS.successful:
                logger.info("NOT CONVERGED - reached end of integration interval")
            case _:
                logger.warning(f"NOT CONVERGED - {solution.result}")
        logger.info(f"Final Error: {l2_loss(sol_y[-1], cfg.y_target, cfg.w_oe):.5f}")
        run_end = datetime.now()
        run_duration = run_end - run_start
        logger.info(f"Completed in {run_duration}")
        logger.info(f"Steps: {solution.stats['num_steps']}")

        # Post-run saving of results
        with open(mission_dir / "vars.npz", "wb") as f:
            jnp.savez(f, y=sol_y, t=sol_t)

        logger.info(f"Saved to {mission_dir.resolve()}")

    return run_id, sol_t, sol_y
