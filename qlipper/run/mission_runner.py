from datetime import datetime
from json import dump
from pathlib import Path

import jax.numpy as jnp
from diffrax import ODETerm, SaveAt, diffeqsolve
from jax import Array

from qlipper.configuration import SimConfig
from qlipper.constants import OUTPUT_DIR, P_SCALING
from qlipper.sim.dymamics_mee import dyn_mee


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

    # create an identifier for the mission
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss')}"

    # create a directory for the mission
    mission_dir = Path(OUTPUT_DIR) / cfg.name / run_id
    mission_dir.mkdir(parents=True)

    # save the configuration
    cfg_path = mission_dir / "config.json"
    with open(cfg_path, "w") as f:
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

    # TODO: post-run saving of results

    return sol_y, sol_t
