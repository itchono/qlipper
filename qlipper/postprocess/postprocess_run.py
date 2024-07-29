from datetime import datetime
from json import dump
from pathlib import Path

from jax.typing import ArrayLike

from qlipper.configuration import SimConfig
from qlipper.constants import OUTPUT_DIR
from qlipper.postprocess.elem_plotters import plot_elements_mee
from qlipper.postprocess.traj_plotters import plot_trajectory_mee


def postprocess_run(t: ArrayLike, y: ArrayLike, cfg: SimConfig) -> None:
    """
    Postprocess the results of a mission run.

    Parameters
    ----------
    t : ArrayLike
        Time vector.
    y : ArrayLike
        State vector.
    cfg : SimConfig
        Simulation configuration.
    """

    # Pre-run
    run_start = datetime.now()

    # create an identifier for the mission
    run_id = f"postprocess_{run_start:%Y%m%d_%Hh%Mm%Ss}"

    save_dir = Path(OUTPUT_DIR) / cfg.name / run_id
    save_dir.mkdir(parents=True)

    # save the configuration
    with open(save_dir / "cfg.json", "w") as f:
        dump(cfg.serialize(), f, indent=4)

    plot_trajectory_mee(y, save_path=save_dir / "trajectory.pdf", show=False)
    plot_elements_mee(t, y, cfg, save_path=save_dir / "elements.pdf", show=False)


# TODO: postprocess from a folder, loading results from a run
