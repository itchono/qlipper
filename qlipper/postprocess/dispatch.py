import logging
from pathlib import Path

import numpy as np
from jax.typing import ArrayLike

from qlipper.configuration import SimConfig
from qlipper.constants import OUTPUT_DIR
from qlipper.postprocess.elem_plotters import plot_elements_mee
from qlipper.postprocess.traj_plotters import plot_trajectory_mee

logger = logging.getLogger(__name__)


def postprocess_run(run_id: str, t: ArrayLike, y: ArrayLike, cfg: SimConfig) -> None:
    """
    Postprocess the results of a mission run.

    Parameters
    ----------
    run_id : str
        Unique identifier for the mission
    t : ArrayLike
        Time vector.
    y : ArrayLike
        State vector.
    cfg : SimConfig
        Simulation configuration.
    """
    logger.info(f"Postprocessing run: {run_id}")

    # Plotting
    plot_save_dir = Path(OUTPUT_DIR) / cfg.name / run_id / "plots"
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    plot_trajectory_mee(t, y, save_path=plot_save_dir / "trajectory.pdf", show=True)
    plot_elements_mee(t, y, cfg, save_path=plot_save_dir / "elements.pdf", show=False)

    logger.info(f"Postprocessing complete for run: {run_id}")


def postprocess_from_folder(folder: Path) -> None:
    """
    Postprocess a qlipper run, writing outputs to the same folder.

    Parameters
    ----------
    folder : Path
        Folder containing the run.
    """

    # run sanity checks
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a directory.")
    if not (folder / "cfg.json").is_file():
        raise FileNotFoundError(f"{folder} does not contain a cfg.json file.")
    if not (folder / "vars.npz").is_file():
        raise FileNotFoundError(f"{folder} does not contain a vars.npz file.")

    # get run_id
    run_id = folder.name

    assert run_id.startswith("run_"), f"{run_id} is not a valid run ID"

    # load config
    cfg = SimConfig.from_file(folder / "cfg.json")

    # load arrays
    loaded_obj = np.load(folder / "vars.npz")
    t = loaded_obj["t"]
    y = loaded_obj["y"]

    postprocess_run(run_id, t, y, cfg)
