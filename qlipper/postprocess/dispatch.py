import logging
from pathlib import Path

import numpy as np
from jax.typing import ArrayLike
from matplotlib import pyplot as plt

from qlipper.configuration import SimConfig
from qlipper.constants import MU_EARTH, OUTPUT_DIR
from qlipper.converters import batch_mee_to_cartesian
from qlipper.postprocess.plot_cart import plot_cart_wrt_moon, plot_trajectory_cart
from qlipper.postprocess.plot_mee import plot_elements_mee, plot_trajectory_mee

logger = logging.getLogger(__name__)


def postprocess_run(
    run_id: str, t: ArrayLike, y: ArrayLike, cfg: SimConfig, show_plots: bool = False
) -> None:
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
    show_plots : bool, optional
        Whether to show the plots, default False.
    """
    logger.info(f"Postprocessing run: {run_id}")
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

    # Plotting
    plot_save_dir = Path(OUTPUT_DIR) / cfg.name / run_id / "plots"
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    # convert mee to cartesian
    y_cart = batch_mee_to_cartesian(y, MU_EARTH)

    # HACK: try to plot MEE trajectory, if it fails, plot Cartesian
    try:
        plot_trajectory_mee(
            t, y, cfg, save_path=plot_save_dir / "trajectory.pdf", show=show_plots
        )
    except Exception:
        plot_trajectory_cart(
            t,
            y_cart,
            cfg,
            save_path=plot_save_dir / "trajectory.pdf",
            show=show_plots,
        )
    plot_elements_mee(
        t, y, cfg, save_path=plot_save_dir / "elements.pdf", show=show_plots
    )

    if cfg.steering_law in ["bbq_law", "qbbq_law"]:
        plot_cart_wrt_moon(
            t,
            y_cart,
            cfg,
            save_path=plot_save_dir / "trajectory_moon.pdf",
            show=show_plots,
        )
        CLOSE_FRAC = 0.02

        plot_cart_wrt_moon(
            t[-int(len(t) * CLOSE_FRAC) :],
            y_cart[-int(len(t) * CLOSE_FRAC) :, :],
            cfg,
            save_path=plot_save_dir / "trajectory_moon_close.pdf",
            show=show_plots,
        )

        plot_elements_mee(
            t[-int(len(t) * CLOSE_FRAC) :],
            y[-int(len(t) * CLOSE_FRAC) :, :],
            cfg,
            save_path=plot_save_dir / "elements_close.pdf",
            show=show_plots,
        )

    logger.info(f"Postprocessing complete for run: {run_id}")


def postprocess_from_folder(folder: Path, show_plots: bool = False) -> None:
    """
    Postprocess a qlipper run, writing outputs to the same folder.

    Parameters
    ----------
    folder : Path
        Folder containing the run.
    show_plots : bool, optional
        Whether to show the plots, default False.
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

    postprocess_run(run_id, t, y, cfg, show_plots)
