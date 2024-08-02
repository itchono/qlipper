from pathlib import Path
from typing import Any

from jax.typing import ArrayLike
from matplotlib import pyplot as plt

from qlipper.constants import R_EARTH
from qlipper.converters import batch_mee_to_cartesian
from qlipper.postprocess.interpolation import interpolate_mee
from qlipper.postprocess.plotting_utils import plot_sphere


def plot_trajectory_mee(
    t: ArrayLike,
    y: ArrayLike,
    plot_kwargs: dict[str, Any] = {},
    save_path: Path | None = None,
    save_kwargs: dict[str, Any] = {},
    show: bool = False,
) -> None:
    """
    Plots full 3D trajectory from modified equinoctial elements.

    Single colour output.

    Parameters
    ----------
    t : ArrayLike
        Time array.
    y : ArrayLike
        Modified equinoctial elements array.
    save_path : Path | None, optional
        Path to save the plot, by default None.
    save_kwargs : dict[str, Any], optional
        Keyword arguments for saving the plot, by default {}.
    show : bool, optional
        Whether to show the plot, by default False.
    """

    # smooth the trajectory
    t_interp, y_interp = interpolate_mee(t, y, seg_per_orbit=100)

    cart = batch_mee_to_cartesian(y_interp)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    plot_sphere(ax, radius=R_EARTH, plot_kwargs={"color": "C0", "alpha": 0.5})

    default_plot_kwargs = {"label": "Trajectory", "color": "C2", "linewidth": 1}
    actual_plot_kwargs = default_plot_kwargs | plot_kwargs

    ax.plot(cart[:, 0], cart[:, 1], cart[:, 2], **actual_plot_kwargs)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Earth Inertial Coordinates")
    # equal aspect ratio
    ax.set_aspect("equal")

    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)

    if show:
        plt.show()
