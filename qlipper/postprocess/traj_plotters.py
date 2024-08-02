from pathlib import Path
from typing import Any

from jax import vmap
from jax.typing import ArrayLike
from matplotlib import pyplot as plt

from qlipper.converters import mee_to_cartesian
from qlipper.postprocess.interpolation import interpolate_mee


def plot_trajectory_mee(
    t: ArrayLike,
    y: ArrayLike,
    save_path: Path | None = None,
    save_kwargs: dict[str, Any] = {},
    show: bool = False,
) -> None:
    """
    Plots 3D trajectory from modified equinoctial elements.
    """

    # smooth the trajectory
    t_interp, y_interp = interpolate_mee(t, y, seg_per_orbit=100)

    cart = vmap(mee_to_cartesian)(y_interp)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(cart[:, 0], cart[:, 1], cart[:, 2], label="Trajectory")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Earth Inertial Coordinates")

    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)

    if show:
        plt.show()
