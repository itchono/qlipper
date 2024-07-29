from pathlib import Path
from typing import Any

from jax import vmap
from jax.typing import ArrayLike
from matplotlib import pyplot as plt

from qlipper.converters import mee_to_cartesian


def plot_trajectory_mee(
    y: ArrayLike,
    save_path: Path | None = None,
    save_kwargs: dict[str, Any] = {},
    show: bool = False,
) -> None:
    cart = vmap(mee_to_cartesian)(y)

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
