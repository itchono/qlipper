from pathlib import Path
from typing import Any

from jax.typing import ArrayLike
from matplotlib import pyplot as plt

from qlipper.configuration import SimConfig


def plot_elements_mee(
    t: ArrayLike,
    y: ArrayLike,
    cfg: SimConfig,
    save_path: Path | None = None,
    save_kwargs: dict[str, Any] = {},
    show: bool = False,
) -> None:
    """
    Plot the modified equinoctial elements over time.
    """
    t_jd = cfg.epoch_jd + t / 86400

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    axs: list[plt.Axes]

    axs[0].plot(t_jd, y[:, 0], label="p")
    axs[0].set_ylabel("Orbit Size (m)")
    axs[0].legend()

    axs[1].plot(t_jd, y[:, 1], label="f")
    axs[1].plot(t_jd, y[:, 2], label="g")
    axs[1].set_ylabel("Eccentricity Vector")

    axs[2].plot(t_jd, y[:, 3], label="h")
    axs[2].plot(t_jd, y[:, 4], label="k")
    axs[2].set_ylabel("Nodal Position")

    axs[2].set_xlabel("Time (jd)")

    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)

    if show:
        plt.show()
