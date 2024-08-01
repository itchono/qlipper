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

    axs[0].plot(t_jd, y[:, 0], label="p", color="C0")
    axs[0].axhline(cfg.y_target[0], color="C0", linestyle="--", label="p_target")
    axs[0].set_ylabel("Orbit Size (m)")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(t_jd, y[:, 1], label="f", color="C1")
    axs[1].axhline(cfg.y_target[1], color="C1", linestyle="--", label="f_target")
    axs[1].plot(t_jd, y[:, 2], label="g", color="C2")
    axs[1].axhline(cfg.y_target[2], color="C2", linestyle="--", label="g_target")
    axs[1].set_ylabel("Eccentricity Vector")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(t_jd, y[:, 3], label="h", color="C3")
    axs[2].axhline(cfg.y_target[3], color="C3", linestyle="--", label="h_target")
    axs[2].plot(t_jd, y[:, 4], label="k", color="C4")
    axs[2].axhline(cfg.y_target[4], color="C4", linestyle="--", label="k_target")
    axs[2].set_ylabel("Nodal Position")
    axs[2].legend()
    axs[2].grid()

    axs[2].set_xlabel("Time (jd)")

    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)

    if show:
        plt.show()
