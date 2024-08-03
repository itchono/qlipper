from typing import Any

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray


def plot_sphere(
    ax: Axes,
    radius: float = 1.0,
    offset: NDArray[np.floating] = np.zeros((3,)),
    plot_kwargs: dict[str, Any] = {},
) -> None:
    """
    Plots a sphere with radius and center at the origin.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis object.

    radius : float, optional
        Sphere radius, by default 1.0.

    offset : NDArray[np.floating], optional
        Sphere offset, by default zero
    """
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    x_ = radius * x + offset[0]
    y_ = radius * y + offset[1]
    z_ = radius * z + offset[2]

    ax.plot_surface(x_, y_, z_, **plot_kwargs)
