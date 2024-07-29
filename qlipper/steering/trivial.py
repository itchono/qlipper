from jax.typing import ArrayLike

from qlipper.sim import Params


def trivial_steering(t: float, y: ArrayLike, params: Params) -> tuple[float, float]:
    """
    Points the spacecraft prograde.
    """
    return 0, 0
