from jax.typing import ArrayLike


def trivial_steering(
    y: ArrayLike, target: ArrayLike, w_oe: ArrayLike
) -> tuple[float, float]:
    """
    Points the spacecraft prograde.
    """
    return 0, 0
