import jax.numpy as jnp
from jax.typing import ArrayLike

from qlipper.constants import P_SCALING


def norm_loss(y: ArrayLike, y_target: ArrayLike, weights: ArrayLike) -> float:
    """
    Vector norm guidance loss.

    Parameters
    ----------
    y : ArrayLike
        State vector in modified equinoctial elements.
    y_target : ArrayLike
        Target state vector.
    weights : ArrayLike
        Weights for each state variable.

    Returns
    -------
    loss : float
        guidance loss.

    Notes
    -----
    Normalizes the semilatus rectum by P_SCALING.
    """

    return jnp.linalg.norm((y[:5] - y_target).at[0].divide(P_SCALING) * weights)
