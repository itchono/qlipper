import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from qlipper.constants import P_SCALING


@jax.jit
def l2_loss(y: ArrayLike, y_target: ArrayLike, weights: ArrayLike) -> float:
    """
    L2 norm loss function for guidance.

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

    return jnp.linalg.norm((y[:5] - y_target).at[0].divide(P_SCALING) * weights, ord=2)


@jax.jit
def l1_loss(y: ArrayLike, y_target: ArrayLike, weights: ArrayLike) -> float:
    """
    L1 norm loss function for guidance.

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

    return jnp.linalg.norm((y[:5] - y_target).at[0].divide(P_SCALING) * weights, ord=1)
