import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from nrs.constants import MU, P_SCALING
from nrs.sim.dymamics_mee import gve_coefficients


def approx_max_roc(y: ArrayLike) -> Array:
    p, f, g, h, k, L = y

    q = 1 + f * jnp.cos(L) + g * jnp.sin(L)

    d_p_max = 2 * p / q * jnp.sqrt(p / MU)
    d_f_max = 2 * jnp.sqrt(p / MU)
    d_g_max = 2 * jnp.sqrt(p / MU)

    # singularity detection for d_h_max and d_k_max
    if jnp.abs(jnp.sqrt(1 - g**2) + f) < 1e-6:
        d1 = 1e-6 * jnp.sign(jnp.sqrt(1 - g**2) + f)
    else:
        d1 = jnp.sqrt(1 - g**2) + f

    if jnp.abs(jnp.sqrt(1 - f**2) + g) < 1e-6:
        d2 = 1e-6 * jnp.sign(jnp.sqrt(1 - f**2) + g)
    else:
        d2 = jnp.sqrt(1 - f**2) + g

    d_h_max = 1 / 2 * jnp.sqrt(p / MU) * (1 + h**2 + k**2) / d1
    d_k_max = 1 / 2 * jnp.sqrt(p / MU) * (1 + h**2 + k**2) / d2

    # TODO missing characteristic thrust factor
    return jnp.array([d_p_max, d_f_max, d_g_max, d_h_max, d_k_max])


def q_law(y: ArrayLike, target: ArrayLike, w_oe: ArrayLike) -> tuple[float, float]:
    """
    Q-Law.

    Parameters
    ----------
    y : ArrayLike
        Current state vector.
    target : ArrayLike
        Target state vector.
    w_oe : ArrayLike
        Weighting factors for the OE states.

    Returns
    -------
    alpha : float
        Steering angle.
    beta : float
        Steering angle.
    """
    S = jnp.array([1 / P_SCALING, 1, 1, 1, 1])
    d_oe_max = approx_max_roc(y)

    oe = y[:5]
    oe_hat = target

    A, _ = gve_coefficients(oe)

    Xi_E = 2 * (oe - oe_hat) / d_oe_max

    A: Array = A.at[:5, :]
    D = A.T @ (w_oe * S * Xi_E)

    alpha = jnp.atan2(-D[0], -D[1])
    beta = jnp.atan2(-D[2], jnp.linalg.norm(D[:2]))

    return alpha, beta
