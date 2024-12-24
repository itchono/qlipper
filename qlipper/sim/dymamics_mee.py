import jax
import jax.numpy as jnp
from jax.numpy import cos, sin, sqrt
from jax.typing import ArrayLike

from qlipper.constants import MU_EARTH


def gve_coefficients(state: ArrayLike, mu: float) -> tuple[jax.Array, jax.Array]:
    """
    Gauss variational equation coefficients for
    a-modified equinoctial elements under no additional perturbations.

    i.e. [a f g h k L]

    Parameters
    ----------
    state : ArrayLike
        State vector in modified equinoctial elements.
    mu : float
        Gravitational parameter.

    Returns
    -------
    A : jax.Array
        A-matrix for Gauss variational equation.
    b : jax.Array
        b-vector for Gauss variational equation.

    """
    # unpack state vector
    a, f, g, h, k, L = state

    # convert SMA and ecc to p
    p = a * (1 - f**2 - g**2)  # semi-latus rectum p = a(1 - e^2)

    # shorthand quantities
    q = 1 + f * cos(L) + g * sin(L)

    leading_coefficient = 1 / q * sqrt(p / mu)

    # A-matrix
    A = (
        jnp.array(
            [
                [
                    2 * a * q * (f * sin(L) - g * cos(L)) / (1 - f**2 - g**2),
                    2 * a * q**2 / (1 - f**2 - g**2),
                    0,
                ],
                [
                    q * sin(L),
                    (q + 1) * cos(L) + f,
                    -g * (h * sin(L) - k * cos(L)),
                ],
                [
                    -q * cos(L),
                    (q + 1) * sin(L) + g,
                    f * (h * sin(L) - k * cos(L)),
                ],
                [0, 0, cos(L) / 2 * (1 + h**2 + k**2)],
                [0, 0, sin(L) / 2 * (1 + h**2 + k**2)],
                [0, 0, h * sin(L) - k * cos(L)],
            ]
        )
        * leading_coefficient
    )

    # b-vector
    b = jnp.array([0, 0, 0, 0, 0, q**2 * sqrt(mu * p) / p**2])

    return A, b


def gve_mee(state: ArrayLike, acc_lvlh: ArrayLike) -> jax.Array:
    """
    Gauss variational equation for modified equinoctial elements.

    Parameters
    ----------
    state : ArrayLike
        State vector in modified equinoctial elements.
    acc_lvlh : ArrayLike
        Perturbing acceleration vector in LVLH frame.

    Returns
    -------
    dstate_dt : jax.Array
        Time derivative of state vector.

    """
    # Gauss variational equation coefficients
    A, b = gve_coefficients(state, MU_EARTH)

    # time derivative of state vector
    dstate_dt = A @ acc_lvlh + b

    return dstate_dt
