from typing import Callable

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from qlipper.constants import MU, P_SCALING
from qlipper.converters import mee_to_cartesian
from qlipper.run.prebake import Params


def gve_coefficients(state: ArrayLike) -> tuple[Array, Array]:
    """
    Gauss variational equation coefficients for
    modified equinoctial elements.

    Parameters
    ----------
    state : ArrayLike
        State vector in modified equinoctial elements.

    Returns
    -------
    A : Array
        A-matrix for Gauss variational equation.
    b : Array
        b-vector for Gauss variational equation.
    """

    # unpack state vector
    p, f, g, h, k, L = state

    # shorthand quantities
    q = 1 + f * jnp.cos(L) + g * jnp.sin(L)

    leading_coefficient = 1 / q * jnp.sqrt(p / MU)

    # A-matrix
    A = (
        jnp.array(
            [
                [0, 2 * p, 0],
                [
                    q * jnp.sin(L),
                    (q + 1) * jnp.cos(L) + f,
                    -g * (h * jnp.sin(L) - k * jnp.cos(L)),
                ],
                [
                    -q * jnp.cos(L),
                    (q + 1) * jnp.sin(L) + g,
                    f * (h * jnp.sin(L) - k * jnp.cos(L)),
                ],
                [0, 0, jnp.cos(L) / 2 * (1 + h**2 + k**2)],
                [0, 0, jnp.sin(L) / 2 * (1 + h**2 + k**2)],
                [0, 0, h * jnp.sin(L) - k * jnp.cos(L)],
            ]
        )
        * leading_coefficient
    )

    # b-vector
    b = jnp.array([0, 0, 0, 0, 0, q**2 * jnp.sqrt(MU * p) / p**2])

    return A, b


def gve_mee(state: ArrayLike, acc_lvlh: ArrayLike) -> Array:
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
    dstate_dt : Array
        Time derivative of state vector.
    """

    # Gauss variational equation coefficients
    A, b = gve_coefficients(state)

    # time derivative of state vector
    dstate_dt = A @ acc_lvlh + b

    return dstate_dt


def dyn_mee(
    t: float,
    y: ArrayLike,
    params: Params,
    steering_law: Callable[[float, Array, Params], tuple[float, float]],
    propulsion_model: Callable[[float, Array, Params, float, float], Array],
    perturbations: list[Callable[[float, Array], Array]],
) -> Array:
    """
    Top level dynamics function for modified equinoctial elements.

    Parameters
    ----------
    t : float
        Time since epoch (s).
    y : ArrayLike
        State vector in modified equinoctial elements.
    params : Params
        Configuration object.
    steering_law : Callable
        Control law, should be pre-baked with the ODE.
    propulsion_model : Callable
        Propulsion model, should be pre-baked with the ODE.
    perturbations : list[Callable]
        List of perturbation functions, should be pre-baked with the ODE.

    Returns
    -------
    dy_dt : Array
        Time derivative of state vector.

    Outline
    -------
    1. Rescale state vector (p specifically) from normalized to physical units.
    2. Compute control.
    3. Compute acceleration.
    4. Compute perturbations (if any).
    5. Apply Gauss variational equation.
    6. Rescale time derivative of state vector.
    """

    # Scaling
    y = y.at[0].mul(P_SCALING)

    # Convert state to cartesian for thrust model
    cart = mee_to_cartesian(y)

    # Control
    alpha, beta = steering_law(t, y, params)

    # Acceleration from propulsion
    acc_lvlh = propulsion_model(t, cart, params, alpha, beta)

    # Perturbations
    for perturbation in perturbations:
        acc_lvlh += perturbation(t, y)

    # Gauss variational equation
    dy_dt = gve_mee(y, acc_lvlh)

    return dy_dt.at[0].divide(P_SCALING)
