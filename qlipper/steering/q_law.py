import jax
import jax.numpy as jnp
from jax import Array
from jax.lax import cond
from jax.typing import ArrayLike

from qlipper.constants import MU_EARTH, P_SCALING
from qlipper.converters import cartesian_to_mee
from qlipper.sim import Params
from qlipper.sim.dymamics_mee import gve_coefficients


@jax.jit
def periapsis_penalty(y_mee: ArrayLike, rp_min: float, pen_param: float) -> float:
    p = y_mee[0]
    f = y_mee[1]
    g = y_mee[2]

    ecc = jnp.sqrt(f**2 + g**2)
    a = p / (1 - ecc**2)
    rp = a * (1 - ecc)
    # TODO: considerations for hyperbolic orbits (this is wrong rn)

    penalty = jnp.exp((1 - rp / rp_min) * pen_param)

    return penalty


def approx_max_roc(y_mee: ArrayLike, characteristic_accel: float, mu: float) -> Array:
    """
    Approximate the maximum rate of change for each element.

    Assumes a conventional low-thrust spacecraft with a fixed acceleration.

    Formulated by Varga and Perez (2016).

    Parameters
    ----------
    y_mee : ArrayLike
        Current state vector in modified equinoctial elements.
    characteristic_accel : float
        Characteristic acceleration of the spacecraft
    mu : float
        Gravitational parameter of the central body.

    Returns
    -------
    d_oe_max : Array
        Maximum rate of change for each element.
    """
    p, f, g, h, k, L = y_mee

    q = 1 + f * jnp.cos(L) + g * jnp.sin(L)

    d_p_max = 2 * p / q * jnp.sqrt(p / mu)
    d_f_max = 2 * jnp.sqrt(p / mu)
    d_g_max = 2 * jnp.sqrt(p / mu)

    # singularity detection for d_h_max and d_k_max
    # small modification to always take abs of sqrt to avoid problems with hyperbolic orbits
    singularity_h = jnp.abs(jnp.sqrt(jnp.abs(1 - g**2)) + f) < 1e-6
    d1 = cond(
        singularity_h,
        lambda: 1e-6 * jnp.sign(jnp.sqrt(jnp.abs(1 - g**2)) + f),
        lambda: jnp.sqrt(jnp.abs(1 - g**2)) + f,
    )

    singularity_k = jnp.abs(jnp.sqrt(jnp.abs(1 - f**2)) + g) < 1e-6
    d2 = cond(
        singularity_k,
        lambda: 1e-6 * jnp.sign(jnp.sqrt(jnp.abs(1 - f**2)) + g),
        lambda: jnp.sqrt(jnp.abs(1 - f**2)) + g,
    )

    d_h_max = 1 / 2 * jnp.sqrt(p / mu) * (1 + h**2 + k**2) / d1
    d_k_max = 1 / 2 * jnp.sqrt(p / mu) * (1 + h**2 + k**2) / d2

    return (
        jnp.array([d_p_max, d_f_max, d_g_max, d_h_max, d_k_max]) / characteristic_accel
    )


def approx_max_roc_improved(
    y_mee: ArrayLike, characteristic_accel: float, mu: float
) -> Array:
    """
    Approximate the maximum rate of change for each element.

    Assumes a conventional low-thrust spacecraft with a fixed acceleration.

    Formulated by Varga and Perez (2016) and improved by Yuan et al. (2007).

    Parameters
    ----------
    y_mee : ArrayLike
        Current state vector in modified equinoctial elements.
    characteristic_accel : float
        Characteristic acceleration of the spacecraft
    mu : float
        Gravitational parameter of the central body.

    Returns
    -------
    d_oe_max : Array
        Maximum rate of change for each element.
    """
    p, f, g, h, k, L = y_mee

    q = 1 + f * jnp.cos(L) + g * jnp.sin(L)

    d_p_max = 2 * p / q * jnp.sqrt(p / mu)

    # f and g max rates are L dependent, use a 50-elem grid to find max)*
    def d_f_max_fcn(L):
        return (
            1
            / q
            * jnp.sqrt(p / mu)
            * jnp.sqrt(
                (f * jnp.sin(L) * (q + 1)) ** 2
                + q**2 * jnp.sin(L) ** 2
                + g**2 * (k * jnp.cos(L) - h * jnp.sin(L)) ** 2
            )
        )

    def d_g_max_fcn(L):
        return (
            1
            / q
            * jnp.sqrt(p / mu)
            * jnp.sqrt(
                (g * jnp.sin(L) * (q + 1)) ** 2
                + q**2 * jnp.cos(L) ** 2
                + f**2 * (k * jnp.cos(L) - h * jnp.sin(L)) ** 2
            )
        )

    d_f_max = jnp.max(d_f_max_fcn(jnp.linspace(0, 2 * jnp.pi, 50)))
    d_g_max = jnp.max(d_g_max_fcn(jnp.linspace(0, 2 * jnp.pi, 50)))

    # singularity detection for d_h_max and d_k_max
    # small modification to always take abs of sqrt to avoid problems with hyperbolic orbits
    singularity_h = jnp.abs(jnp.sqrt(jnp.abs(1 - g**2)) + f) < 1e-6
    d1 = cond(
        singularity_h,
        lambda: 1e-6 * jnp.sign(jnp.sqrt(jnp.abs(1 - g**2)) + f),
        lambda: jnp.sqrt(jnp.abs(1 - g**2)) + f,
    )

    singularity_k = jnp.abs(jnp.sqrt(jnp.abs(1 - f**2)) + g) < 1e-6
    d2 = cond(
        singularity_k,
        lambda: 1e-6 * jnp.sign(jnp.sqrt(jnp.abs(1 - f**2)) + g),
        lambda: jnp.sqrt(jnp.abs(1 - f**2)) + g,
    )

    d_h_max = 1 / 2 * jnp.sqrt(p / mu) * (1 + h**2 + k**2) / d1
    d_k_max = 1 / 2 * jnp.sqrt(p / mu) * (1 + h**2 + k**2) / d2

    return (
        jnp.array([d_p_max, d_f_max, d_g_max, d_h_max, d_k_max]) / characteristic_accel
    )


def _q_law_mee(
    y_mee: ArrayLike,
    target: ArrayLike,
    w_oe: ArrayLike,
    characteristic_accel: float,
    mu: float,
) -> tuple[float, float]:
    """
    Q-Law, as formulated by Varga and Perez (2016)

    Parameters
    ----------
    y_mee : ArrayLike
        Current state vector.
    target : ArrayLike
        Target state vector.
    w_oe : ArrayLike
        Q-law weights
    characteristic_accel : float
        Characteristic acceleration of the spacecraft
    mu : float
        Gravitational parameter of the central body.

    Returns
    -------
    alpha : float
        Steering angle.
    beta : float
        Steering angle.
    """

    S = jnp.array([1 / P_SCALING, 1, 1, 1, 1])
    d_oe_max = approx_max_roc(y_mee, characteristic_accel, mu)
    # EXPERIMENT with old and new approx_max_roc (everything works well with old)

    A, _ = gve_coefficients(y_mee, mu)

    oe = y_mee[:5]
    oe_hat = target[:5]

    Xi_E = 2 * (oe - oe_hat) / d_oe_max

    # EXPERIMENT with penalty (THIS IS HARDCODED RIGHT NOW!) (default: no penalty)
    P, dPdoe = jax.value_and_grad(periapsis_penalty)(y_mee, 7000e3, 2)
    w_p = 0.5
    Xi_P = dPdoe[:5] * ((oe - oe_hat) / d_oe_max) ** 2

    D = A[:5, :].T @ (w_oe * S * (w_p * Xi_P + (1 + w_p * P) * Xi_E))

    alpha = jnp.atan2(-D[0], -D[1])
    beta = jnp.atan2(-D[2], jnp.linalg.norm(D[:2]))

    return alpha, beta


def q_law(_: float, y: ArrayLike, params: Params) -> tuple[float, float]:
    """
    'API-level' wrapper for the Q-law, operating on
    Cartesian state vectors. Converts to MEE under the hood.

    Designed for use around the Earth.

    Parameters
    ----------
    t : float
        Time.
    y : ArrayLike
        Cartesian state vector.
    params : Params
        Simulation parameters.

    Returns
    -------
    alpha : float
        Steering angle.
    beta : float
        Steering angle.
    """

    target = params.y_target
    w_oe = params.w_oe

    y_mee = cartesian_to_mee(y, MU_EARTH)

    return _q_law_mee(y_mee, target, w_oe, params.characteristic_accel, MU_EARTH)


def _rq_law_mee(
    y_mee: ArrayLike,
    target: ArrayLike,
    w_oe: ArrayLike,
    characteristic_accel: float,
    mu: float,
) -> tuple[float, float]:
    """
    Rendezvous capable Q-law, as formulated by Narayanaswamy (2023).

    Augments target p state based on difference in true longitude with target.
    """
    # Augment target set
    L_curr = y_mee[5] % (2 * jnp.pi)
    L_target = target[5] % (2 * jnp.pi)
    delta_L = jnp.atan2(
        jnp.sin(L_target - L_curr), jnp.cos(L_target - L_curr)
    )  # [-pi, pi]
    augmentation = (
        2 / jnp.pi * jnp.arctan(delta_L) * target[0] * 0.1
    )  # jank, prevents the change from being too large
    target = target.at[0].add(augmentation)

    return _q_law_mee(y_mee, target, w_oe, characteristic_accel, mu)
