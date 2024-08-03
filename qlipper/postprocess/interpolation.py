import jax.numpy as jnp
from diffrax import CubicInterpolation, backward_hermite_coefficients
from jax import Array, vmap


def interpolate_mee(t: Array, y: Array, seg_per_orbit: int) -> tuple[Array, Array]:
    """
    Interpolate the state vector to a higher resolution by using
    a fixed number of segments per orbit.

    Parameters
    ----------
    t : Array, shape (N,)
        Time vector in seconds elapsed.
    y : Array, shape (N, 6)
        State vector at the end of the simulation.
    seg_per_orbit : int
        Number of segments per orbit.

    Returns
    -------
    t_interp : Array, shape (N_interp,)
        Interpolated time vector in seconds elapsed.
    y_interp : Array, shape (N_interp, 6)
        Interpolated state vector.
    """

    l_step = jnp.pi * 2 / seg_per_orbit
    n_interp = jnp.astype(y[-1, -1] / l_step, int)
    l_eval = jnp.linspace(y[0, -1], y[-1, -1], n_interp)

    l_v = y[:, -1]
    s_v = jnp.concatenate((y[:, :-1], t[:, None]), axis=1)

    # interpolate the state vector
    coeffs = backward_hermite_coefficients(l_v, s_v)
    interp = CubicInterpolation(l_v, coeffs)

    interpolated = vmap(interp.evaluate)(l_eval)
    y_interp = jnp.concatenate((interpolated[:, :-1], l_eval[:, None]), axis=1)
    t_interp = interpolated[:, -1]

    return t_interp, y_interp
