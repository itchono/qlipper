import jax
import jax.numpy as jnp
import pytest

from qlipper.utils.converters import cartesian_to_mee, mee_to_cartesian


def test_cartesian_to_mee():
    cart = jnp.array([1e6, 2e6, 3e6, 4e3, 5e3, 6e3])
    mee = cartesian_to_mee(cart)

    mee_ref = jnp.array([1.3547e5, -0.8809, 0.4217, -1.3798, -0.6899, -0.6087])

    assert mee == pytest.approx(mee_ref, rel=1e-4)


def test_mee_to_cartesian():
    mee = jnp.array([1.3547e5, -0.8809, 0.4217, -1.3798, -0.6899, -0.6087])
    cart = mee_to_cartesian(mee)

    cart_ref = jnp.array([1e6, 2e6, 3e6, 4e3, 5e3, 6e3])

    assert cart == pytest.approx(cart_ref, rel=1e-3)


def test_roundtrip_mee_cartesian():
    cart = jnp.array([1e6, 2e6, 3e6, 4e3, 5e3, 6e3])

    mee = cartesian_to_mee(cart)
    cart_back = mee_to_cartesian(mee)

    assert cart_back == pytest.approx(cart, rel=1e-8)


def test_converters_batched():
    c2m_vmap = jax.vmap(cartesian_to_mee, (0,))
    m2c_vmap = jax.vmap(mee_to_cartesian, (0,))

    cart = jnp.array([[1e6, 2e6, 3e6, 4e3, 5e3, 6e3], [1e6, 2e6, 3e6, 4e3, 5e3, 6e3]])

    mee = c2m_vmap(cart)

    cart_back = m2c_vmap(mee)

    assert cart_back == pytest.approx(cart, rel=1e-8)
