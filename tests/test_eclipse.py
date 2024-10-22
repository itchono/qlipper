from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from qlipper.sim.eclipse import simple_eclipse
from qlipper.sim.params import Params


@pytest.fixture
def slyga_sun_ephemeris():
    def evaluate(t):
        T_sun = 31557600
        angle = 2 * jnp.pi * t / T_sun

        epsilon = jnp.deg2rad(23.439)
        dir = jnp.array(
            [
                jnp.cos(angle),
                jnp.sin(angle) * jnp.cos(epsilon),
                jnp.sin(angle) * jnp.sin(epsilon),
                0,
                0,
                0,
            ]
        )
        pos = dir * 149597870691
        return pos

    return SimpleNamespace(evaluate=evaluate)


def test_eclipse_curtis():
    state = jnp.array([2_817.899, -14_110.473, -7_502.672, 0, 0, 0]) * 1e3

    sun_position = jnp.array([-11_747_041, 139_486_985, 60_472_278]) * 1e3

    fake_ci = SimpleNamespace(evaluate=lambda t: sun_position)

    fake_params = Params(0, 0, 0, 0, 0, 0, 0, 0, fake_ci, fake_ci)

    assert simple_eclipse(0, state, fake_params)
