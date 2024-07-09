import jax.numpy as jnp
from diffrax import Dopri8

from nrs.configuration import DynamicsType, SimConfig
from nrs.sim.propulsion import constant_thrust
from nrs.steering.q_law import q_law

sim_case = SimConfig(
    y0=jnp.array([20000e3, 0.5, -0.2, 0.5, 0, 0]),
    y_target=jnp.array([25000e3, 0.2, 0.5, 0, 0.3]),
    propulsion_model=constant_thrust,
    steering_law=q_law,
    t_span=(0, 1e8),
    solver=Dopri8(),
    conv_tol=1e-3,
    w_oe=jnp.array([1, 1, 1, 1, 1]),
    w_penalty=0,
    penalty_function=lambda x: 0,
    kappa=jnp.deg2rad(64),
    dynamics_type=DynamicsType.MEE,
    perturbations=[],
    characteristic_accel=0.1,
)
