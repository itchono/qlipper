import jax.numpy as jnp
from diffrax import Dopri8

from qlipper.configuration import SimConfig
from qlipper.postprocess.traj_plotters import plot_trajectory_mee
from qlipper.run.mission_runner import run_mission
from qlipper.sim.propulsion import constant_thrust
from qlipper.steering import trivial_steering

sim_case = SimConfig(
    name="demo",
    y0=jnp.array([20000e3, 0, 0, 0, 0, 0]),
    y_target=jnp.array([20000e3, 0, 0, 0, 0]),
    propulsion_model=constant_thrust,
    steering_law=trivial_steering,
    t_span=(0, 2e5),
    solver=Dopri8(),
    conv_tol=1e-3,
    w_oe=jnp.array([1, 1, 1, 1, 1]),
    w_penalty=0,
    penalty_function=lambda x: 0,
    kappa=jnp.deg2rad(64),
    dynamics="mee",
    perturbations=[],
    characteristic_accel=0.01,
    epoch_jd=2451545.0,
)


y, t = run_mission(sim_case)

plot_trajectory_mee(y)
