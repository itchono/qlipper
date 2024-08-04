import jax.numpy as jnp

from qlipper.configuration import SimConfig
from qlipper.postprocess import postprocess_run
from qlipper.run.diagnostics import single_step_debug
from qlipper.run.mission_runner import run_mission

sim_case = SimConfig(
    name="demo",
    y0=jnp.array([20000e3, 0, 0, 0, 0, 0]),
    y_target=jnp.array([21000e3, 0, 0, 0, 0]),
    propulsion_model="constant_thrust",
    steering_law="trivial_steering",
    t_span=(0, 2e5),
    conv_tol=1e-3,
    w_oe=jnp.array([1, 0, 0, 0, 0]),
    w_penalty=0,
    penalty_function="",
    kappa=jnp.deg2rad(64),
    dynamics="cartesian",
    perturbations=[],
    epoch_jd=2451545.0,
    characteristic_accel=0.001,
)


run_id, t, y = run_mission(sim_case)
postprocess_run(run_id, t, y, sim_case)
