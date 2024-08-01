import jax.numpy as jnp

from qlipper.configuration import SimConfig
from qlipper.postprocess import postprocess_run
from qlipper.run.mission_runner import run_mission

sim_case = SimConfig(
    name="demo",
    y0=jnp.array([20000e3, 0, 0, 0, 0, 0]),
    y_target=jnp.array([21000e3, 0, 0, 0.5, 0]),
    propulsion_model="constant_thrust",
    steering_law="q_law",
    t_span=(0, 10 * 86400),
    conv_tol=1e-1,
    w_oe=jnp.array([1, 1, 1, 1, 1]),
    w_penalty=0,
    penalty_function="",
    kappa=jnp.deg2rad(64),
    dynamics="mee",
    perturbations=[],
    characteristic_accel=0.01,
    epoch_jd=2451545.0,
)


y, t = run_mission(sim_case)
postprocess_run(t, y, sim_case)
