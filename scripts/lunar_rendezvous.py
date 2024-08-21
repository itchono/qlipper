import jax.numpy as jnp

from qlipper.configuration import SimConfig
from qlipper.postprocess import postprocess_run
from qlipper.run.mission_runner import run_mission

sim_case = SimConfig(
    name="lunar",
    y0=jnp.array([100000e3, 0, 0, 0, 0, 0]),
    y_target=jnp.array([10000e3, 0.1, 0, 0.5, 0]),
    propulsion_model="constant_thrust",
    steering_law="bbq_law",
    t_span=(0, 2e7),
    conv_tol=2,
    w_oe=jnp.array([10, 0.5, 1, 2, 1]),
    w_penalty=0,
    penalty_function="",
    kappa=jnp.deg2rad(64),
    dynamics="cartesian",
    perturbations=["moon_gravity"],
    epoch_jd=2451545.0,
    characteristic_accel=0.001,
)


run_id, t, y = run_mission(sim_case)
postprocess_run(run_id, t, y, sim_case)
