import jax.numpy as jnp

from qlipper.configuration import SimConfig
from qlipper.postprocess import postprocess_run
from qlipper.run.diagnostics import single_step_debug
from qlipper.run.mission_runner import run_mission

sim_case = SimConfig(
    name="lunar",
    y0=jnp.array([100000e3, 0, 0, 0, 0, 0]),
    y_target=jnp.array([10000e3, 0, 0, 0, 0]),
    propulsion_model="constant_thrust",
    steering_law="bbq_law",
    t_span=(0, 86400 * 200),
    conv_tol=0.5,
    earth_w_oe=jnp.array([1, 1, 1, 1, 1]),
    earth_penalty_weight=1,
    earth_penalty_scaling=100,
    earth_rp_min=6400e3,
    moon_w_oe=jnp.array([10, 1, 1, 1, 1]),
    moon_penalty_weight=1,
    moon_penalty_scaling=10,
    moon_rp_min=1800e3,
    perturbations=["moon_gravity"],
    epoch_jd=2453545.0,
    characteristic_accel=0.001,
)

# print(single_step_debug(sim_case))
run_id, t, y = run_mission(sim_case)
postprocess_run(run_id, t, y, sim_case)
