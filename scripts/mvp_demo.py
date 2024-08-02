import jax.numpy as jnp

from qlipper.configuration import SimConfig
from qlipper.postprocess import postprocess_run
from qlipper.run.mission_runner import run_mission

sim_case = SimConfig(
    name="demo",
    y0=jnp.array([20000e3, 0, 0, 0, 0, 0]),
    y_target=jnp.array([22000e3, 0, 0, 0, 0]),
    propulsion_model="ideal_solar_sail",
    steering_law="quail",
    t_span=(0, 1e7),
    conv_tol=1e-2,
    w_oe=jnp.array([1, 1, 1, 1, 1]),
    w_penalty=0,
    penalty_function="",
    kappa=64,
    dynamics="mee",
    perturbations=[],
    characteristic_accel=0.01,
    epoch_jd=2451545.0,
)


run_id, t, y = run_mission(sim_case)
postprocess_run(run_id, t, y, sim_case)
