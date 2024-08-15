import jax.numpy as jnp

from qlipper.configuration import SimConfig
from qlipper.constants import MU_EARTH
from qlipper.converters import cartesian_to_mee
from qlipper.postprocess import postprocess_run
from qlipper.run.mission_runner import run_mission
from qlipper.sim.ephemeris import generate_ephem_arrays, lookup_body_id

# precompute Moon's orbital elements
moon_t_sample, moon_state_sample = generate_ephem_arrays(
    lookup_body_id("earth"), lookup_body_id("moon"), 2451545.0, (0, 1), 2
)

# moon's orbital elements wrt Earth
mee_moon = cartesian_to_mee(moon_state_sample[:, 0] * 1e3, MU_EARTH)[:5]


sim_case = SimConfig(
    name="lunar",
    y0=jnp.array([200000e3, 0, 0, 0, 0, 0]),
    y_target=mee_moon,
    propulsion_model="constant_thrust",
    steering_law="bbq_law",
    t_span=(0, 1e8),
    conv_tol=1e-2,
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
