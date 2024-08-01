from __future__ import annotations

from dataclasses import asdict, dataclass
from json import JSONEncoder, loads

import jax.numpy as jnp
from jax import Array


class ArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, jnp.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


@dataclass(frozen=True)
class SimConfig:
    """
    Simulation configuration.

    Attributes
    ----------
    name : str
        Mission name.
    y0 : Array, shape (6,)
        Initial state vector.
    y_target : Array, shape (5,)
        Target state vector.
    propulsion_model : str
        Propulsion model.
    steering_law : str
        Steering law.
    t_span : tuple[float, float]
        Time span in seconds.
    solver : str
        ODE solver.
    conv_tol : float
        Guidance convergence tolerance.
    w_oe : Array, shape (5,)
        Guidance weights.
    w_penalty : float
        Penalty weight.
    penalty_function : str
        Penalty function.
    kappa : float
        Cone angle parameter.
    dynamics : str
        Dynamics type.
    perturbations : list[str]
        List of perturbation functions.
    characteristic_accel : float
        Characteristic acceleration.
    epoch_jd : float
        Epoch Julian date.
    """

    name: str  # mission name
    y0: Array  # shape (6,)
    y_target: Array  # shape (5,)
    propulsion_model: str
    steering_law: str
    t_span: tuple[float, float]
    conv_tol: float
    w_oe: Array
    w_penalty: float
    penalty_function: str
    kappa: float
    dynamics: str
    perturbations: list[str]
    characteristic_accel: float
    epoch_jd: float

    def __post_init__(self):
        # Validate the configuration
        assert len(self.y0) == 6, "Initial state vector must have length 6"
        assert len(self.y_target) == 5, "Target state vector must have length 5"

    def serialize(self) -> str:
        return ArrayEncoder(indent=4).encode(asdict(self))

    @classmethod
    def deserialize(cls, s: str) -> SimConfig:
        d = loads(s)

        # Make sure array types are properly set
        d["y0"] = jnp.array(d["y0"], dtype=jnp.float64)
        d["y_target"] = jnp.array(d["y_target"], dtype=jnp.float64)
        d["w_oe"] = jnp.array(d["w_oe"], dtype=jnp.float64)

        return cls(**d)
