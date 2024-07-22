from __future__ import annotations

from enum import Enum
from typing import Callable

from diffrax import AbstractSolver
from jax import Array
from jax_dataclasses import asdict, pytree_dataclass

PROPULSION_MODEL_TYPE = Callable[[float, Array, "SimConfig", float, float], Array]
# thrust vector as function of (t, y, cfg, alpha, beta) -> Array (3,)

STEERING_LAW_TYPE = Callable[[float, Array, "SimConfig"], tuple[float, float]]
# steering control as function of (t, y, cfg) -> (alpha, beta)

PENALTY_FUNCTION_TYPE = Callable[[Array], float]
# penalty function as function of state vector -> float

PERTURBATION_TYPE = Callable[[float, Array], Array]
# perturbation as function of state vector -> Array (3,)


@pytree_dataclass(frozen=True)
class SimConfig:
    name: str  # mission name
    y0: Array  # initial state vector, shape (6,)
    y_target: Array  # target state vector, shape (5,)
    propulsion_model: PROPULSION_MODEL_TYPE  # propulsion model
    steering_law: STEERING_LAW_TYPE  # steering law
    t_span: tuple[float, float]  # (s)
    solver: AbstractSolver  # solver
    conv_tol: float  # convergence tolerance
    w_oe: Array  # guidance weights, shape (5,)
    w_penalty: float  # penalty weight
    penalty_function: PENALTY_FUNCTION_TYPE  # penalty as function of state vector
    kappa: float  # cone angle parameter
    dynamics: str  # dynamics type "mee" or "cart"
    perturbations: list[PERTURBATION_TYPE]  # list of perturbation functions
    characteristic_accel: float  # characteristic acceleration (m/s^2)
    epoch_jd: float  # epoch Julian date

    def serialize(self) -> dict:
        # replace all functions with their names so that the config can be serialized
        d = asdict(self)

        # (Arrays)
        for key, value in d.items():
            if isinstance(value, Array):
                d[key] = value.tolist()

        # solver
        solver_name = self.solver.__class__.__name__
        d["solver"] = solver_name

        # propulsion model
        propulsion_model_name = self.propulsion_model.__name__
        d["propulsion_model"] = propulsion_model_name

        # steering law
        steering_law_name = self.steering_law.__name__
        d["steering_law"] = steering_law_name

        # penalty function
        penalty_function_name = self.penalty_function.__name__
        d["penalty_function"] = penalty_function_name

        # perturbations
        perturbations_names = [p.__name__ for p in self.perturbations]
        d["perturbations"] = perturbations_names

        return d

    @classmethod
    def deserialize(cls, d: dict) -> SimConfig:
        # replace all function names with the actual functions
        d = d.copy()

        # TODO: some importlib magic to get the functions from their names

        raise NotImplementedError()

        return cls(**d)
