from qlipper.steering.multibody import bbq_law, qbbq_law
from qlipper.steering.q_law import q_law
from qlipper.steering.quail import quail
from qlipper.steering.trivial import trivial_steering

__all__ = ["q_law", "trivial_steering", "quail", "bbq_law", "qbbq_law"]


STEERING_LAWS = {
    "q_law": q_law,
    "trivial_steering": trivial_steering,
    "quail": quail,
    "qbbq_law": qbbq_law,
    "bbq_law": bbq_law,
}
