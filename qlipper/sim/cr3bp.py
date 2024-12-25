# cr3bp ephemeris computation
import logging
import urllib.request
from collections import defaultdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jplephem.names import target_name_pairs, target_names
from jplephem.spk import SPK, Segment
from numpy.typing import NDArray


def generate_ephem_arrays(
    observer: int, target: int, epoch: float, t_span: ArrayLike, num_samples: int
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Generate arrays of ephemeris data for a given observer and target

    Parameters
    ----------
    observer : int
        The observer body
    target : int
        The target body
    epoch : float
        The epoch of the ephemeris data
    t_span : ArrayLike
        The time span over which to compute the ephemeris
    num_samples : int
        The number of samples to take over t_span

    Returns
    -------
    tuple[NDArray[np.floating], NDArray[np.floating]]
        The times and positions of the ephemeris data
    """
    # ensure it's moon and Earth only

    # observer = 399
    # target = 301
