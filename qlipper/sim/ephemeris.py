import urllib.request
from collections import defaultdict
from pathlib import Path

import jax.numpy as jnp
from jax.typing import ArrayLike
from jplephem.names import target_names
from jplephem.spk import SPK, Segment

DE440S_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp"
EPHEM_CACHE_DIR = Path(__file__).parent / "ephem_cache"


def _ensure_ephemeris() -> SPK:
    # cache the ephemeris file
    EPHEM_CACHE_DIR.mkdir(exist_ok=True)

    ephemeris_file = EPHEM_CACHE_DIR / "de440s.bsp"

    if not ephemeris_file.exists():
        print("First Time Setup: Downloading DE440S ephemeris...")
        urllib.request.urlretrieve(DE440S_URL, ephemeris_file)

    return SPK.open(ephemeris_file)


def find_path_bfs(start: int, end: int, graph: dict[int, list[int]]) -> list[int]:
    """
    Breadth-first search to find a path between two nodes in a graph

    Parameters
    ----------
    start : int
        The starting node
    end : int
        The ending node
    graph : dict[int, list[int]]
        The graph as an adjacency list

    Returns
    -------
    list[int]
        The path from start to end
    """
    queue = [(start, [])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in set(graph[vertex]) - set(path):
            if next == end:
                return path + [vertex, next]
            else:
                queue.append((next, path + [vertex]))


def resolve_spk_path(kernel: SPK, observer: int, target: int) -> list[int]:
    """
    Determine the path to compute the kernel

    Parameters
    ----------
    kernel : SPK
        The SPK kernel to use
    observer : int
        The observer body ID
    target : int
        The target body ID

    Returns
    -------
    list[int]
        The path from observer to target
    """
    adj_list = defaultdict(list)

    for bo, bt in kernel.pairs.keys():
        adj_list[bo].append(bt)
        adj_list[bt].append(bo)

    return find_path_bfs(observer, target, adj_list)


def path_to_named_string(path: list[int]) -> str:
    """
    Convert a path to a human-readable string

    Parameters
    ----------
    path : list[int]
        The path to convert

    Returns
    -------
    str
        The human-readable string
    """
    return " -> ".join([target_names[body_id].title() for body_id in path])


def compute_kernel_at_times(
    kernel: SPK, observer: int, target: int, jd_samples: ArrayLike
) -> ArrayLike:
    """
    Evaluate kernel at given times

    Automatically resolves the correct set of segments to use for the interpolation

    Parameters
    ----------
    kernel : SPK
        The SPK kernel to use
    observer : int
        The observer body ID
    target : int
        The target body ID
    jd_range : ArrayLike
        The Julian dates to evaluate the kernel at

    Returns
    -------
    ArrayLike
        The position vectors of the target body at the given times
    """
    path = resolve_spk_path(kernel, observer, target)

    result = jnp.zeros((3, len(jd_samples)))

    # the kernel only stores "forward pairs",
    # so we try to evaluate the segment in the forward direction first,
    # and if it fails, we try the reverse direction, with a sign flip
    # this is the EAFP way to do it
    for i in range(0, len(path) - 1):
        try:
            segment: Segment = kernel[path[i], path[i + 1]]
            result += segment.compute(jd_samples)
        except KeyError:
            segment: Segment = kernel[path[i + 1], path[i]]
            result -= segment.compute(jd_samples)

    return result
