import pytest

from qlipper.sim.ephemeris import (
    _ensure_ephemeris,
    compute_kernel_at_times,
    path_to_named_string,
    resolve_spk_path,
)


@pytest.fixture
def kernel():
    return _ensure_ephemeris()


def test_retrieval(kernel):
    assert kernel is not None


def test_resolve_spk_path(kernel):
    path = resolve_spk_path(kernel, 399, 4)
    assert path == [399, 3, 0, 4]


def test_path_to_named_string():
    path = [399, 3, 0, 4]
    assert (
        path_to_named_string(path)
        == "Earth -> Earth Barycenter -> Solar System Barycenter -> Mars Barycenter"
    )


def test_simple_ephemeris(kernel):
    position_ref = kernel[0, 4].compute(2457061.5)
    position = compute_kernel_at_times(kernel, 0, 4, [2457061.5])[:, 0]

    assert position_ref == pytest.approx(position, rel=1e-6)


def test_multileg_ephemeris(kernel):
    position_ref = kernel[0, 4].compute(2457061.5)
    position_ref -= kernel[0, 3].compute(2457061.5)
    position_ref -= kernel[3, 399].compute(2457061.5)
    position = compute_kernel_at_times(kernel, 399, 4, [2457061.5])[:, 0]

    assert position_ref == pytest.approx(position, rel=1e-6)
