import numpy as np
import pytest
from membrane_report.synthetic import _hh_rhs
from membrane_report.hh import simulate_hh


def test_temperature_scales_gate_derivatives_and_preserves_membrane_equation():
    state = [-40, 0.1, 0.4, 0.3]
    cold = np.array(_hh_rhs(0, state, lambda _: 7))
    warm = np.array(_hh_rhs(0, state, lambda _: 7, temperature_c=16.3))
    assert np.all(np.isfinite(cold))
    assert warm[0] == cold[0]
    np.testing.assert_allclose(warm[1:], cold[1:] * 3, rtol=1e-14)


def test_pulse_pair_recovers_a_second_spike_at_longer_interval():
    short = simulate_hh(20, protocol="pair", interval_ms=4)
    long = simulate_hh(20, protocol="pair", interval_ms=25)
    assert short["spike_count"] == 1
    assert long["spike_count"] == 2
    assert np.sum(long["i"] > 0) == 40
    assert long["state"][np.searchsorted(long["t_ms"], 75), 2] > short["state"][np.searchsorted(short["t_ms"], 54), 2]


def test_no_sodium_no_spike_and_invalid_domains_fail():
    assert simulate_hh(20, sodium=0)["spike_count"] == 0
    for args in [{"temperature_c":np.nan}, {"temperature_c":40}, {"protocol":"pair", "interval_ms":2}]:
        with pytest.raises(ValueError): simulate_hh(**args)
