import numpy as np
import pytest

from membrane_report.epochs import long_square_stimulus, recording_end_index, split_test_pulse, step_segments
from membrane_report.recording import Sweep, sweep_from_arrays
from membrane_report.synthetic import RATE_HZ, command, passive_sweep


def test_segments_test_pulse_and_step():
    t = np.arange(int(2.75 * RATE_HZ)) / RATE_HZ
    segs = step_segments(command(t, -70e-12), RATE_HZ)
    assert len(segs) == 2
    test, rest = split_test_pulse(segs)
    assert test[0] == pytest.approx(0.045) and test[1] == pytest.approx(0.055) and test[2] == pytest.approx(50e-12)
    stim = long_square_stimulus(segs)
    assert stim[0] == pytest.approx(0.6) and stim[1] == pytest.approx(1.6) and stim[2] == pytest.approx(-70e-12)


def test_no_step_and_two_steps_are_not_guessed():
    t = np.arange(int(2.75 * RATE_HZ)) / RATE_HZ
    assert long_square_stimulus(step_segments(command(t, 0.0), RATE_HZ)) is None
    i = command(t, -50e-12)
    i[(t >= 1.9) & (t < 2.5)] = -20e-12
    assert long_square_stimulus(step_segments(i, RATE_HZ)) is None


def test_recording_end_detects_zero_padding():
    v = np.full(1000, -0.07)
    assert recording_end_index(v) == 1000
    v[700:] = 0.0
    assert recording_end_index(v) == 700
    v2 = np.full(1000, -0.07)
    v2[-3:] = np.nan
    assert recording_end_index(v2) == 997
    v3 = np.full(100_000, -0.07)
    v3[-10:] = 0.0
    assert recording_end_index(v3, 50_000.0) == 100_000


def test_timestamps_are_validated():
    t = np.arange(10) / 1000.0
    v = np.zeros(10) - 70
    i = np.zeros(10)
    s = sweep_from_arrays(t, v, i, v_unit="mV", i_unit="pA", cell_id="c", sweep_id=1, source_asset_id="x",
                          protocol_name="p")
    assert s.sampling_rate_hz == pytest.approx(1000.0)
    assert s.voltage_volts[0] == pytest.approx(-0.07)
    bad = t.copy()
    bad[5] = bad[4]
    with pytest.raises(ValueError, match="strictly increasing"):
        sweep_from_arrays(bad, v, i, cell_id="c", sweep_id=1, source_asset_id="x", protocol_name="p")
    uneven = t.copy()
    uneven[5:] += 0.0004
    with pytest.raises(ValueError, match="uniformly"):
        sweep_from_arrays(uneven, v, i, cell_id="c", sweep_id=1, source_asset_id="x", protocol_name="p")
    nan_t = t.copy()
    nan_t[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sweep_from_arrays(nan_t, v, i, cell_id="c", sweep_id=1, source_asset_id="x", protocol_name="p")


def test_sweep_rejects_mismatched_arrays():
    with pytest.raises(ValueError):
        Sweep(cell_id="c", sweep_id=1, source_asset_id="x", protocol_name="p", sampling_rate_hz=1000.0,
              voltage_volts=np.zeros(10), injected_current_amperes=np.zeros(9))


def test_unit_conversion_is_applied_once():
    s = passive_sweep(-50e-12)
    mv = sweep_from_arrays(s.time_seconds, s.voltage_volts * 1e3, s.injected_current_amperes * 1e12, v_unit="mV",
                           i_unit="pA", cell_id="c", sweep_id=1, source_asset_id="x", protocol_name="X1PS_SubThresh")
    np.testing.assert_allclose(mv.voltage_volts, s.voltage_volts, rtol=1e-15)
    np.testing.assert_allclose(mv.injected_current_amperes, s.injected_current_amperes, rtol=1e-15, atol=1e-30)


def test_timestamps_cannot_describe_a_different_number_of_samples():
    with pytest.raises(ValueError, match="same length"):
        sweep_from_arrays(np.arange(10) / 1000, np.full(5, -0.07), np.zeros(5), cell_id="c", sweep_id=1,
                          source_asset_id="x", protocol_name="p")


def test_contiguous_steps_with_different_currents_are_distinct():
    current = np.zeros(2000)
    current[100:700] = -50e-12
    current[700:1300] = -80e-12
    segments = step_segments(current, 1000)
    assert len(segments) == 2
    assert long_square_stimulus(segments) is None


def test_empty_recording_end_is_defined():
    assert recording_end_index([]) == 0
