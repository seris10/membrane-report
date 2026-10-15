"""NWB adapter tests on a synthetic NWB file."""

from datetime import datetime, timezone

import numpy as np
import pytest

from membrane_report.recording import read_nwb


def write_nwb(path, v_raw, i_raw, v_conv, v_off, i_conv, rate=50_000.0, bias=None,
              stimulus_rate=None, stimulus_start=0.0, timestamps=None):
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.file import Subject
    from pynwb.icephys import CurrentClampSeries, CurrentClampStimulusSeries

    nwb = NWBFile(session_description="synthetic", identifier="synthetic-0001",
                  session_start_time=datetime(2026, 10, 13, tzinfo=timezone.utc),
                  subject=Subject(subject_id="synthetic", species="Mus musculus"))
    dev = nwb.create_device(name="amp")
    el = nwb.create_icephys_electrode(name="e0", description="synthetic", device=dev)
    stim = CurrentClampStimulusSeries(name="data_00001_DA0", data=i_raw, unit="amperes", conversion=float(i_conv),
                                      rate=stimulus_rate or rate, starting_time=stimulus_start, electrode=el, gain=1.0,
                                      sweep_number=np.uint32(1), stimulus_description="X1PS_SubThresh_DA_0")
    resp = CurrentClampSeries(name="data_00001_AD0", data=v_raw, unit="volts", conversion=float(v_conv), offset=float(v_off),
                              rate=rate if timestamps is None else None,
                              starting_time=0.0 if timestamps is None else None, timestamps=timestamps,
                              electrode=el, gain=1.0, sweep_number=np.uint32(1),
                              stimulus_description="X1PS_SubThresh_DA_0", bias_current=bias)
    nwb.add_stimulus(stim)
    nwb.add_acquisition(resp)
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)


def test_conversion_and_offset_applied_once(tmp_path):
    rate = 50_000.0
    n = int(2.75 * rate)
    v_mv = np.full(n, -70.0, dtype=np.float32)
    i_pa = np.zeros(n, dtype=np.float32)
    i_pa[int(0.6 * rate):int(1.6 * rate)] = -50.0
    p = tmp_path / "s.nwb"
    write_nwb(p, v_mv, i_pa, 1e-3, 0.005, 1e-12, rate, bias=-20.0)
    cell, sweeps = read_nwb(p, "synthetic")
    s = sweeps[0]
    assert s.voltage_volts[0] == pytest.approx(-0.070 + 0.005, rel=1e-6)
    assert s.injected_current_amperes[int(1.0 * rate)] == pytest.approx(-50e-12, rel=1e-6)
    assert s.recording_metadata["voltage_conversion"] == pytest.approx(1e-3)
    assert s.recording_metadata["voltage_offset"] == pytest.approx(0.005)
    assert any("bias_current" in f for f in s.source_quality_flags)
    assert cell["recording_temperature"] is None
    assert cell["stimulus_codes"] == ["X1PS_SubThresh"]


@pytest.mark.parametrize("options", [{"stimulus_rate": 40_000.0}, {"stimulus_start": .01}])
def test_misaligned_stimulus_is_rejected(tmp_path, options):
    p = tmp_path / "unaligned.nwb"
    write_nwb(p, np.full(100, -0.07), np.zeros(100), 1, 0, 1, **options)
    with pytest.raises(ValueError, match="do not align"):
        read_nwb(p, "synthetic")


def test_unequal_stimulus_and_voltage_lengths_are_not_truncated(tmp_path):
    p = tmp_path / "unequal.nwb"
    write_nwb(p, np.full(100, -0.07), np.zeros(90), 1, 0, 1)
    with pytest.raises(ValueError, match="stimulus length"):
        read_nwb(p, "synthetic")


def test_regular_nwb_timestamps_are_supported_and_irregular_ones_rejected(tmp_path):
    t = np.arange(100) / 50_000
    p = tmp_path / "timestamps.nwb"
    write_nwb(p, np.full(100, -0.07), np.zeros(100), 1, 0, 1, timestamps=t)
    _, sweeps = read_nwb(p, "synthetic")
    assert sweeps[0].sampling_rate_hz == pytest.approx(50_000)
    t[50:] += .00001
    bad = tmp_path / "irregular.nwb"
    write_nwb(bad, np.full(100, -0.07), np.zeros(100), 1, 0, 1, timestamps=t)
    with pytest.raises(ValueError, match="uniformly spaced"):
        read_nwb(bad, "synthetic")


def test_negative_conversion_factors_are_applied_once(tmp_path):
    p = tmp_path / "inverted.nwb"
    write_nwb(p, np.full(100, 70.0), np.full(100, 50.0), -1e-3, 0, -1e-12)
    _, sweeps = read_nwb(p, "synthetic")
    np.testing.assert_allclose(sweeps[0].voltage_volts, -.07, rtol=1e-15)
    np.testing.assert_allclose(sweeps[0].injected_current_amperes, -50e-12, rtol=1e-15)
