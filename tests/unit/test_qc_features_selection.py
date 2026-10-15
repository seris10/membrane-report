import numpy as np
import pytest

from membrane_report.features import baseline_voltage, detect_spikes, fi_slope, firing_summary, peak_deflection
from membrane_report.qc import sweep_qc
from membrane_report.selection import evaluate_cell, session_id
from membrane_report.synthetic import STEP, hh_sweep, passive_sweep


def test_clean_sweep_passes_qc():
    q = sweep_qc(passive_sweep(-50e-12))
    assert q["qc_pass"], q["qc_reasons"]
    assert q["complete"] and q["test_pulse_present"]
    # pre-step window includes the test-pulse tail
    assert 0 < q["vm_delta_mv"] < 0.01


def test_aborted_noisy_and_drifting_sweeps_fail_with_reasons():
    q = sweep_qc(passive_sweep(-50e-12, stop_at_s=0.9))
    assert not q["qc_pass"] and "recording stops" in q["qc_reasons"]
    q = sweep_qc(passive_sweep(-50e-12, noise_sd=0.2e-3, seed=3))
    assert not q["qc_pass"] and "pre_noise_rms_mv" in q["qc_reasons"]
    s = passive_sweep(-50e-12)
    s.voltage_volts[int(1.8 * s.sampling_rate_hz):] += 0.002
    q = sweep_qc(s)
    assert not q["qc_pass"] and "vm_delta_mv" in q["qc_reasons"]


def test_hodgkin_huxley_spike_counts_match_the_simulation():
    for j in (3.0, 9.0):
        s, n_true = hh_sweep(j)
        t, pk = detect_spikes(s.voltage_volts, s.sampling_rate_hz, STEP[0], STEP[1])
        assert len(t) == n_true
        assert np.all(pk > 0)
    s, n_true = hh_sweep(1.0)
    assert n_true == 0 and firing_summary(s, *STEP)["spike_count"] == 0


def test_padding_jump_is_not_a_spike():
    s = passive_sweep(100e-12, stop_at_s=1.2)
    assert firing_summary(s, *STEP)["spike_count"] == 0
    raw_t, _ = detect_spikes(s.voltage_volts, s.sampling_rate_hz, STEP[0], STEP[1])
    assert len(raw_t) == 1, "the unguarded detector sees the padding jump; firing_summary must not"


def test_baseline_and_peak_deflection():
    s = passive_sweep(-50e-12, r_in=200e6, tau=0.01, v0=-0.065)
    assert baseline_voltage(s, 0.6) == pytest.approx(-0.065, abs=1e-12)
    vpk, tpk = peak_deflection(s, 0.6, 1.6)
    assert vpk == pytest.approx(-0.065 - 0.01, rel=1e-6)
    assert tpk > 0.6 + 5 * 0.01


def test_fi_slope():
    assert fi_slope([100, 150, 200, 250], [0, 5, 10, 15]) == pytest.approx(0.1)
    assert np.isnan(fi_slope([100, 150], [0, 5]))


def test_selection_rules():
    from membrane_report.recording import Sweep

    def sw(code, amp, sid):
        s = passive_sweep(amp, sweep_id=sid)
        return Sweep(cell_id="c", sweep_id=sid, source_asset_id="x", protocol_name=code,
                     sampling_rate_hz=s.sampling_rate_hz, voltage_volts=s.voltage_volts,
                     injected_current_amperes=s.injected_current_amperes)

    ok = [sw("X1PS_SubThresh", a, k) for k, a in enumerate((-30e-12, -50e-12, -70e-12))]
    ok += [sw("X4PS_SupraThresh", a, 10 + k) for k, a in enumerate((200e-12, 250e-12))]
    assert evaluate_cell(ok)["qualifies"]
    few = ok[:2] + ok[3:]
    ev = evaluate_cell(few)
    assert not ev["qualifies"] and ev["reasons"][0].startswith("C2")
    old = [sw("C1LSCOARSE150216", -50e-12, 1)]
    ev = evaluate_cell(old, ["C1LSCOARSE150216", "C1SSFINEST150112"])
    assert not ev["qualifies"] and ev["reasons"][0].startswith("C0") and "C1LSCOARSE150216" in ev["reasons"][0]
    assert session_id("sub-1/sub-1_ses-993300347_icephys.nwb") == 993300347


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_invalid_voltage_inside_recording_fails_qc(value):
    s = passive_sweep(-50e-12)
    s.voltage_volts[int(0.2 * s.sampling_rate_hz)] = value
    q = sweep_qc(s)
    assert not q["qc_pass"] and "non-finite" in q["qc_reasons"]


def test_missing_prestep_qc_window_does_not_pass():
    s = passive_sweep(-50e-12)
    s.injected_current_amperes[:] = 0
    s.injected_current_amperes[int(.2 * s.sampling_rate_hz):int(1.2 * s.sampling_rate_hz)] = -50e-12
    q = sweep_qc(s)
    assert not q["qc_pass"] and "insufficient pre-step" in q["qc_reasons"]


def test_invalid_command_fails_qc():
    s = passive_sweep(-50e-12)
    s.injected_current_amperes[0] = np.inf
    assert not sweep_qc(s)["qc_pass"]
