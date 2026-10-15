"""Sweep quality control using IPFX 2.1 windows and "Ephys QC Criteria v1.1" limits."""

from __future__ import annotations

import numpy as np

from .epochs import long_square_stimulus, recording_end_index, split_test_pulse

NOISE_EPOCH_S = 0.0015
STABILITY_EPOCH_S = 0.5
POST_STIM_REQUIRED_S = 0.5
LIMITS = {
    "pre_noise_rms_mv": 0.07,
    "post_noise_rms_mv": 0.07,
    "slow_noise_rms_mv": 0.5,
    "vm_delta_mv": 1.0,
}
LIMITS_SOURCE = "IPFX 2.1 defaults/qc_criteria.json, 'Ephys QC Criteria v1.1'"


def _mean_rms(x):
    if len(x) == 0 or not np.all(np.isfinite(x)):
        return np.nan, np.nan
    m = float(np.mean(x))
    return m, float(np.sqrt(np.mean((x - m) ** 2)))


def sweep_qc(sweep) -> dict:
    """QC measures, pass flag and failure reasons for one sweep."""
    hz = sweep.sampling_rate_hz
    v_mv = sweep.voltage_volts * 1e3
    segs = sweep.stimulus_intervals
    test, _ = split_test_pulse(segs)
    stim = long_square_stimulus(segs)
    rec_end = recording_end_index(sweep.voltage_volts, hz)
    out = {
        "cell_id": sweep.cell_id,
        "sweep_id": sweep.sweep_id,
        "protocol": sweep.protocol_name,
        "test_pulse_present": test is not None,
        "stimulus_start_s": stim[0] if stim else np.nan,
        "stimulus_end_s": stim[1] if stim else np.nan,
        "stimulus_amplitude_pa": stim[2] * 1e12 if stim else np.nan,
        "recording_end_s": rec_end / hz,
        "nominal_end_s": len(v_mv) / hz,
        "source_flags": "; ".join(sweep.source_quality_flags),
    }
    reasons = []
    if not np.all(np.isfinite(sweep.injected_current_amperes)):
        reasons.append("command current contains non-finite values")
    if not np.all(np.isfinite(sweep.voltage_volts[:rec_end])):
        reasons.append("voltage contains non-finite values before the recording end")
    if stim is None:
        reasons.append("no single long step identified")
        out.update({k: np.nan for k in ("pre_noise_rms_mv", "slow_noise_rms_mv", "pre_vm_mv", "post_noise_rms_mv",
                                        "post_vm_mv", "vm_delta_mv")})
        out["complete"] = False
    else:
        s_idx = int(round(stim[0] * hz))
        e_idx = int(round(stim[1] * hz))
        complete = rec_end >= e_idx + int(POST_STIM_REQUIRED_S * hz)
        out["complete"] = bool(complete)
        if not complete:
            reasons.append(f"recording stops at {rec_end / hz:.3f} s, before step end + {POST_STIM_REQUIRED_S} s")
        i0 = s_idx - int(STABILITY_EPOCH_S * hz)
        if i0 < 1:
            reasons.append("insufficient pre-step baseline for QC")
            out["pre_noise_rms_mv"] = out["pre_vm_mv"] = out["slow_noise_rms_mv"] = np.nan
        else:
            _, out["pre_noise_rms_mv"] = _mean_rms(v_mv[i0:i0 + int(NOISE_EPOCH_S * hz)])
            out["pre_vm_mv"], out["slow_noise_rms_mv"] = _mean_rms(v_mv[s_idx - 1 - int(STABILITY_EPOCH_S * hz):s_idx - 1])
        if complete:
            # IPFX post windows end before the last valid sample
            last = rec_end - 1
            out["post_vm_mv"], _ = _mean_rms(v_mv[last - int(STABILITY_EPOCH_S * hz):last])
            _, out["post_noise_rms_mv"] = _mean_rms(v_mv[last - int(NOISE_EPOCH_S * hz):last])
            out["vm_delta_mv"] = abs(out["pre_vm_mv"] - out["post_vm_mv"])
        else:
            out["post_vm_mv"] = out["post_noise_rms_mv"] = out["vm_delta_mv"] = np.nan
        for k, lim in LIMITS.items():
            val = out[k]
            if complete and not np.isfinite(val):
                reasons.append(f"{k} is unavailable or non-finite")
            elif np.isfinite(val) and val > lim:
                reasons.append(f"{k} {val:.3g} > {lim}")
    out["qc_pass"] = not reasons
    out["qc_reasons"] = "; ".join(reasons)
    return out
