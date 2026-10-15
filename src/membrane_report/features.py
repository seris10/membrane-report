"""Baseline, peak deflection, spike detection and firing rate."""

from __future__ import annotations

import numpy as np
from scipy.signal import bessel, sosfiltfilt

DVDT_THRESHOLD_V_PER_S = 20.0
MIN_PEAK_V = -0.030
MIN_HEIGHT_V = 0.002
PEAK_SEARCH_S = 0.002
FILTER_HZ = 10_000.0
BASELINE_WINDOW_S = 0.1


def filtered(v, rate_hz, cutoff_hz=FILTER_HZ):
    if cutoff_hz >= rate_hz / 2:
        return np.asarray(v, float)
    sos = bessel(4, cutoff_hz, btype="low", fs=rate_hz, output="sos", norm="phase")
    return sosfiltfilt(sos, v)


def baseline_voltage(sweep, start_s, window_s=BASELINE_WINDOW_S):
    hz = sweep.sampling_rate_hz
    i1 = int(round(start_s * hz))
    i0 = max(i1 - int(round(window_s * hz)), 0)
    return float(np.mean(sweep.voltage_volts[i0:i1]))


def peak_deflection(sweep, start_s, end_s, kind="min"):
    """(voltage, time) of the most negative (or positive) voltage within the step."""
    hz = sweep.sampling_rate_hz
    i0, i1 = int(round(start_s * hz)), int(round(end_s * hz))
    w = sweep.voltage_volts[i0:i1]
    k = int(np.argmin(w) if kind == "min" else np.argmax(w))
    return float(w[k]), (i0 + k) / hz


def detect_spikes(v, rate_hz, start_s=0.0, end_s=None):
    """Spike crossing times (s) and peak voltages (V) within [start_s, end_s): dV/dt upstroke, then peak test."""
    v = np.asarray(v, float)
    vf = filtered(v, rate_hz)
    dvdt = np.gradient(vf) * rate_hz
    i0 = int(round(start_s * rate_hz))
    i1 = len(v) if end_s is None else min(int(round(end_s * rate_hz)), len(v))
    above = dvdt[i0:i1] >= DVDT_THRESHOLD_V_PER_S
    ups = np.flatnonzero(above[1:] & ~above[:-1]) + 1 + i0
    times, peaks = [], []
    next_allowed = i0
    w = int(round(PEAK_SEARCH_S * rate_hz))
    for u in ups:
        if u < next_allowed:
            continue
        seg = v[u:min(u + w, len(v))]
        if len(seg) == 0:
            continue
        k = int(np.argmax(seg))
        vpk = seg[k]
        if vpk > MIN_PEAK_V and vpk - v[u] >= MIN_HEIGHT_V:
            times.append(u / rate_hz)
            peaks.append(float(vpk))
            after = np.flatnonzero(dvdt[u + k:i1] < 0)
            next_allowed = u + k + (after[0] if len(after) else 0)
    return np.array(times), np.array(peaks)


def firing_summary(sweep, start_s, end_s):
    """Spike count, rate and latency in [start_s, end_s), detected before the recording end."""
    valid = sweep.voltage_volts[:int(round(sweep.recording_end_s * sweep.sampling_rate_hz))]
    t, pk = detect_spikes(valid, sweep.sampling_rate_hz, start_s, end_s)
    dur = end_s - start_s
    return {"spike_count": int(len(t)), "rate_hz": len(t) / dur if dur > 0 else np.nan,
            "first_spike_latency_s": float(t[0] - start_s) if len(t) else np.nan, "spike_times_s": t,
            "lowest_spike_peak_v": float(pk.min()) if len(pk) else np.nan}


def fi_slope(amps_pa, rates_hz):
    """Least-squares slope (Hz/pA) of rate against amplitude over sweeps that fired."""
    a = np.asarray(amps_pa, float)
    r = np.asarray(rates_hz, float)
    m = r > 0
    if m.sum() < 2 or len(np.unique(a[m])) < 2:
        return np.nan
    return float(np.polyfit(a[m], r[m], 1)[0])
