"""Synthetic passive and Hodgkin-Huxley sweeps with known answers, and the checks that use them."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from . import passive as pv
from .recording import Sweep, sweep_from_arrays

RATE_HZ = 50_000.0
TEST_PULSE = (0.045, 0.055, 50e-12)
STEP = (0.6, 1.6)
DURATION_S = 2.75


def command(t, amp, test_pulse=True, step=STEP):
    i = np.zeros_like(t)
    if test_pulse:
        a, b, ap = TEST_PULSE
        i[(t >= a) & (t < b)] = ap
    i[(t >= step[0]) & (t < step[1])] = amp
    return i


def passive_voltage(t, i, r_in, tau, v0):
    """Exact solution of C dV/dt = -(V - v0)/R + I(t) for piecewise-constant I sampled at t."""
    v = np.empty_like(t)
    v[0] = v0
    dt = t[1] - t[0]
    decay = np.exp(-dt / tau)
    for k in range(1, len(t)):
        target = v0 + r_in * i[k - 1]
        v[k] = target + (v[k - 1] - target) * decay
    return v


def passive_sweep(amp, r_in=150e6, tau=0.012, v0=-0.070, noise_sd=0.0, seed=0, stop_at_s=None, sweep_id=1,
                  rate=RATE_HZ):
    t = np.arange(int(DURATION_S * rate)) / rate
    i = command(t, amp)
    v = passive_voltage(t, i, r_in, tau, v0)
    if noise_sd:
        v = v + np.random.default_rng(seed).normal(0.0, noise_sd, len(v))
    if stop_at_s is not None:
        v[int(stop_at_s * rate):] = 0.0
    return Sweep(cell_id="synthetic/passive", sweep_id=sweep_id, source_asset_id="synthetic",
                 protocol_name="X1PS_SubThresh" if amp < 0 else "X4PS_SupraThresh", sampling_rate_hz=rate,
                 voltage_volts=v, injected_current_amperes=i)


def _hh_rhs(t_ms, y, j_of_t, *, temperature_c=6.3, sodium=120.0, potassium=36.0):
    v, m, h, n = y
    am = 0.1 * (v + 40) / -np.expm1(-(v + 40) / 10) if abs(v + 40) > 1e-9 else 1.0
    bm = 4.0 * np.exp(-(v + 65) / 18)
    ah = 0.07 * np.exp(-(v + 65) / 20)
    bh = 1.0 / (1 + np.exp(-(v + 35) / 10))
    an = 0.01 * (v + 55) / -np.expm1(-(v + 55) / 10) if abs(v + 55) > 1e-9 else 0.1
    bn = 0.125 * np.exp(-(v + 65) / 80)
    i_ion = sodium * m ** 3 * h * (v - 50.0) + potassium * n ** 4 * (v + 77.0) + 0.3 * (v + 54.387)
    phi = 3.0 ** ((temperature_c - 6.3) / 10.0)
    return [j_of_t(t_ms) - i_ion, phi * (am * (1 - m) - bm * m),
            phi * (ah * (1 - h) - bh * h), phi * (an * (1 - n) - bn * n)]


def hh_sweep(j_ua_per_cm2, area_cm2=1e-5, rate=RATE_HZ, sweep_id=1):
    """Hodgkin-Huxley sweep for a 1 s step of ``j_ua_per_cm2``; returns (sweep, 0 mV crossings in the step)."""
    t_s = np.arange(int(DURATION_S * rate)) / rate
    t_ms = t_s * 1e3

    def j_of_t(tm):
        return j_ua_per_cm2 if STEP[0] * 1e3 <= tm < STEP[1] * 1e3 else 0.0

    y0 = [-65.0, 0.0529, 0.5961, 0.3177]
    sol = solve_ivp(_hh_rhs, (0, t_ms[-1]), y0, t_eval=t_ms, args=(j_of_t,), method="LSODA", max_step=0.02,
                    rtol=1e-7, atol=1e-9)
    v_mv = sol.y[0]
    in_step = (t_s >= STEP[0]) & (t_s < STEP[1])
    up = (v_mv[1:] >= 0) & (v_mv[:-1] < 0) & in_step[1:]
    i_amp = command(t_s, j_ua_per_cm2 * area_cm2 * 1e-6, test_pulse=False)
    sweep = sweep_from_arrays(t_s, v_mv, i_amp * 1e12, v_unit="mV", i_unit="pA", cell_id="synthetic/hh",
                              sweep_id=sweep_id, source_asset_id="synthetic", protocol_name="X4PS_SupraThresh")
    return sweep, int(up.sum())


def validate_synthetic(verbose=False) -> bool:
    """Run the known-answer checks; True when all pass."""
    checks = synthetic_checks()
    if verbose:
        for c in checks:
            print(f"{'PASS' if c['passed'] else 'FAIL'}  {c['check']}: {c['detail']}")
    return all(c["passed"] for c in checks)


def synthetic_checks() -> list[dict]:
    """Known-answer checks as dicts of check name, pass flag and detail."""
    from .epochs import long_square_stimulus
    from .features import detect_spikes
    from .qc import sweep_qc

    results = []

    def check(name, ok, detail):
        results.append({"check": name, "passed": bool(ok), "detail": detail, "data_kind": "synthetic"})

    r_true, tau_true, v0 = 150e6, 0.012, -0.070
    sweeps = [passive_sweep(a, r_true, tau_true, v0, sweep_id=k) for k, a in enumerate((-30e-12, -50e-12, -90e-12))]
    steps = []
    for s in sweeps:
        stim = long_square_stimulus(s.stimulus_intervals)
        steps.append(pv.prepare_step(s, stim, s.recording_end_s))
    fit = pv.fit_steps(steps)
    err = max(abs(fit.r_in_ohm / r_true - 1), abs(fit.tau_s / tau_true - 1))
    check("noiseless passive recovery", err < 1e-8, f"max relative error {err:.2e} (1 ms bin means)")
    noisy = [passive_sweep(a, r_true, tau_true, v0, noise_sd=0.5e-3, seed=k, sweep_id=k)
             for k, a in enumerate((-30e-12, -50e-12))]
    nsteps = [pv.prepare_step(s, long_square_stimulus(s.stimulus_intervals), s.recording_end_s) for s in noisy]
    nfit = pv.fit_steps(nsteps)
    nerr = max(abs(nfit.r_in_ohm / r_true - 1), abs(nfit.tau_s / tau_true - 1))
    check("noisy passive recovery (0.5 mV noise)", nerr < 0.05, f"max relative error {nerr:.2e}")
    t = steps[0].t
    a = pv.step_response(t, 0.6, 1.6, -50e-12, r_true, tau_true, v0)
    b = pv.simulate_ode(t, 0.6, 1.6, -50e-12, r_true, tau_true, v0)
    check("analytic vs numerical integration", np.max(np.abs(a - b)) < 1e-9, f"max |diff| {np.max(np.abs(a - b)):.1e} V")
    mv = sweep_from_arrays(sweeps[1].time_seconds, sweeps[1].voltage_volts * 1e3,
                           sweeps[1].injected_current_amperes * 1e12, v_unit="mV", i_unit="pA",
                           cell_id="synthetic/units", sweep_id=9, source_asset_id="synthetic",
                           protocol_name="X1PS_SubThresh")
    f_si = pv.fit_steps([steps[1]])
    f_mv = pv.fit_steps([pv.prepare_step(mv, long_square_stimulus(mv.stimulus_intervals), mv.recording_end_s)])
    uerr = abs(f_mv.r_in_ohm / f_si.r_in_ohm - 1) + abs(f_mv.tau_s / f_si.tau_s - 1)
    check("V/A versus mV/pA input", uerr < 1e-9, f"relative difference {uerr:.1e}")
    cut = passive_sweep(-50e-12, stop_at_s=0.8)
    q = sweep_qc(cut)
    check("aborted sweep detected", (not q["complete"]) and abs(q["recording_end_s"] - 0.8) < 1e-4,
          f"recording_end_s {q['recording_end_s']:.4f}, complete {q['complete']}")
    zero = passive_sweep(0.0)
    zq = sweep_qc(zero)
    check("zero-current sweep has no step", (not zq["qc_pass"]) and "no single long step" in zq["qc_reasons"],
          zq["qc_reasons"])
    for j in (2.0, 7.0, 15.0):
        hs, n_true = hh_sweep(j)
        times, _ = detect_spikes(hs.voltage_volts, hs.sampling_rate_hz, STEP[0], STEP[1])
        check(f"Hodgkin-Huxley step {j} uA/cm2 spike count", len(times) == n_true,
              f"detected {len(times)}, simulation {n_true}")
    return results
