"""Single-compartment passive model C dV/dt = -(V - V0)/R + dI(t): solution, fitting, bootstrap, sag."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

BIN_S = 0.001
PRE_S = 0.1
POST_S = 0.3
SKIP_S = 0.001
R_BOUNDS_OHM = (1e6, 1e10)
TAU_BOUNDS_S = (2e-4, 1.0)
SAG_WINDOW_S = 0.005
SAG_STEADY_S = 0.03
SAG_FLAG = 0.10


def bin_mean_factor(tau, n_samples, dt):
    """Bin mean of exp(-(t - c)/tau) relative to its value at the bin center c."""
    if n_samples <= 1:
        return 1.0
    return float(np.sinh(n_samples * dt / (2 * tau)) / (n_samples * np.sinh(dt / (2 * tau))))


def step_response(t, t_on, t_off, delta_i, r_in, tau, v0, n_samples=1, dt=0.0):
    """Analytic passive response at ``t``, or its bin mean when ``n_samples`` > 1."""
    t = np.asarray(t, float)
    k = bin_mean_factor(tau, n_samples, dt)
    out = np.full_like(t, v0, dtype=float)
    during = (t >= t_on) & (t < t_off)
    after = t >= t_off
    out[during] = v0 + r_in * delta_i * (1.0 - k * np.exp(-(t[during] - t_on) / tau))
    v_off = r_in * delta_i * (1.0 - np.exp(-(t_off - t_on) / tau))
    out[after] = v0 + v_off * k * np.exp(-(t[after] - t_off) / tau)
    return out


def simulate_ode(t, t_on, t_off, delta_i, r_in, tau, v0, rtol=1e-10, atol=1e-14):
    """Integrate the passive ODE numerically, one piece per constant-current interval."""
    t = np.asarray(t, float)
    c = tau / r_in
    out = np.empty_like(t)
    edges = [t[0], t_on, t_off, t[-1]]
    v_start = v0
    for k, current in enumerate((0.0, delta_i, 0.0)):
        a, b = edges[k], edges[k + 1]
        last = k == 2
        sel = (t >= a) & ((t <= b) if last else (t < b))
        if b <= a:
            continue

        def rhs(_, v, current=current):
            return [(-(v[0] - v0) / r_in + current) / c]

        pts = t[sel]
        t_eval = pts if (len(pts) and pts[-1] == b) else np.r_[pts, b]
        sol = solve_ivp(rhs, (a, b), [v_start], method="DOP853", rtol=rtol, atol=atol, t_eval=t_eval)
        if len(pts):
            out[sel] = sol.y[0][:len(pts)]
        v_start = sol.y[0][-1]
    return out


@dataclass
class StepData:
    sweep_id: int
    delta_i: float
    t_on: float
    t_off: float
    v0: float
    t: np.ndarray
    v: np.ndarray
    fit_mask: np.ndarray
    n_per_bin: int = 1
    dt: float = 0.0


def prepare_step(sweep, stim, rec_end_s, bin_s=BIN_S):
    """Bin one subthreshold step for fitting; masks bins that straddle a current transition."""
    t_on, t_off, amp = stim
    hz = sweep.sampling_rate_hz
    t_end = min(t_off + POST_S, rec_end_s)
    i0 = int(round((t_on - PRE_S) * hz))
    i1 = int(round(t_end * hz))
    if not np.isfinite(bin_s) or bin_s <= 0:
        raise ValueError("bin_s must be positive and finite")
    per = int(round(bin_s * hz))
    if per < 1:
        raise ValueError("bin_s must span at least one sample")
    if i0 < 0 or i1 > len(sweep.voltage_volts) or i1 <= i0:
        raise ValueError("step needs a complete pre-step baseline and a valid recording window")
    n = (i1 - i0) // per
    v = sweep.voltage_volts[i0:i0 + n * per].reshape(n, per).mean(axis=1)
    first = (i0 + per * np.arange(n)) / hz
    t = first + (per - 1) / (2 * hz)
    v0 = float(np.mean(sweep.voltage_volts[int(round((t_on - PRE_S) * hz)):int(round(t_on * hz))]))
    mask = first >= t_on
    for edge in (t_on, t_off):
        mask &= ~((first < edge + SKIP_S) & (first + per / hz > edge))
    return StepData(sweep.sweep_id, float(amp), float(t_on), float(t_off), v0, t, v, mask, per, 1.0 / hz)


def _residuals(x, steps):
    r_in, tau = np.exp(x)
    return np.concatenate([(step_response(s.t[s.fit_mask], s.t_on, s.t_off, s.delta_i, r_in, tau, s.v0,
                                          s.n_per_bin, s.dt) - s.v[s.fit_mask]) for s in steps])


@dataclass
class PassiveFit:
    r_in_ohm: float
    tau_s: float
    rss_v2: float
    n_bins: int
    rms_residual_v: float
    naive_se_r_ohm: float
    naive_se_tau_s: float
    status: str

    @property
    def c_eff_f(self):
        return self.tau_s / self.r_in_ohm


def fit_steps(steps) -> PassiveFit:
    """Shared R and tau for one or more steps, from a small grid of starts."""
    if not steps:
        raise ValueError("no steps to fit")
    for s in steps:
        if s.t.ndim != 1 or s.t.shape != s.v.shape or s.fit_mask.shape != s.t.shape or not (
                np.all(np.isfinite(s.t)) and np.all(np.isfinite(s.v)) and np.all(np.diff(s.t) > 0)):
            raise ValueError("step times and voltages must be finite, aligned and strictly increasing")
        if s.fit_mask.sum() < 3 or not np.isfinite(s.delta_i) or s.delta_i == 0:
            raise ValueError("passive fitting needs at least three bins and a nonzero finite current step")
    lo = np.log([R_BOUNDS_OHM[0], TAU_BOUNDS_S[0]])
    hi = np.log([R_BOUNDS_OHM[1], TAU_BOUNDS_S[1]])
    best = None
    for r0 in (5e7, 2e8, 8e8):
        for tau0 in (0.005, 0.02, 0.08):
            res = least_squares(_residuals, np.log([r0, tau0]), args=(steps,), bounds=(lo, hi),
                                x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if res.success and np.all(np.isfinite(res.fun)) and (best is None or res.cost < best.cost):
                best = res
    if best is None:
        raise ValueError("passive fitting failed: no successful optimizer start")
    r_in, tau = np.exp(best.x)
    n = len(best.fun)
    rss = float(2 * best.cost)
    sigma2 = rss / max(n - 2, 1)
    try:
        cov_log = sigma2 * np.linalg.inv(best.jac.T @ best.jac)
        se_log = np.sqrt(np.clip(np.diag(cov_log), 0, np.inf))
    except np.linalg.LinAlgError:
        se_log = np.array([np.nan, np.nan])
    at_bound = np.any(np.isclose(best.x, lo, atol=1e-6, rtol=0)) or np.any(np.isclose(best.x, hi, atol=1e-6, rtol=0))
    return PassiveFit(float(r_in), float(tau), rss, n, float(np.sqrt(rss / n)),
                      float(r_in * se_log[0]), float(tau * se_log[1]),
                      "converged_at_bound" if at_bound else ("converged" if best.success else "failed"))


def predict_step(step: StepData, fit: PassiveFit):
    return step_response(step.t, step.t_on, step.t_off, step.delta_i, fit.r_in_ohm, fit.tau_s, step.v0,
                         step.n_per_bin, step.dt)


def step_errors(step: StepData, fit: PassiveFit) -> dict:
    """Prediction error of a fitted model on one step, in volts and relative to the step size."""
    pred = predict_step(step, fit)
    m = step.fit_mask
    resid = step.v[m] - pred[m]
    ss = (step.t >= step.t_off - 0.1) & (step.t < step.t_off)
    obs_ss = float(np.mean(step.v[ss]) - step.v0)
    pred_ss = float(np.mean(pred[ss]) - step.v0)
    return {"rms_error_v": float(np.sqrt(np.mean(resid ** 2))),
            "max_abs_error_v": float(np.max(np.abs(resid))),
            "observed_steady_deflection_v": obs_ss, "predicted_steady_deflection_v": pred_ss,
            "relative_rms_error": float(np.sqrt(np.mean(resid ** 2)) / abs(obs_ss)) if obs_ss else np.nan}


def block_bootstrap(steps, fit: PassiveFit, n_boot=200, block_s=0.05, seed=0):
    """Moving-block bootstrap of residuals within each step; returns refitted R and tau arrays."""
    rng = np.random.default_rng(seed)
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 2:
        raise ValueError("n_boot must be an integer of at least two")
    if not np.isfinite(block_s) or block_s <= 0:
        raise ValueError("block_s must be positive and finite")
    fitted = [predict_step(s, fit) for s in steps]
    resid = [s.v - f for s, f in zip(steps, fitted)]
    rs, taus = [], []
    for _ in range(n_boot):
        new = []
        for s, f, r in zip(steps, fitted, resid):
            idx = np.flatnonzero(s.fit_mask)
            rr = r[idx]
            n = len(rr)
            bin_width = s.n_per_bin * s.dt if s.dt > 0 else float(np.median(np.diff(s.t)))
            block = min(max(int(round(block_s / bin_width)), 1), n)
            starts = rng.integers(0, max(n - block + 1, 1), size=int(np.ceil(n / block)))
            boot = np.concatenate([rr[a:a + block] for a in starts])[:n]
            v = s.v.copy()
            v[idx] = f[idx] + boot
            new.append(StepData(s.sweep_id, s.delta_i, s.t_on, s.t_off, s.v0, s.t, v, s.fit_mask, s.n_per_bin,
                                s.dt))
        bf = fit_steps(new)
        rs.append(bf.r_in_ohm)
        taus.append(bf.tau_s)
    return np.array(rs), np.array(taus)


def sag_ratio(sweep, stim):
    """(V_peak - V_steady) / (V_peak - V_baseline) for a hyperpolarizing step, as in IPFX 2.1."""
    t_on, t_off, _ = stim
    hz = sweep.sampling_rate_hz
    v = sweep.voltage_volts
    i0, i1 = int(round(t_on * hz)), int(round(t_off * hz))
    k = i0 + int(np.argmin(v[i0:i1]))
    h = int(round(SAG_WINDOW_S / 2 * hz))
    v_peak = float(np.mean(v[k - h:k + h]))
    v_ss = float(np.mean(v[i1 - int(round(SAG_STEADY_S * hz)):i1]))
    v_base = float(np.mean(v[i0 - int(round(SAG_STEADY_S * hz)):i0]))
    return (v_peak - v_ss) / (v_peak - v_base)
