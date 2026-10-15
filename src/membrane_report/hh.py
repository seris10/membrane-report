"""Single-compartment squid-axon HH experiments (mV, ms, µA/cm², mS/cm²).

Kinetics multiply by 3**((T-6.3)/10), as in the NEURON hh mechanism.
Reversal potentials and maximal conductances stay fixed during this temperature experiment.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from .synthetic import _hh_rhs


def pulse_intervals(protocol="step", interval_ms=12.0):
    if protocol not in {"step", "pair"}:
        raise ValueError("protocol must be step or pair")
    if not np.isfinite(interval_ms) or not 4 <= interval_ms <= 30:
        raise ValueError("onset interval must be in [4,30] ms")
    return [(50.0, 250.0)] if protocol == "step" else [(50.0, 52.0), (50.0 + interval_ms, 52.0 + interval_ms)]


def simulate_hh(drive=7.0, sodium=120.0, potassium=36.0, *, temperature_c=6.3,
                protocol="step", interval_ms=12.0, sample_ms=0.1):
    """DOP853 integrates each constant-current interval separately; transitions are exact.

    Returns sampled state, ionic currents and upward 0 mV crossings. Pair experiments use two
    2 ms current pulses, with a 4–30 ms onset-to-onset interval. These are simulated responses.
    """
    controls = [drive, sodium, potassium, temperature_c, sample_ms]
    if not np.all(np.isfinite(controls)) or not (0 <= drive <= 20 and 0 <= sodium <= 180
            and 0 <= potassium <= 54 and 0 <= temperature_c <= 25 and 0.01 <= sample_ms <= 1):
        raise ValueError("invalid HH controls or sampling interval")
    intervals = pulse_intervals(protocol, interval_ms)
    edges = sorted({0.0, 300.0, *(x for pulse in intervals for x in pulse)})
    grid = np.linspace(0, 300, int(round(300 / sample_ms)) + 1)
    states = np.empty((len(grid), 4))
    state = [-65.0, 0.0529, 0.5961, 0.3177]
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        current = drive if any(a <= (start + end) / 2 < b for a, b in intervals) else 0.0
        rhs = lambda t, y: _hh_rhs(t, y, lambda _: current, temperature_c=temperature_c,
                                  sodium=sodium, potassium=potassium)
        sol = solve_ivp(rhs, (start, end), state, method="DOP853", dense_output=True,
                        max_step=0.05, rtol=1e-10, atol=1e-12)
        if not sol.success:
            raise RuntimeError("HH integration failed")
        mask = (grid >= start) & (grid <= end)
        states[mask] = sol.sol(grid[mask]).T
        state = sol.y[:, -1]
    v, m, h, n = states.T
    current = np.where(np.any([(grid >= a) & (grid < b) for a, b in intervals], axis=0), drive, 0.0)
    ina = sodium * m ** 3 * h * (v - 50)
    ik = potassium * n ** 4 * (v + 77)
    leak = 0.3 * (v + 54.387)
    crossings = np.flatnonzero((v[:-1] < 0) & (v[1:] >= 0) & (grid[:-1] >= 50) & (grid[:-1] < intervals[-1][1] + (15 if protocol == "pair" else 0)))
    spikes = grid[crossings] + (grid[crossings + 1] - grid[crossings]) * (-v[crossings]) / (v[crossings + 1] - v[crossings])
    if not np.all(np.isfinite(states)) or np.any(states[:, 1:] < 0) or np.any(states[:, 1:] > 1):
        raise RuntimeError("HH integration violates gate bounds")
    return {"t_ms": grid, "state": states, "i": current, "ina": ina, "ik": ik, "leak": leak,
            "dvdt": current - ina - ik - leak, "spikes_ms": spikes, "spike_count": len(spikes),
            "temperature_c": temperature_c, "kinetic_factor": 3 ** ((temperature_c - 6.3) / 10),
            "protocol": protocol, "interval_ms": interval_ms, "intervals_ms": intervals}
