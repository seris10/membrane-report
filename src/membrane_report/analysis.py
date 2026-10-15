"""Per-cell analysis: QC, simple features, passive fits with held-out amplitudes, reference checks."""

from __future__ import annotations

import sys

import numpy as np

from . import passive as pv
from .features import baseline_voltage, fi_slope, firing_summary, peak_deflection
from .qc import sweep_qc
from .recording import read_nwb
from .selection import DEPOLARIZING_CODES, LONG_SQUARE_CODES, SUBTHRESHOLD_CODES

STEADY_WINDOW_S = 0.1
N_TRAIN_AMPLITUDES = 2
PEAK_RIN_MIN_AMP_PA = -100.0


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def peak_deflection_resistance(rows):
    """Least-squares slope (ohm) of peak voltage against command, as IPFX input_resistance."""
    i = np.array([r["stimulus_amplitude_pa"] * 1e-12 for r in rows])
    v = np.array([r["peak_v"] for r in rows])
    if len(i) == 1:
        i = np.r_[i, 0.0]
        v = np.r_[v, rows[0]["baseline_v"]]
    if len(np.unique(i)) < 2:
        return np.nan
    a = np.vstack([i, np.ones_like(i)]).T
    return float(np.linalg.lstsq(a, v, rcond=None)[0][0])


def analyze_cell(nwb_path, asset_id, *, n_boot=200, seed=0, with_references=True) -> dict:
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 2:
        raise ValueError("n_boot must be an integer of at least two")
    cell, sweeps = read_nwb(nwb_path, asset_id, set(LONG_SQUARE_CODES))
    _log(f"cell {cell['cell_id']}: {len(sweeps)} long-square sweeps")
    qc_rows = [sweep_qc(s) for s in sweeps]
    by_id = {s.sweep_id: s for s in sweeps}
    feat_rows = []
    for s, q in zip(sweeps, qc_rows):
        if not np.isfinite(q["stimulus_start_s"]):
            continue
        start, end, amp_pa = q["stimulus_start_s"], q["stimulus_end_s"], q["stimulus_amplitude_pa"]
        end_avail = min(end, q["recording_end_s"])
        valid_end = int(round(q["recording_end_s"] * s.sampling_rate_hz))
        if end_avail <= start or start < 0.1 or not np.all(np.isfinite(s.voltage_volts[:valid_end])):
            continue
        row = {"cell_id": cell["cell_id"], "sweep_id": s.sweep_id, "protocol": s.protocol_name,
               "stimulus_amplitude_pa": amp_pa, "qc_pass": q["qc_pass"], "complete": q["complete"],
               "baseline_v": baseline_voltage(s, start)}
        kind = "min" if amp_pa < 0 else "max"
        row["peak_v"], row["peak_time_s"] = peak_deflection(s, start, end_avail, kind)
        hz = s.sampling_rate_hz
        if q["complete"]:
            row["steady_v"] = float(np.mean(s.voltage_volts[int(round((end - STEADY_WINDOW_S) * hz)):int(round(end * hz))]))
        else:
            row["steady_v"] = np.nan
        fs = firing_summary(s, start, end_avail)
        row.update({k: fs[k] for k in ("spike_count", "rate_hz", "first_spike_latency_s", "lowest_spike_peak_v")})
        row["spike_times_from_onset_s"] = (fs["spike_times_s"] - start).tolist()
        row["sag_ratio"] = pv.sag_ratio(s, (start, end, amp_pa * 1e-12)) if (amp_pa < 0 and q["complete"]) else np.nan
        md = s.recording_metadata
        row.update({k: md.get(k) for k in ("voltage_conversion", "voltage_offset", "current_conversion",
                                            "current_offset", "bias_current_as_stored", "bridge_balance_as_stored")})
        row["source_flags"] = "; ".join(s.source_quality_flags)
        feat_rows.append(row)

    passive_rows = []
    candidates = []
    for r in feat_rows:
        reason = None
        if r["protocol"] not in SUBTHRESHOLD_CODES:
            reason = "not a subthreshold protocol sweep"
        elif not r["qc_pass"]:
            q = next(x for x in qc_rows if x["sweep_id"] == r["sweep_id"])
            reason = "failed QC: " + q["qc_reasons"]
        elif r["stimulus_amplitude_pa"] >= 0:
            reason = "depolarizing step"
        elif r["spike_count"] > 0:
            reason = "spikes during the step"
        if reason is None:
            candidates.append(r)
        else:
            passive_rows.append({"cell_id": cell["cell_id"], "sweep_id": r["sweep_id"],
                                 "stimulus_amplitude_pa": r["stimulus_amplitude_pa"], "role": "excluded",
                                 "reason": reason})
    amps = sorted({round(r["stimulus_amplitude_pa"], 1) for r in candidates}, key=abs)
    train_amps = set(amps[:N_TRAIN_AMPLITUDES])
    steps, roles = [], []
    for r in candidates:
        s = by_id[r["sweep_id"]]
        q = next(x for x in qc_rows if x["sweep_id"] == r["sweep_id"])
        stim = (q["stimulus_start_s"], q["stimulus_end_s"], q["stimulus_amplitude_pa"] * 1e-12)
        steps.append(pv.prepare_step(s, stim, q["recording_end_s"]))
        roles.append("train" if round(r["stimulus_amplitude_pa"], 1) in train_amps else "test")

    result = {"cell": cell, "qc": qc_rows, "features": feat_rows, "passive_sweeps": passive_rows,
              "train_amplitudes_pa": sorted(train_amps), "n_candidate_sweeps": len(steps), "_sweeps": by_id}
    train = [st for st, ro in zip(steps, roles) if ro == "train"]
    if len(train) == 0 or len(amps) < N_TRAIN_AMPLITUDES + 1:
        result["passive_status"] = (f"not estimated: {len(amps)} usable subthreshold amplitudes "
                                    f"(need {N_TRAIN_AMPLITUDES + 1} for a held-out check)")
        _log(result["passive_status"])
    else:
        _log("passive fits")
        fit_train = pv.fit_steps(train)
        fit_all = pv.fit_steps(steps)
        boot_r, boot_tau = pv.block_bootstrap(train, fit_train, n_boot=n_boot, seed=seed)
        for st, ro in zip(steps, roles):
            single = pv.fit_steps([st])
            err = pv.step_errors(st, fit_train)
            fr = next(r for r in feat_rows if r["sweep_id"] == st.sweep_id)
            passive_rows.append({"cell_id": cell["cell_id"], "sweep_id": st.sweep_id,
                                 "stimulus_amplitude_pa": st.delta_i * 1e12, "role": ro, "reason": "",
                                 "sag_ratio": fr["sag_ratio"],
                                 "single_sweep_r_in_mohm": single.r_in_ohm / 1e6,
                                 "single_sweep_tau_ms": single.tau_s * 1e3,
                                 "single_sweep_status": single.status,
                                 "train_model_rms_error_mv": err["rms_error_v"] * 1e3,
                                 "train_model_max_abs_error_mv": err["max_abs_error_v"] * 1e3,
                                 "observed_steady_deflection_mv": err["observed_steady_deflection_v"] * 1e3,
                                 "predicted_steady_deflection_mv": err["predicted_steady_deflection_v"] * 1e3,
                                 "relative_rms_error": err["relative_rms_error"]})
        ode_t = train[0].t
        analytic = pv.step_response(ode_t, train[0].t_on, train[0].t_off, train[0].delta_i, fit_train.r_in_ohm,
                                    fit_train.tau_s, train[0].v0)
        numeric = pv.simulate_ode(ode_t, train[0].t_on, train[0].t_off, train[0].delta_i, fit_train.r_in_ohm,
                                  fit_train.tau_s, train[0].v0)

        def fit_dict(f, label):
            return {"label": label, "r_in_mohm": f.r_in_ohm / 1e6, "tau_ms": f.tau_s * 1e3,
                    "c_eff_pf": f.c_eff_f * 1e12, "rms_residual_mv": f.rms_residual_v * 1e3, "n_bins": f.n_bins,
                    "naive_se_r_in_mohm": f.naive_se_r_ohm / 1e6, "naive_se_tau_ms": f.naive_se_tau_s * 1e3,
                    "status": f.status}

        def pct(x, p):
            return float(np.quantile(x, p))

        result.update({
            "passive_status": "estimated",
            "fit_train": fit_dict(fit_train, "train amplitudes"),
            "fit_all": fit_dict(fit_all, "all usable amplitudes"),
            "bootstrap": {"method": "moving-block bootstrap of residuals within sweeps, 50 ms blocks",
                          "n_boot": n_boot,
                          "r_in_mohm_95": [pct(boot_r, 0.025) / 1e6, pct(boot_r, 0.975) / 1e6],
                          "tau_ms_95": [pct(boot_tau, 0.025) * 1e3, pct(boot_tau, 0.975) * 1e3],
                          "c_eff_pf_95": [pct(boot_tau / boot_r, 0.025) * 1e12, pct(boot_tau / boot_r, 0.975) * 1e12],
                          "estimates_inside_intervals": bool(
                              pct(boot_r, 0.025) <= fit_train.r_in_ohm <= pct(boot_r, 0.975)
                              and pct(boot_tau, 0.025) <= fit_train.tau_s <= pct(boot_tau, 0.975)),
                          "scope": "within-sweep noise of the training sweeps only"},
            "ode_check": {"max_abs_difference_v": float(np.max(np.abs(analytic - numeric))),
                          "solver": "scipy solve_ivp DOP853, rtol 1e-10, atol 1e-14, piecewise per current level"},
            "_steps": steps, "_roles": roles, "_fit_train": fit_train,
        })

    hyper = [r for r in feat_rows if r["qc_pass"] and r["protocol"] in SUBTHRESHOLD_CODES
             and PEAK_RIN_MIN_AMP_PA < r["stimulus_amplitude_pa"] < 0 and r["spike_count"] == 0]
    result["peak_deflection_r_in_mohm"] = (peak_deflection_resistance(hyper) / 1e6) if hyper else np.nan
    result["peak_deflection_sweeps"] = [r["sweep_id"] for r in hyper]
    dep = [r for r in feat_rows if r["qc_pass"] and r["protocol"] in DEPOLARIZING_CODES]
    fired = [r for r in dep if r["spike_count"] > 0]
    result["rheobase_pa"] = min((r["stimulus_amplitude_pa"] for r in fired), default=np.nan)
    result["fi_slope_hz_per_pa"] = fi_slope([r["stimulus_amplitude_pa"] for r in dep], [r["rate_hz"] for r in dep])
    result["baseline_v_median"] = float(np.median([r["baseline_v"] for r in feat_rows if r["qc_pass"]])) if any(
        r["qc_pass"] for r in feat_rows) else np.nan

    if with_references:
        from .references import efel_spike_counts, ipfx_long_square

        _log("IPFX and eFEL references")
        passing = [r["sweep_id"] for r in feat_rows if r["qc_pass"]]
        try:
            result["ipfx"] = ipfx_long_square(nwb_path, passing)
        except Exception as exc:  # reported as NOT TESTED
            result["ipfx"] = {"error": f"{type(exc).__name__}: {exc}"}
        dep_sweeps = []
        for r in dep:
            q = next(x for x in qc_rows if x["sweep_id"] == r["sweep_id"])
            dep_sweeps.append((by_id[r["sweep_id"]], (q["stimulus_start_s"], q["stimulus_end_s"], 0.0)))
        for key, thr in (("efel_spike_counts", None), ("efel_spike_counts_0mv", 0.0)):
            try:
                result[key] = efel_spike_counts(dep_sweeps, threshold_mv=thr) if dep_sweeps else {}
            except Exception as exc:
                result[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return result
