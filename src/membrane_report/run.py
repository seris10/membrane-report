"""Analyze the configured cells and write the report tables, comparisons and manifest."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .analysis import analyze_cell
from .dandi import CITATION, DANDISET, DOI, LICENSE, VERSION, sha256_file

TOLERANCES = {
    "baseline_mv": 0.01,
    "sag": 1e-3,
    "qc_feature_mv": 1e-3,
    "peak_r_in_relative": 1e-3,
    "fi_slope_relative": 1e-6,
}
TOLERANCE_NOTE = ("Tolerances were set after the first cell's comparison and before the other cells were analyzed; "
                  "the definitions are aligned, so agreement is expected to the precision of the stored float32 data.")


def load_config(path) -> dict:
    with open(path, "rb") as fh:
        cfg = json.load(fh) if str(path).endswith(".json") else tomllib.load(fh)
    cfg.setdefault("analysis", {}).setdefault("n_boot", 200)
    cfg["analysis"].setdefault("seed", 20261013)
    n_boot = cfg["analysis"]["n_boot"]
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 2:
        raise ValueError("analysis.n_boot must be an integer of at least two")
    if not cfg.get("cells"):
        raise ValueError("configuration must select at least one cell")
    return cfg


def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def compare_with_references(res) -> list[dict]:
    """One row per checked quantity: this package, the reference, difference, tolerance, verdict."""
    rows = []
    cid = res["cell"]["cell_id"]
    ip = res.get("ipfx", {})
    ef = res.get("efel_spike_counts", {})

    def add(quantity, mine, ref, ref_name, tol=None, kind="abs", note=""):
        mine_f, ref_f = _float(mine), _float(ref)
        if not (np.isfinite(mine_f) and np.isfinite(ref_f)):
            verdict, diff = "NOT TESTED", np.nan
        else:
            diff = (abs(mine_f - ref_f) if kind == "abs" else
                    (abs(mine_f - ref_f) / abs(ref_f) if ref_f else (0.0 if mine_f == 0 else np.inf)))
            verdict = "informational" if tol is None else ("SUPPORTED" if diff <= tol else "CONTRADICTED")
        rows.append({"cell_id": cid, "quantity": quantity, "this_package": mine_f, "reference": ref_f,
                     "reference_name": ref_name, "difference": diff, "difference_kind": kind,
                     "tolerance": tol, "verdict": verdict, "note": note})

    if "error" in ip:
        rows.append({"cell_id": cid, "quantity": "IPFX comparison", "verdict": "NOT TESTED", "note": ip["error"]})
        ip_sweeps = {}
    else:
        ip_sweeps = {s["sweep_id"]: s for s in ip.get("sweeps", [])}
        if not ip_sweeps:
            rows.append({"cell_id": cid, "quantity": "IPFX comparison", "verdict": "NOT TESTED",
                         "note": "no IPFX reference sweeps were supplied"})
    feats = {f["sweep_id"]: f for f in res["features"]}
    qcs = {q["sweep_id"]: q for q in res["qc"]}
    worst = {}
    for sid, s in ip_sweeps.items():
        f, q = feats[sid], qcs[sid]
        worst.setdefault("baseline", []).append(abs(f["baseline_v"] * 1e3 - s["v_baseline_mv"]))
        if np.isfinite(f["sag_ratio"]) and np.isfinite(s["sag"]):
            worst.setdefault("sag", []).append(abs(f["sag_ratio"] - s["sag"]))
        for k in ("pre_noise_rms_mv", "slow_noise_rms_mv", "post_noise_rms_mv", "vm_delta_mv"):
            if s.get(k) is not None and np.isfinite(q[k]):
                worst.setdefault(k, []).append(abs(q[k] - s[k]))
        worst.setdefault("spike_count_mismatch_ipfx", []).append(int(f["spike_count"] != s["spike_count"]))
        mine_t = np.array(f["spike_times_from_onset_s"])
        ref_t = np.array(s["spike_threshold_times_from_onset_s"])
        if len(mine_t) and len(mine_t) == len(ref_t):
            worst.setdefault("spike_time_offset_ms", []).extend(list((mine_t - ref_t) * 1e3))
    if "baseline" in worst:
        add("baseline voltage, max |difference| over sweeps (mV)", max(worst["baseline"]), 0.0, "IPFX v_baseline",
            TOLERANCES["baseline_mv"])
    if "sag" in worst:
        add("sag ratio, max |difference| over hyperpolarizing sweeps", max(worst["sag"]), 0.0, "IPFX sag",
            TOLERANCES["sag"])
    for k in ("pre_noise_rms_mv", "slow_noise_rms_mv", "post_noise_rms_mv", "vm_delta_mv"):
        if k in worst:
            add(f"QC {k}, max |difference| over sweeps", max(worst[k]), 0.0, "IPFX sweep_qc_features",
                TOLERANCES["qc_feature_mv"])
    if "spike_count_mismatch_ipfx" in worst:
        add("sweeps whose spike count differs from IPFX", sum(worst["spike_count_mismatch_ipfx"]), 0.0,
            "IPFX SpikeFeatureExtractor", 0)
    ef0 = res.get("efel_spike_counts_0mv", {})
    if isinstance(ef, dict) and "error" not in ef and ef:
        bad = {sid: n for sid, n in ef.items() if feats[sid]["spike_count"] != n}
        note = ("eFEL default: peaks above -20 mV. "
                + ("; ".join(f"sweep {sid}: eFEL {n}, this package {feats[sid]['spike_count']}" for sid, n in bad.items())
                   if bad else "all counts equal"))
        add("depolarizing sweeps whose spike count differs from eFEL at its default threshold", len(bad), 0.0,
            "eFEL spike_count_stimint, Threshold -20 mV", None, note=note)
    elif isinstance(ef, dict) and "error" in ef:
        rows.append({"cell_id": cid, "quantity": "eFEL comparison", "verdict": "NOT TESTED", "note": ef["error"]})
    if isinstance(ef0, dict) and "error" not in ef0 and ef0:
        mism = sum(int(feats[sid]["spike_count"] != n) for sid, n in ef0.items())
        peaks = [feats[sid]["lowest_spike_peak_v"] for sid in ef0
                 if np.isfinite(feats[sid].get("lowest_spike_peak_v", np.nan))]
        low = f"; lowest spike peak this package detected in the cell: {min(peaks) * 1e3:.1f} mV" if peaks else ""
        add("depolarizing sweeps whose spike count differs from eFEL with a 0 mV peak threshold", mism, 0.0,
            "eFEL spike_count_stimint, Threshold 0 mV", 0,
            note="threshold moved above the interspike voltage, below the spike peaks" + low)
    if worst.get("spike_time_offset_ms"):
        off = np.array(worst["spike_time_offset_ms"])
        add("spike time: this package's dV/dt crossing minus IPFX threshold time, median (ms)", float(np.median(off)),
            0.0, "IPFX threshold_t", None,
            note=f"{len(off)} matched spikes; range {off.min():.3f} to {off.max():.3f} ms. IPFX places threshold where "
                 "dV/dt reaches 5% of the spike's maximum upstroke, so a small constant offset is expected.")
    if ip.get("cell"):
        c = ip["cell"]
        same_sweeps = sorted(c.get("input_resistance_sweeps", [])) == sorted(res["peak_deflection_sweeps"])
        add("peak-deflection input resistance (Mohm)", res["peak_deflection_r_in_mohm"], c["input_resistance"],
            "IPFX input_resistance", TOLERANCES["peak_r_in_relative"], kind="rel",
            note="same sweeps" if same_sweeps else "sweep sets differ")
        add("rheobase (pA)", res["rheobase_pa"], c["rheobase_i"], "IPFX rheobase_i", 0.5)
        add("f-I slope (Hz/pA)", res["fi_slope_hz_per_pa"], c["fi_fit_slope"], "IPFX fi_fit_slope",
            TOLERANCES["fi_slope_relative"], kind="rel")
        if res.get("passive_status") == "estimated":
            add("membrane time constant (ms): passive fit vs IPFX", res["fit_train"]["tau_ms"],
                (c["tau"] or np.nan) * 1e3, "IPFX tau", None, kind="rel",
                note="different definitions: whole step and relaxation versus 10% of peak to peak; not a pass/fail")
            add("input resistance (Mohm): passive steady-state fit vs IPFX peak-deflection", res["fit_train"]["r_in_mohm"],
                c["input_resistance"], "IPFX input_resistance", None, kind="rel",
                note="different definitions: the passive fit uses small steps and the whole response; IPFX the peak "
                     "of steps down to -90 pA")
    return rows


def interpret(res) -> list[str]:
    """Summarize recorded features and predefined amplitude-comparison criteria."""
    out = []
    qc = res["qc"]
    n_fail = sum(not q["qc_pass"] for q in qc)
    if n_fail:
        reasons = {}
        for q in qc:
            if not q["qc_pass"]:
                for part in q["qc_reasons"].split("; "):
                    key = "incomplete recording" if part.startswith("recording stops") else part.split(" ")[0]
                    reasons[key] = reasons.get(key, 0) + 1
        out.append(f"QC excluded {n_fail}/{len(qc)} long-square sweeps. Criterion counts (a sweep may meet "
                   "multiple exclusion criteria): " + "; ".join(f"{k}, n = {v}" for k, v in reasons.items()) + ".")
    else:
        out.append(f"All {len(qc)} long-square sweeps satisfied the QC criteria.")
    if res.get("passive_status") != "estimated":
        out.append(f"Passive-model estimation status: {res.get('passive_status')}.")
        return out
    ft, bs = res["fit_train"], res["bootstrap"]
    out.append(f"The fit to {', '.join(f'{a:g}' for a in res['train_amplitudes_pa'])} pA steps yields apparent "
               f"input resistance R = {ft['r_in_mohm']:.1f} MΩ, time constant τ = {ft['tau_ms']:.2f} ms and "
               f"effective capacitance τ/R = {ft['c_eff_pf']:.0f} pF. The within-sweep block-bootstrap 95% "
               f"percentile interval for R is {bs['r_in_mohm_95'][0]:.1f}–{bs['r_in_mohm_95'][1]:.1f} MΩ. "
               f"The independent-bin standard-error approximation is {ft['naive_se_r_in_mohm']:.2f} MΩ.")
    if not bs["estimates_inside_intervals"]:
        out.append("The point estimate falls outside its bootstrap percentile interval. This interval-consistency "
                   "flag requires assessment of residual structure and bootstrap coverage before interpreting the interval.")
    ps = [r for r in res["passive_sweeps"] if r["role"] in ("train", "test")]
    test = [r for r in ps if r["role"] == "test"]
    if test:
        big = max(test, key=lambda r: abs(r["stimulus_amplitude_pa"]))
        err = (big["predicted_steady_deflection_mv"] - big["observed_steady_deflection_mv"]) / abs(
            big["observed_steady_deflection_mv"])
        rs = sorted(ps, key=lambda r: abs(r["stimulus_amplitude_pa"]))
        trend = rs[-1]["single_sweep_r_in_mohm"] - rs[0]["single_sweep_r_in_mohm"]
        width = bs["r_in_mohm_95"][1] - bs["r_in_mohm_95"][0]
        criterion = "within" if abs(err) <= 0.05 else "outside"
        relation = "exceeds" if abs(trend) > width else "does not exceed"
        out.append(f"At the largest held-out step ({big['stimulus_amplitude_pa']:g} pA), predicted steady-state "
                   f"deflection is {big['predicted_steady_deflection_mv']:.2f} mV and observed deflection is "
                   f"{big['observed_steady_deflection_mv']:.2f} mV. Relative error is {err:+.1%}, {criterion} the "
                   f"specified ±5% criterion. Single-sweep resistance changes by {trend:+.1f} MΩ between the "
                   f"smallest and largest current amplitudes; its absolute change {relation} the conditional "
                   f"bootstrap-interval width of {width:.1f} MΩ.")
    sags = [r["sag_ratio"] for r in ps if np.isfinite(r.get("sag_ratio", np.nan))]
    if sags:
        mx = max(sags)
        out.append(f"Maximum sag ratio is {mx:.3f}, using the most negative 5 ms voltage window."
                   + (" This feature does not distinguish a voltage-dependent conductance from spontaneous "
                      "hyperpolarizing events." if mx >= 0.10 else ""))
    ip = res.get("ipfx", {}).get("cell") or {}
    if ip.get("tau"):
        rel = ft["tau_ms"] / (ip["tau"] * 1e3) - 1
        out.append(f"The IPFX charging-window time constant is {ip['tau'] * 1e3:.2f} ms, compared with "
                   f"{ft['tau_ms']:.2f} ms for the whole-response passive fit (relative difference {rel:+.2%}, "
                   "IPFX denominator). These estimates use different fitting windows.")
    if np.isfinite(res["rheobase_pa"]):
        text = (f"Recorded-step rheobase is {res['rheobase_pa']:.0f} pA; firing-rate/current slope is "
                f"{res['fi_slope_hz_per_pa']:.3f} Hz pA⁻¹.")
        silent_above = sorted({f["stimulus_amplitude_pa"] for f in res["features"]
                               if f["qc_pass"] and f["stimulus_amplitude_pa"] > res["rheobase_pa"]
                               and f["spike_count"] == 0})
        if silent_above:
            text += (" Suprarheobase steps at " + ", ".join(f"{a:.0f}" for a in silent_above)
                     + " pA produced no detected spike, indicating a nonmonotonic response across sampled amplitudes.")
        out.append(text)
    else:
        dep = [f for f in res["features"] if f["qc_pass"] and f["protocol"] in ("X3LP_Rheo", "X4PS_SupraThresh")]
        out.append("No eligible depolarizing step evoked a spike; rheobase exceeds the largest tested amplitude."
                   if dep else "Rheobase was not estimated because no depolarizing step satisfied QC.")
    return out


def run(cfg: dict, project_dir: Path):
    rows = {"sweep_qc": [], "sweep_features": [], "passive_sweeps": [], "cell_features": [], "reference_checks": []}
    cells_out = []
    sources = []
    for cell_cfg in cfg["cells"]:
        path = project_dir / cell_cfg["path"]
        digest = sha256_file(path)
        if digest != cell_cfg["sha256"]:
            raise ValueError(f"{path}: sha256 {digest} does not match the pinned {cell_cfg['sha256']}")
        res = analyze_cell(path, cell_cfg["asset_id"], n_boot=cfg["analysis"]["n_boot"], seed=cfg["analysis"]["seed"])
        res["interpretation"] = interpret(res)
        res["comparisons"] = compare_with_references(res)
        cid = res["cell"]["cell_id"]
        rows["sweep_qc"] += res["qc"]
        rows["sweep_features"] += [{k: v for k, v in f.items() if k != "spike_times_from_onset_s"} for f in res["features"]]
        rows["passive_sweeps"] += res["passive_sweeps"]
        rows["reference_checks"] += res["comparisons"]
        cf = {"cell_id": cid, "asset_id": cell_cfg["asset_id"], "role": cell_cfg.get("role", ""),
              "n_long_square_sweeps": len(res["qc"]), "n_qc_fail": sum(not q["qc_pass"] for q in res["qc"]),
              "baseline_mv_median": res["baseline_v_median"] * 1e3,
              "peak_deflection_r_in_mohm": res["peak_deflection_r_in_mohm"],
              "rheobase_pa": res["rheobase_pa"], "fi_slope_hz_per_pa": res["fi_slope_hz_per_pa"],
              "passive_status": res["passive_status"]}
        if res["passive_status"] == "estimated":
            ft, bs = res["fit_train"], res["bootstrap"]
            test = [r for r in res["passive_sweeps"] if r["role"] == "test"]
            ps = sorted([r for r in res["passive_sweeps"] if r["role"] in ("train", "test")],
                        key=lambda r: abs(r["stimulus_amplitude_pa"]))
            cf["single_sweep_r_change_mohm"] = ps[-1]["single_sweep_r_in_mohm"] - ps[0]["single_sweep_r_in_mohm"]
            cf["r_change_exceeds_interval"] = bool(abs(cf["single_sweep_r_change_mohm"])
                                                   > bs["r_in_mohm_95"][1] - bs["r_in_mohm_95"][0])
            cf["bootstrap_contains_estimates"] = bs["estimates_inside_intervals"]
            cf.update({"passive_r_in_mohm": ft["r_in_mohm"], "passive_r_in_lo95": bs["r_in_mohm_95"][0],
                       "passive_r_in_hi95": bs["r_in_mohm_95"][1], "passive_tau_ms": ft["tau_ms"],
                       "passive_tau_lo95": bs["tau_ms_95"][0], "passive_tau_hi95": bs["tau_ms_95"][1],
                       "passive_c_eff_pf": ft["c_eff_pf"], "naive_se_r_in_mohm": ft["naive_se_r_in_mohm"],
                       "train_rms_residual_mv": ft["rms_residual_mv"],
                       "heldout_max_relative_rms_error": max((r["relative_rms_error"] for r in test), default=np.nan),
                       "max_sag_ratio": np.nanmax([r["sag_ratio"] for r in res["passive_sweeps"]
                                                   if r["role"] in ("train", "test")]),
                       "ode_check_max_abs_v": res["ode_check"]["max_abs_difference_v"]})
        ipc = (res.get("ipfx") or {}).get("cell") or {}
        cf.update({"ipfx_input_resistance_mohm": ipc.get("input_resistance"),
                   "ipfx_tau_ms": (ipc["tau"] * 1e3) if ipc.get("tau") else np.nan,
                   "ipfx_sag": ipc.get("sag"), "ipfx_rheobase_pa": ipc.get("rheobase_i")})
        rows["cell_features"].append(cf)
        cells_out.append(res)
        sources.append({"asset_id": cell_cfg["asset_id"], "path_in_dandiset": cell_cfg["dandi_path"],
                        "local_path": cell_cfg["path"], "sha256": digest, "role": cell_cfg.get("role", ""),
                        "download_url": f"https://api.dandiarchive.org/api/assets/{cell_cfg['asset_id']}/download/",
                        "cell_metadata": {k: res["cell"].get(k) for k in ("subject_id", "species", "age", "sex",
                                                                          "genotype", "session_start_time",
                                                                          "recording_temperature", "temperature_note")}})
    manifest = {
        "schema": "portfolio.data_manifest/1",
        "project": "membrane_report",
        "package_version": __version__,
        "dataset": {"dandiset": DANDISET, "version": VERSION, "doi": DOI, "citation": CITATION, "license": LICENSE,
                    "modality_used": "intracellular electrophysiology (current clamp) only"},
        "retrieved": cfg.get("retrieved"),
        "selection": cfg.get("selection", {}),
        "assets": sources,
        "transformations": [
            "NWB CurrentClampSeries and matching stimulus series read with pynwb; value = data * conversion + offset, "
            "applied once; conversion and offset recorded per sweep in sweep_features.csv",
            "only long-square protocols (X1PS_SubThresh, X3LP_Rheo, X4PS_SupraThresh) are read",
            "trailing exact zeros or NaN of at least 1 ms are treated as padding after an aborted sweep",
        ],
    }
    tables = {k: pd.DataFrame(v) for k, v in rows.items()}
    return cells_out, tables, manifest


def json_safe(o):
    """Standard JSON values with missing numerical results represented as null."""
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, np.generic):
        return json_safe(o.item())
    if o is pd.NA or (isinstance(o, float) and not np.isfinite(o)):
        return None
    return o


def write(cells_out, tables, manifest, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.8g")
    (out_dir / "source_manifest.json").write_text(json.dumps(json_safe(manifest), indent=2, allow_nan=False) + "\n")
    summary = [{k: v for k, v in c.items() if not k.startswith("_") and k not in ("qc", "features")} for c in cells_out]
    (out_dir / "cells.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n")
