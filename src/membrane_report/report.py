"""Static HTML report: QC, raw traces, features, passive fits with held-out steps, reference checks."""

from __future__ import annotations

import html
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import passive as pv  # noqa: E402
from .run import TOLERANCE_NOTE  # noqa: E402
from .selection import DEPOLARIZING_CODES, SUBTHRESHOLD_CODES  # noqa: E402

plt.rcParams.update({
    "svg.fonttype": "none", "svg.hashsalt": "membrane-report", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "legend.frameon": False,
})
TRAIN_COLOR = "#2b6cb0"
TEST_COLOR = "#c05621"
FAIL_COLOR = "#a0aec0"
CELL_COLORS = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#b7791f", "#319795"]


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def _esc(x) -> str:
    return html.escape(str(x))


def _fmt(v, digits=4):
    if v is None:
        return ""
    if isinstance(v, str):
        return _esc(v)
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    v = float(v)
    if v == 0:
        return "0"
    if not np.isfinite(v):
        return "∞" if v > 0 else "−∞"
    if abs(v) >= 1e5 or abs(v) < 1e-3:
        coefficient, exponent = f"{v:.{digits - 1}e}".split("e")
        power = str(int(exponent)).replace("-", "−")
        return f"{coefficient} × 10<sup>{power}</sup>"
    return f"{v:.{digits}g}"


def _table(df, cols=None, headers=None, digits=4) -> str:
    if df is None or len(df) == 0:
        return "<p class='muted'>No rows.</p>"
    cols = cols or list(df.columns)
    headers = headers or cols
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = []
    for _, r in df.reindex(columns=cols).iterrows():
        cells = []
        for c in cols:
            v = r[c]
            num = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
            cls = "num" if num else ""
            if isinstance(v, str) and v in ("SUPPORTED", "CONTRADICTED", "NOT TESTED"):
                cls = {"SUPPORTED": "ok", "CONTRADICTED": "bad", "NOT TESTED": "warn"}[v]
            display = {"SUPPORTED": "Within tolerance", "CONTRADICTED": "Outside tolerance", "NOT TESTED": "Not evaluated"}.get(v, v) if isinstance(v, str) else v
            cells.append(f"<td class='{cls}'>{_fmt(display, digits)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _short(cid):
    return str(cid).split("/")[0]


def _cell_figure(res):
    sweeps = res["_sweeps"]
    qc = {q["sweep_id"]: q for q in res["qc"]}
    fig = plt.figure(figsize=(10.5, 6.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 0.8, 2.0], hspace=0.55, wspace=0.22)
    ax_v = fig.add_subplot(gs[0, 0])
    ax_i = fig.add_subplot(gs[1, 0], sharex=ax_v)
    ax_d = fig.add_subplot(gs[0:2, 1])
    ax_p = fig.add_subplot(gs[2, 0])
    ax_r = fig.add_subplot(gs[2, 1])
    for sid, s in sweeps.items():
        if s.protocol_name not in SUBTHRESHOLD_CODES:
            continue
        t = s.time_seconds
        end = int(round(qc[sid]["recording_end_s"] * s.sampling_rate_hz))
        ok = qc[sid]["qc_pass"]
        kw = {"color": TRAIN_COLOR if ok else FAIL_COLOR, "lw": 0.6, "alpha": 0.9 if ok else 0.7,
              "linestyle": "-" if ok else "--"}
        ax_v.plot(t[:end:10], s.voltage_volts[:end:10] * 1e3, **kw)
        ax_i.plot(t[::10], s.injected_current_amperes[::10] * 1e12, **kw)
    ax_v.set_ylabel("membrane potential (mV)")
    ax_v.set_title("Subthreshold current-clamp responses", loc="left")
    ax_i.set_ylabel("command (pA)")
    ax_i.set_xlabel("time (s)")
    plt.setp(ax_v.get_xticklabels(), visible=False)
    dep = sorted([f for f in res["features"] if f["protocol"] in DEPOLARIZING_CODES and f["qc_pass"]],
                 key=lambda f: f["stimulus_amplitude_pa"])
    shown = []
    fired = [f for f in dep if f["spike_count"] > 0]
    if dep:
        below = [f for f in dep if f["spike_count"] == 0]
        if below:
            shown.append(below[-1])
        if fired:
            shown.append(fired[0])
            if len(fired) > 1:
                shown.append(fired[-1])
    for k, f in enumerate(shown):
        s = sweeps[f["sweep_id"]]
        t = s.time_seconds
        sl = slice(int(0.5 * s.sampling_rate_hz), int(2.0 * s.sampling_rate_hz))
        ax_d.plot(t[sl], s.voltage_volts[sl] * 1e3, lw=0.6, color=CELL_COLORS[k % 6],
                  label=f"{f['stimulus_amplitude_pa']:.0f} pA: {f['spike_count']} spikes")
    ax_d.set_title("Depolarizing current-clamp responses", loc="left")
    ax_d.set_xlabel("time (s)")
    ax_d.set_ylabel("membrane potential (mV)")
    if shown:
        ax_d.legend(loc="upper right")
    steps, roles, fit = res.get("_steps"), res.get("_roles"), res.get("_fit_train")
    if steps:
        for st, ro in zip(steps, roles):
            pred = pv.predict_step(st, fit)
            color = TRAIN_COLOR if ro == "train" else TEST_COLOR
            ax_p.plot(st.t - st.t_on, st.v * 1e3, ".", ms=1.2, color="#4a5568", alpha=0.5)
            ax_p.plot(st.t - st.t_on, pred * 1e3, color=color, lw=1.2, linestyle="-" if ro == "train" else "--")
            m = st.fit_mask
            ax_r.plot(st.t[m] - st.t_on, (st.v[m] - pred[m]) * 1e3, lw=0.7, color=color,
                      label=f"{st.delta_i * 1e12:.0f} pA ({ro})")
        ax_p.set_title("Passive-model fit and held-out predictions", loc="left")
        ax_p.set_xlabel("time from step onset (s)")
        ax_p.set_ylabel("mV (1 ms bins)")
        ax_r.axhline(0, color="#4a5568", lw=0.8)
        ax_r.set_title("Voltage residuals (observed − predicted)", loc="left")
        ax_r.set_xlabel("time from step onset (s)")
        ax_r.set_ylabel("mV")
        handles, labels = ax_r.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax_r.legend(uniq.values(), uniq.keys(), ncol=2, fontsize=7)
    else:
        for ax in (ax_p, ax_r):
            ax.text(0.5, 0.5, res.get("passive_status", ""), ha="center", va="center", transform=ax.transAxes,
                    wrap=True)
            ax.set_axis_off()
    return _svg(fig)


def _linearity_figure(cells):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2))
    ax, ax2 = axes
    for k, res in enumerate(cells):
        color = CELL_COLORS[k % len(CELL_COLORS)]
        ps = [r for r in res["passive_sweeps"] if r["role"] in ("train", "test")]
        if ps:
            amps = np.array([r["stimulus_amplitude_pa"] for r in ps])
            obs = np.array([r["observed_steady_deflection_mv"] for r in ps])
            ax.plot(amps, obs, "o", ms=3.5, color=color, label=_short(res["cell"]["cell_id"]))
            grid = np.linspace(amps.min() * 1.05, 0, 20)
            ax.plot(grid, res["fit_train"]["r_in_mohm"] * 1e6 * grid * 1e-12 * 1e3, color=color, lw=0.9)
        dep = sorted([f for f in res["features"] if f["protocol"] in DEPOLARIZING_CODES and f["qc_pass"]],
                     key=lambda f: f["stimulus_amplitude_pa"])
        if dep:
            ax2.plot([f["stimulus_amplitude_pa"] for f in dep], [f["rate_hz"] for f in dep], "o-", ms=3.5, lw=0.9,
                     color=color, label=_short(res["cell"]["cell_id"]))
    ax.axhline(0, color="#a0aec0", lw=0.6)
    ax.set_title("Steady-state current–voltage relationship", loc="left")
    ax.set_xlabel("command step (pA)")
    ax.set_ylabel("steady deflection (mV)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    ax2.set_title("Firing-rate response", loc="left")
    ax2.set_xlabel("command step (pA)")
    ax2.set_ylabel("spikes per second")
    fig.tight_layout()
    return _svg(fig)


CSS = """
:root{--fg:#1a202c;--muted:#4a5568;--line:#e2e8f0;--bg:#ffffff;--panel:#f7fafc}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",
Helvetica,Arial,sans-serif}
main{max-width:1120px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:17px;margin:28px 0 8px;padding-top:10px;border-top:1px solid var(--line)}
h3{font-size:14px;margin:16px 0 6px}
p{margin:6px 0 10px;max-width:92ch}
.muted{color:var(--muted)}
.lede{font-size:15px}
.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:12px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px 12px}
.card h3{margin:0 0 4px;font-size:13px}
.card p{margin:0;font-size:13px}
.tw{overflow-x:auto;margin:8px 0 14px}
table{border-collapse:collapse;font-size:12px;width:100%}
th,td{padding:3px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{background:var(--panel);font-weight:600;vertical-align:bottom}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.ok{color:#276749;font-weight:600}td.bad{color:#c53030;font-weight:600}td.warn{color:#975a16;font-weight:600}
figure{margin:10px 0 16px}
figure svg{width:100%;height:auto}
figcaption{font-size:12px;color:var(--muted);max-width:110ch}
ul{margin:4px 0 10px;padding-left:20px}
code{font-size:12px;background:var(--panel);padding:0 3px;border-radius:3px}
@media (max-width:760px){.grid3{grid-template-columns:1fr}}
"""


def render(cells, tables, manifest, selection_log=None, version="") -> str:
    cf = tables["cell_features"]
    rc = tables["reference_checks"]
    parts = ["<h1>Passive membrane properties and current-clamp responses</h1>",
             f"<p class='muted'>membrane_report {version} | DANDI {manifest['dataset']['dandiset']} version "
             f"{manifest['dataset']['version']} | {len(cells)} cells</p>",
             "<p class='lede'>Current-clamp recordings from Allen Institute Patch-seq experiments are analyzed to estimate "
             "apparent input resistance, membrane time constant and firing-response features. A single-compartment "
             "passive model is fitted to small hyperpolarizing steps and evaluated on larger held-out steps. "
             f"The sample comprises {len(cells)} protocol-selected mouse visual-cortex cells. Estimates characterize "
             "these recordings under the specified model and selection criteria.</p>"]
    n_sup = int((rc.verdict == "SUPPORTED").sum()) if len(rc) else 0
    n_con = int((rc.verdict == "CONTRADICTED").sum()) if len(rc) else 0
    n_nt = int((rc.verdict == "NOT TESTED").sum()) if len(rc) else 0
    est = cf[cf.passive_status == "estimated"] if len(cf) else cf
    worst_heldout = est.heldout_max_relative_rms_error.max() if len(est) else np.nan
    parts.append("<div class='grid3'>")
    sc = tables.get("synthetic_checks")
    sc_text = (f"{int(sc.passed.sum())} of {len(sc)} verification tests on synthetic recordings pass (section 6)."
               if sc is not None and len(sc) else "Synthetic checks not run.")
    ode_text = (f"Analytic and numerically integrated passive responses agree to {est.ode_check_max_abs_v.max():.1e} V "
                "on the eligible cells' fitted parameters." if len(est) else
                "No eligible passive fit was available for the numerical comparison.")
    parts.append(f"<div class='card'><h3>Numerical verification</h3><p>{ode_text} {sc_text}</p></div>")
    parts.append(f"<div class='card'><h3>Cross-implementation feature comparisons</h3><p>{n_sup} checks against IPFX and eFEL "
                 f"within tolerance, {n_con} outside tolerance and {n_nt} not evaluated. Comparable feature definitions "
                 "are specified in section 4.</p></div>")
    n_sub = int(((est.single_sweep_r_change_mohm < 0) & est.r_change_exceeds_interval.astype(bool)).sum()) \
        if len(est) else 0
    applicability_text = (f"Fitted on the two smallest steps, held-out steps are predicted with relative RMS errors "
                          f"up to {worst_heldout:.1%}. In {n_sub} of {len(est)} cells the apparent resistance falls with "
                          "step size by more than its bootstrap interval, which one passive compartment cannot "
                          "reproduce (section 3)." if len(est) else
                          "No cell had enough eligible subthreshold amplitudes for a passive fit and held-out check.")
    parts.append(f"<div class='card'><h3>Passive-model prediction error</h3><p>{applicability_text}</p></div></div>")

    parts.append("<h2>1. Dataset provenance and cell selection</h2>")
    parts.append(f"<p>{_esc(manifest['dataset']['citation'])} License {_esc(manifest['dataset']['license'])}.</p>")
    sel = manifest.get("selection", {})
    parts.append(f"<p>Pre-analysis selection criteria: {_esc(sel.get('rule', ''))}</p>")
    if selection_log is not None and len(selection_log):
        examined = len(selection_log)
        n_q = int(selection_log.qualifies.astype(str).eq("True").sum())
        parts.append(f"<p>{examined} assets were examined in that order; {n_q} qualified and the others are listed "
                     "with their reason in <code>data/selection_log.csv</code>. The first qualifying cell is the "
                     "single-cell case below.</p>")
    rows = []
    for a in manifest["assets"]:
        md = a["cell_metadata"]
        rows.append({"cell (subject)": md["subject_id"], "role": a["role"], "DANDI path": a["path_in_dandiset"],
                     "age": md["age"], "genotype": md["genotype"], "sha256": a["sha256"][:12]})
    parts.append(_table(pd.DataFrame(rows)))
    parts.append("<p class='muted'>Recording temperature, access resistance and electrode resistance are not recorded "
                 "in the source files. Holding current, bridge balance and capacitance "
                 "compensation are stored with magnitudes in pA, megohm and pF although the NWB schema defines amperes, "
                 "ohms and farads; they are flagged per sweep and not used in any calculation.</p>")

    parts.append("<h2>2. Cell-level responses and passive-model fits</h2>")
    for k, res in enumerate(cells):
        cid = res["cell"]["cell_id"]
        role = next((a["role"] for a in manifest["assets"] if a["asset_id"] == res["cell"]["source_asset_id"]), "")
        parts.append(f"<h3>Cell {_esc(_short(cid))}{' (single-cell case)' if role == 'first cell' else ''}</h3>")
        parts.append("<ul>" + "".join(f"<li>{_esc(s)}</li>" for s in res["interpretation"]) + "</ul>")
        parts.append(f"<figure>{_cell_figure(res)}<figcaption><b>Recorded responses and passive-model predictions.</b> Upper panels: subthreshold command and voltage responses, "
                     "and selected depolarizing responses below threshold, at the first spiking step and at the largest spiking "
                     "step. Lower panels: 1 ms voltage-bin means (gray), passive-model fits to "
                     "training steps (blue, solid), held-out predictions (orange, dashed), and "
                     "step-specific voltage residuals. Samples within 1 ms of each transition are excluded from fitting.</figcaption>"
                     "</figure>")
        fails = [q for q in res["qc"] if not q["qc_pass"]]
        if fails:
            parts.append(_table(pd.DataFrame(fails), ["sweep_id", "protocol", "stimulus_amplitude_pa", "recording_end_s",
                                                      "qc_reasons"],
                                ["sweep", "protocol", "step (pA)", "recording ends (s)", "QC exclusion criterion"]))
        ps = pd.DataFrame([r for r in res["passive_sweeps"] if r["role"] in ("train", "test")])
        if len(ps):
            parts.append(_table(ps.sort_values("stimulus_amplitude_pa", ascending=False),
                                ["sweep_id", "stimulus_amplitude_pa", "role", "sag_ratio", "single_sweep_r_in_mohm",
                                 "single_sweep_tau_ms", "observed_steady_deflection_mv",
                                 "predicted_steady_deflection_mv", "train_model_rms_error_mv", "relative_rms_error"],
                                ["sweep", "step (pA)", "role", "sag ratio", "Single-sweep R (MΩ)", "Single-sweep τ (ms)",
                                 "observed steady (mV)", "predicted steady (mV)", "RMS error (mV)", "relative RMS"]))

    parts.append("<h2>3. Variation across cells and stimulus amplitudes</h2>")
    parts.append(f"<figure>{_linearity_figure(cells)}<figcaption><b>Current–voltage and firing-rate relationships.</b> Left: steady-state voltage deflections "
                 "and predictions from each cell's small-step passive fit. Right: firing rate as a function of "
                 "current amplitude for depolarizing sweeps satisfying QC.</figcaption></figure>")
    parts.append(_table(cf, ["cell_id", "n_long_square_sweeps", "n_qc_fail", "baseline_mv_median", "passive_r_in_mohm",
                             "passive_r_in_lo95", "passive_r_in_hi95", "bootstrap_contains_estimates", "passive_tau_ms",
                             "passive_c_eff_pf", "heldout_max_relative_rms_error", "max_sag_ratio",
                             "peak_deflection_r_in_mohm", "ipfx_tau_ms", "rheobase_pa", "fi_slope_hz_per_pa"],
                        ["cell", "sweeps", "QC fail", "baseline (mV)", "passive R (MΩ)", "lo95", "hi95",
                         "interval contains estimate", "Passive τ (ms)", "C eff (pF)", "worst held-out rel. RMS",
                         "max sag", "peak R (MΩ)", "IPFX tau (ms)", "rheobase (pA)", "f-I (Hz/pA)"]))
    if len(est):
        parts.append(f"<p>Across the {len(est)} cells (n = cells, not sweeps or samples): passive input resistance "
                     f"median {est.passive_r_in_mohm.median():.0f} MΩ (range {est.passive_r_in_mohm.min():.0f}-"
                     f"{est.passive_r_in_mohm.max():.0f}), time constant median {est.passive_tau_ms.median():.1f} ms "
                     f"(range {est.passive_tau_ms.min():.1f}-{est.passive_tau_ms.max():.1f}). Cells chosen by "
                     "protocol do not estimate a population distribution.</p>")
    if len(est):
        width = float((est.passive_r_in_hi95 - est.passive_r_in_lo95).median())
        change = float(est.single_sweep_r_change_mohm.abs().median())
        parts.append(f"<p>The bootstrap intervals for R describe only the noise within the training sweeps (median "
                     f"width {width:.1f} MΩ). The change of a cell's single-sweep resistance between its smallest and "
                     f"largest step has median size {change:.1f} MΩ, and the cells span "
                     f"{est.passive_r_in_mohm.max() - est.passive_r_in_mohm.min():.0f} MΩ. Where those exceed the "
                     "interval width, between-amplitude or between-cell variation exceeds the uncertainty represented by the "
                     "conditional within-sweep bootstrap. A single resistance estimate does not capture this variation.</p>")

    parts.append("<h2>4. Feature comparisons with IPFX and eFEL</h2>")
    parts.append("<p>IPFX 2.1.2 (Allen Institute) was run on exactly the QC-passing sweeps this package used, with "
                 "its long-square defaults; eFEL 5.7 counted spikes within each step. Where a definition differs "
                 "(time constant, resistance from the whole response versus the peak), the row is informational and "
                 "the difference is explained rather than tested.</p>")
    if len(rc):
        tested = rc[rc.verdict.isin(["SUPPORTED", "CONTRADICTED", "NOT TESTED"])]
        summary = []
        for q, g in tested.groupby("quantity", sort=False):
            summary.append({"quantity": q, "reference": g.get("reference_name", pd.Series([""])).iloc[0],
                            "difference kind": g.get("difference_kind", pd.Series([""])).iloc[0],
                            "largest difference over cells": g.get("difference", pd.Series([np.nan])).max(),
                            "tolerance": g.get("tolerance", pd.Series([np.nan])).iloc[0],
                            "Cells within tolerance": f"{int((g.verdict == 'SUPPORTED').sum())} of {len(g)}",
                            "verdict": "SUPPORTED" if (g.verdict == "SUPPORTED").all() else (
                                "CONTRADICTED" if (g.verdict == "CONTRADICTED").any() else "NOT TESTED")})
        parts.append(_table(pd.DataFrame(summary), digits=3))
        parts.append("<p class='muted'>Per-cell values and notes are in reference_checks.csv. "
                     + _esc(TOLERANCE_NOTE) + "</p>")
        info = rc[rc.verdict == "informational"]
        if len(info):
            parts.append("<h3>Comparisons involving different feature definitions</h3>")
            wide = []
            for cid, g in info.groupby("cell_id", sort=False):
                row = {"cell": _short(cid)}
                for _, r in g.iterrows():
                    q = r["quantity"]
                    if q.startswith("membrane time constant"):
                        row["Passive τ (ms)"], row["IPFX τ (ms)"] = r.this_package, r.reference
                    elif q.startswith("input resistance"):
                        row["R passive steady (MΩ)"], row["R IPFX peak (MΩ)"] = r.this_package, r.reference
                    elif q.startswith("spike time"):
                        row["spike time offset vs IPFX (ms)"] = r.this_package
                    elif "default threshold" in q:
                        row["sweeps differing from eFEL at -20 mV"] = r.this_package
                        row["eFEL note"] = r.note if r.this_package else ""
                wide.append(row)
            parts.append(_table(pd.DataFrame(wide).fillna("")))
            parts.append("<p class='muted'>The passive fit uses the two smallest steps and the whole response; IPFX's "
                         "resistance uses the most negative voltage of steps down to -90 pA, which includes early sag, "
                         "and its time constant uses the stretch from 10% of the peak deflection to the peak. eFEL's "
                         "default counts any peak above -20 mV, so noise on a depolarized interspike plateau can count; "
                         "with a 0 mV threshold its counts equal this package's in every sweep (table above).</p>")

    parts.append("<h2>5. Parameter interpretation and model scope</h2><ul>"
                 "<li><b>Passive compartment.</b> Supports an apparent input resistance and time constant for small "
                 "hyperpolarizing steps from rest in each cell, and a prediction of nearby steps. It does not "
                 "identify membrane capacitance, specific conductances or cell geometry, and where a cell's response "
                 "is nonlinear in step size (section 3) it cannot follow it.</li>"
                 "<li><b>Peak-deflection resistance (IPFX definition).</b> A reproducible summary of the most negative "
                 "voltage per step; it mixes passive charging with early sag and is not a model parameter.</li>"
                 "<li><b>Spike counts, rheobase and f-I slope.</b> Directly measured on the recorded steps; rheobase is "
                 "only as fine as the step series.</li>"
                 "<li><b>Not supported by any analysis here:</b> ion-channel conductances, ATP use, population "
                 "properties, or cell types; the recordings were not collected by the author of this analysis.</li></ul>")
    if sc is not None and len(sc):
        parts.append("<h2>6. Verification with synthetic recordings</h2>")
        parts.append("<p>Verification inputs comprise simulated passive responses with and without added noise, the "
                     "same response expressed in V/A and mV/pA, truncated and unstimulated recordings, and a "
                     "Hodgkin–Huxley simulation with a known spike count.</p>")
        parts.append(_table(sc, ["check", "passed", "detail"], ["check", "passed", "measured"]))
    parts.append("<p class='muted'>Methods: docs/methods.md. Limitations: docs/limitations.md. Decisions and "
                 "development record: docs/development.md.</p>")
    body = "\n".join(parts)
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Membrane Recording Report</title><style>{CSS}</style></head><body><main>{body}</main></body></html>\n")
