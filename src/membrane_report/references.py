"""Run IPFX and eFEL on the same sweeps for comparison."""

from __future__ import annotations

import logging
import warnings

import numpy as np

SUBTHRESH_MIN_AMP_PA = -100.0


def ipfx_long_square(nwb_path, sweep_numbers, subthresh_min_amp=SUBTHRESH_MIN_AMP_PA) -> dict:
    """Run IPFX's LongSquareAnalysis on the given sweeps; returns plain numbers per cell and sweep."""
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.ERROR)
    try:
        return _ipfx_long_square(nwb_path, sweep_numbers, subthresh_min_amp)
    finally:
        root.setLevel(level)


def _ipfx_long_square(nwb_path, sweep_numbers, subthresh_min_amp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ipfx import stim_features as stf
        from ipfx.data_set_features import detection_parameters, extractors_for_sweeps
        from ipfx.dataset.create import create_ephys_data_set
        from ipfx.qc_feature_extractor import sweep_qc_features
        from ipfx.stimulus import StimulusType
        from ipfx.stimulus_protocol_analysis import LongSquareAnalysis

        ds = create_ephys_data_set(nwb_file=str(nwb_path))
        qc = {int(q["sweep_number"]): q for q in sweep_qc_features(ds)}
        nums = sorted(int(n) for n in sweep_numbers)
        ss = ds.sweep_set(nums)
        ss.align_to_start_of_epoch("experiment")
        start, dur, _, _, _ = stf.get_stim_characteristics(ss.sweeps[0].i, ss.sweeps[0].t)
        spx, sptx = extractors_for_sweeps(ss, start=start, end=start + dur,
                                          **detection_parameters(StimulusType.LONG_SQUARE))
        an = LongSquareAnalysis(spx, sptx, subthresh_min_amp=subthresh_min_amp,
                                require_subthreshold=False, require_suprathreshold=False)
        feats = an.analyze(ss)
    sweeps = []
    sf = feats["sweeps"].reset_index(drop=True)
    for k, n in enumerate(nums):
        spikes = feats["spikes_set"][k]
        thr = (spikes["threshold_t"].to_numpy() - start) if len(spikes) else np.array([])
        q = qc.get(n, {})
        sweeps.append({
            "sweep_id": n,
            "stim_amp_pa": float(sf.loc[k, "stim_amp"]),
            "v_baseline_mv": float(sf.loc[k, "v_baseline"]),
            "spike_count": int(len(spikes)),
            "spike_threshold_times_from_onset_s": thr.tolist(),
            "sag": float(sf.loc[k, "sag"]) if sf.loc[k, "stim_amp"] < 0 else np.nan,
            "pre_noise_rms_mv": q.get("pre_noise_rms_mv"),
            "slow_noise_rms_mv": q.get("slow_noise_rms_mv"),
            "post_noise_rms_mv": q.get("post_noise_rms_mv"),
            "vm_delta_mv": q.get("vm_delta_mv"),
            "qc_tags": "; ".join(q.get("tags", [])),
        })
    cell = {k: feats.get(k) for k in ("v_baseline", "input_resistance", "tau", "sag", "vm_for_sag", "rheobase_i",
                                      "fi_fit_slope")}
    cell = {k: (float(v) if v is not None and np.isscalar(v) else None) for k, v in cell.items()}
    sub = feats.get("subthreshold_membrane_property_sweeps")
    cell["input_resistance_sweeps"] = ([nums[i] for i in sub.index.values] if sub is not None else [])
    cell["subthresh_min_amp_pa"] = subthresh_min_amp
    return {"cell": cell, "sweeps": sweeps, "stim_start_aligned_s": float(start), "stim_duration_s": float(dur)}


def efel_spike_counts(sweeps_with_stim, threshold_mv=None) -> dict:
    """eFEL spike_count_stimint per step for [(sweep, (t_on, t_off, amp))], optionally at ``threshold_mv``."""
    import efel

    traces = []
    ids = []
    for sweep, stim in sweeps_with_stim:
        t_ms = sweep.time_seconds * 1e3
        traces.append({"T": t_ms, "V": sweep.voltage_volts * 1e3, "stim_start": [stim[0] * 1e3],
                       "stim_end": [stim[1] * 1e3]})
        ids.append(sweep.sweep_id)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if threshold_mv is not None:
            efel.set_setting("Threshold", float(threshold_mv))
        try:
            vals = efel.get_feature_values(traces, ["spike_count_stimint"], raise_warnings=False)
        finally:
            efel.reset()
    for sid, v in zip(ids, vals):
        sc = v.get("spike_count_stimint")
        out[sid] = int(sc[0]) if sc is not None and len(sc) else 0
    return out
