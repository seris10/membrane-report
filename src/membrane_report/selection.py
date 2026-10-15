"""Cell selection criteria C0-C4 from protocol codes and sweep completeness."""

from __future__ import annotations

import re

from .epochs import long_square_stimulus

SUBTHRESHOLD_CODES = ("X1PS_SubThresh",)
DEPOLARIZING_CODES = ("X3LP_Rheo", "X4PS_SupraThresh")
LONG_SQUARE_CODES = SUBTHRESHOLD_CODES + DEPOLARIZING_CODES
POST_STIM_REQUIRED_S = 0.5
MIN_SUB_AMPLITUDES = 3
MIN_DEP_AMPLITUDES = 2


def session_id(path: str) -> int:
    m = re.search(r"_ses-(\d+)_", path)
    if not m:
        raise ValueError(f"no session identifier in {path!r}")
    return int(m.group(1))


def complete_step(sweep):
    """(start, end, amplitude) of the long-square step if recording continues 0.5 s past it, else None."""
    stim = long_square_stimulus(sweep.stimulus_intervals)
    if stim is None:
        return None
    if sweep.recording_end_s < stim[1] + POST_STIM_REQUIRED_S:
        return None
    return stim


def evaluate_cell(sweeps, all_codes=None) -> dict:
    """Apply C0, C2 and C3 to a cell's current-clamp sweeps."""
    codes = {s.protocol_name for s in sweeps}
    reasons = []
    if not (set(SUBTHRESHOLD_CODES) & codes and set(DEPOLARIZING_CODES) & codes):
        present = sorted(all_codes or codes)
        reasons.append("C0: X-series long-square protocols absent (codes present: "
                       + ", ".join(present[:8]) + (" ..." if len(present) > 8 else "") + ")")
    sub_amps = set()
    dep_amps = set()
    n_incomplete = 0
    for s in sweeps:
        if s.protocol_name not in LONG_SQUARE_CODES:
            continue
        stim = complete_step(s)
        if stim is None:
            n_incomplete += 1
            continue
        amp_pa = round(stim[2] * 1e12, 1)
        if s.protocol_name in SUBTHRESHOLD_CODES and amp_pa < 0:
            sub_amps.add(amp_pa)
        elif s.protocol_name in DEPOLARIZING_CODES and amp_pa > 0:
            dep_amps.add(amp_pa)
    if not reasons:
        if len(sub_amps) < MIN_SUB_AMPLITUDES:
            reasons.append(f"C2: {len(sub_amps)} complete subthreshold amplitudes (need {MIN_SUB_AMPLITUDES})")
        if len(dep_amps) < MIN_DEP_AMPLITUDES:
            reasons.append(f"C3: {len(dep_amps)} complete depolarizing amplitudes (need {MIN_DEP_AMPLITUDES})")
    return {"qualifies": not reasons, "reasons": reasons,
            "subthreshold_amplitudes_pa": sorted(sub_amps), "depolarizing_amplitudes_pa": sorted(dep_amps),
            "n_incomplete_long_square_sweeps": n_incomplete}
