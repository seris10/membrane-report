"""Stimulus segments, test pulses and the end of valid recording, determined per sweep."""

from __future__ import annotations

import numpy as np

COMMAND_ZERO_TOL_A = 1e-13
PAD_MIN_DURATION_S = 0.001
TEST_PULSE_MAX_DURATION_S = 0.02
TEST_PULSE_MAX_START_S = 0.1


def step_segments(i, rate_hz):
    """Contiguous runs of |i| > 0.1 pA as [(start s, end s, median amplitude A)]."""
    i = np.asarray(i, float)
    nz = np.flatnonzero(np.isfinite(i) & (np.abs(i) > COMMAND_ZERO_TOL_A))
    if len(nz) == 0:
        return []
    breaks = np.flatnonzero((np.diff(nz) > 1) | (np.abs(np.diff(i[nz])) > COMMAND_ZERO_TOL_A))
    starts = np.r_[nz[0], nz[breaks + 1]]
    ends = np.r_[nz[breaks], nz[-1]]
    return [(s / rate_hz, (e + 1) / rate_hz, float(np.median(i[s:e + 1]))) for s, e in zip(starts, ends)]


def recording_end_index(v, rate_hz=None) -> int:
    """Index one past the last valid sample, treating a trailing run of zeros or NaN (>= 1 ms) as padding."""
    v = np.asarray(v, float)
    if v.size == 0:
        return 0
    bad = (v == 0.0) | ~np.isfinite(v)
    if not bad[-1]:
        return len(v)
    good = np.flatnonzero(~bad)
    end = int(good[-1] + 1) if len(good) else 0
    if rate_hz is not None and (len(v) - end) < PAD_MIN_DURATION_S * rate_hz:
        return len(v)
    return end


def split_test_pulse(segments):
    """Split off a test pulse (<= 20 ms, starting in the first 100 ms); returns (test_pulse or None, rest)."""
    test = None
    rest = []
    for seg in segments:
        start, end, _ = seg
        if test is None and end - start <= TEST_PULSE_MAX_DURATION_S and start <= TEST_PULSE_MAX_START_S:
            test = seg
        else:
            rest.append(seg)
    return test, rest


def long_square_stimulus(segments, min_duration_s=0.5):
    """The one segment of at least ``min_duration_s`` after the test pulse, or None if there is not exactly one."""
    _, rest = split_test_pulse(segments)
    long = [s for s in rest if s[1] - s[0] >= min_duration_s]
    if len(long) != 1:
        return None
    return long[0]
