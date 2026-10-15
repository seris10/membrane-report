"""Sweep container in SI units and the NWB reader that fills it."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .epochs import recording_end_index, step_segments

UNIT_SCALE = {"V": 1.0, "mV": 1e-3, "A": 1.0, "pA": 1e-12, "nA": 1e-9}


@dataclass
class Sweep:
    cell_id: str
    sweep_id: int
    source_asset_id: str
    protocol_name: str
    sampling_rate_hz: float
    voltage_volts: np.ndarray
    injected_current_amperes: np.ndarray
    recording_metadata: dict = field(default_factory=dict)
    source_quality_flags: list[str] = field(default_factory=list)

    def __post_init__(self):
        v = np.asarray(self.voltage_volts, dtype=float)
        i = np.asarray(self.injected_current_amperes, dtype=float)
        if v.ndim != 1 or v.shape != i.shape:
            raise ValueError(f"sweep {self.sweep_id}: voltage and current must be 1-D arrays of equal length")
        if len(v) < 2:
            raise ValueError(f"sweep {self.sweep_id}: at least two samples are required")
        if not (np.isfinite(self.sampling_rate_hz) and self.sampling_rate_hz > 0):
            raise ValueError(f"sweep {self.sweep_id}: invalid sampling rate {self.sampling_rate_hz}")
        self.voltage_volts = v
        self.injected_current_amperes = i

    @property
    def time_seconds(self) -> np.ndarray:
        return np.arange(len(self.voltage_volts)) / self.sampling_rate_hz

    @property
    def stimulus_intervals(self) -> list[tuple[float, float, float]]:
        """(start s, end s, amplitude A) of every constant nonzero command segment."""
        return step_segments(self.injected_current_amperes, self.sampling_rate_hz)

    @property
    def recording_end_s(self) -> float:
        return recording_end_index(self.voltage_volts, self.sampling_rate_hz) / self.sampling_rate_hz


def sweep_from_arrays(t, v, i, *, v_unit="V", i_unit="A", **kw) -> Sweep:
    """Build a Sweep from uniformly spaced time, voltage and current arrays in the stated units."""
    t = np.asarray(t, float)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)):
        raise ValueError("timestamps must be a finite 1-D array with at least two samples")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if np.max(np.abs(dt - dt.mean())) > 1e-6 * dt.mean():
        raise ValueError("timestamps must be uniformly spaced")
    if np.shape(v) != t.shape or np.shape(i) != t.shape:
        raise ValueError("timestamps, voltage and current must have the same length")
    if v_unit not in ("V", "mV") or i_unit not in ("A", "pA", "nA"):
        raise ValueError("voltage must use V or mV; current must use A, pA or nA")
    meta = dict(kw.pop("recording_metadata", {}))
    meta["starting_time"] = float(t[0])
    return Sweep(voltage_volts=np.asarray(v, float) * UNIT_SCALE[v_unit],
                 injected_current_amperes=np.asarray(i, float) * UNIT_SCALE[i_unit],
                 sampling_rate_hz=1.0 / dt.mean(), recording_metadata=meta, **kw)


def _scaled(ts):
    """NWB data * conversion + offset, with both factors."""
    data = np.asarray(ts.data[:], dtype=float)
    conv = float(ts.conversion)
    offset = float(getattr(ts, "offset", 0.0) or 0.0)
    if not (np.isfinite(conv) and conv != 0 and np.isfinite(offset)):
        raise ValueError(f"{ts.name}: conversion must be finite and nonzero; offset must be finite")
    return data * conv + offset, conv, offset


def _timing(ts, n):
    """Uniform sample rate and original time origin; reject unsupported irregular timestamps."""
    if ts.timestamps is not None:
        t = np.asarray(ts.timestamps[:], float)
        if t.ndim != 1 or len(t) != n or len(t) < 2 or not np.all(np.isfinite(t)):
            raise ValueError(f"{ts.name}: timestamps must be finite and match the data length")
        dt = np.diff(t)
        if np.any(dt <= 0) or not np.allclose(dt, dt.mean(), rtol=1e-6, atol=0):
            raise ValueError(f"{ts.name}: timestamps must be strictly increasing and uniformly spaced")
        return float(1 / dt.mean()), float(t[0])
    rate, start = ts.rate, ts.starting_time
    if rate is None or start is None or not (np.isfinite(rate) and rate > 0 and np.isfinite(start)):
        raise ValueError(f"{ts.name}: invalid sampling rate or starting time")
    return float(rate), float(start)


def _optional(ts, name):
    v = getattr(ts, name, None)
    if v is None:
        return None
    v = float(v)
    return None if np.isnan(v) else v


def read_nwb(path, asset_id: str, protocols: set[str] | None = None) -> tuple[dict, list[Sweep]]:
    """Read current-clamp sweeps, optionally limited to ``protocols``; returns (cell metadata, sweeps)."""
    from pynwb import NWBHDF5IO
    from pynwb.icephys import CurrentClampSeries, CurrentClampStimulusSeries

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Use of icephys_filtering is deprecated", category=UserWarning)
        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            nwb = io.read()
            subject = nwb.subject
            cell_id = f"{subject.subject_id if subject else 'unknown'}/{nwb.identifier[:12]}"
            stim_by_sweep = {}
            for s in nwb.stimulus.values():
                if not isinstance(s, CurrentClampStimulusSeries):
                    continue
                if s.sweep_number is None:
                    raise ValueError(f"{s.name}: stimulus has no sweep number")
                sn = int(s.sweep_number)
                if sn in stim_by_sweep:
                    raise ValueError(f"sweep {sn}: ambiguous matching stimulus series")
                stim_by_sweep[sn] = s
            electrode = next(iter(nwb.icephys_electrodes.values()), None)
            cell = {
                "cell_id": cell_id,
                "source_asset_id": asset_id,
                "nwb_identifier": nwb.identifier,
                "session_start_time": str(nwb.session_start_time),
                "subject_id": subject.subject_id if subject else None,
                "species": subject.species if subject else None,
                "age": subject.age if subject else None,
                "sex": getattr(subject, "sex", None) if subject else None,
                "genotype": subject.genotype if subject else None,
                "institution": nwb.institution,
                "electrode_description": electrode.description if electrode else None,
                "electrode_resistance": getattr(electrode, "resistance", None) if electrode else None,
                "recording_temperature": None,
                "temperature_note": "the file records no bath or recording temperature",
            }
            cell["stimulus_codes"] = sorted({str(ts.stimulus_description or "").replace("_DA_0", "")
                                             for ts in nwb.acquisition.values()})
            sweeps = []
            for ts in nwb.acquisition.values():
                if not isinstance(ts, CurrentClampSeries):
                    continue
                code = str(ts.stimulus_description or "").replace("_DA_0", "")
                if protocols is not None and code not in protocols:
                    continue
                if ts.sweep_number is None:
                    raise ValueError(f"{ts.name}: response has no sweep number")
                sn = int(ts.sweep_number)
                if any(s.sweep_id == sn for s in sweeps):
                    raise ValueError(f"sweep {sn}: ambiguous response series")
                st = stim_by_sweep.get(sn)
                flags = []
                v, v_conv, v_off = _scaled(ts)
                rate, start_time = _timing(ts, len(v))
                if st is None:
                    flags.append("no stimulus series for this sweep")
                    i = np.full_like(v, np.nan)
                    i_conv = i_off = None
                    i_unit = None
                else:
                    i, i_conv, i_off = _scaled(st)
                    i_unit = st.unit
                    if len(i) != len(v):
                        raise ValueError(f"sweep {sn}: stimulus length {len(i)} != response length {len(v)}")
                    i_rate, i_start = _timing(st, len(i))
                    if not np.isclose(i_rate, rate, rtol=1e-9, atol=0) or not np.isclose(
                            i_start, start_time, rtol=0, atol=1e-6 / rate):
                        raise ValueError(f"sweep {sn}: stimulus and response sampling times do not align")
                    if st.electrode is not ts.electrode:
                        raise ValueError(f"sweep {sn}: stimulus and response electrodes do not match")
                if ts.unit != "volts":
                    raise ValueError(f"sweep {sn}: response unit {ts.unit!r} is not volts")
                if i_unit is not None and i_unit != "amperes":
                    raise ValueError(f"sweep {sn}: stimulus unit {i_unit!r} is not amperes")
                bias = _optional(ts, "bias_current")
                if bias is not None and abs(bias) > 1e-6:
                    flags.append("bias_current magnitude implies pA although NWB defines the field in amperes")
                bridge = _optional(ts, "bridge_balance")
                if bridge is not None and 0 < abs(bridge) < 1e3:
                    flags.append("bridge_balance magnitude implies megohms although NWB defines the field in ohms")
                capcomp = _optional(ts, "capacitance_compensation")
                if capcomp is not None and abs(capcomp) > 1e-6:
                    flags.append("capacitance_compensation magnitude implies pF although NWB defines the field "
                                 "in farads")
                if ts.starting_time not in (None, 0.0):
                    flags.append(f"starting_time {ts.starting_time} s not zero")
                meta = {
                    "sweep_number": sn,
                    "protocol_code": code,
                    "voltage_conversion": v_conv, "voltage_offset": v_off,
                    "current_conversion": i_conv, "current_offset": i_off,
                    "bias_current_as_stored": bias,
                    "bridge_balance_as_stored": bridge,
                    "capacitance_compensation_as_stored": capcomp,
                    "gain": _optional(ts, "gain"),
                    "starting_time": start_time,
                }
                sweeps.append(Sweep(cell_id=cell_id, sweep_id=sn, source_asset_id=asset_id, protocol_name=code,
                                    sampling_rate_hz=rate, voltage_volts=v,
                                    injected_current_amperes=i, recording_metadata=meta,
                                    source_quality_flags=flags))
    sweeps.sort(key=lambda s: s.sweep_id)
    return cell, sweeps
