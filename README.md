# Membrane Report

Work in progress; development and validation are ongoing.

Python analysis of NWB current-clamp recordings: stimulus segmentation, sweep quality control, electrophysiological features, and passive membrane-model fitting. The selected dataset contains five cells from Allen Institute mouse visual cortex recordings, DANDI 000020 version 0.210913.1639.

## Install and run

Python 3.13 is the tested baseline; package metadata requires Python 3.11 or later. Run from the repository root:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-tested.txt .
.venv/bin/membrane-report validate
.venv/bin/python -m pytest -W error
```

Synthetic checks run without a data download. Raw NWB recordings are not bundled. To analyze the selected recordings:

```sh
.venv/bin/membrane-report fetch --config examples/cells.json
.venv/bin/membrane-report analyze --config examples/cells.json --out reports/example
```

Open `reports/example/report.html` to inspect the generated analysis. Downloads are verified against the archive SHA-256 values in the configuration. `tools/select_cells.py` repeats the protocol-based asset selection with explicit cache and output arguments.

## Methods and outputs

The adapter applies NWB conversion and offset metadata once, validates stimulus/response alignment, and rejects irregular time grids. Quality control measures noise and baseline drift in defined windows; absent or invalid measurements fail the relevant check. Features include baseline voltage, deflection, sag ratio, spike count, rheobase, and firing-rate/current slope.

A single-compartment passive model is fitted to eligible sweeps at the two smallest hyperpolarizing amplitudes and evaluated on larger held-out steps. Bootstrap intervals resample within-sweep residual blocks. IPFX and eFEL provide comparisons for compatible feature definitions; independent ODE integration checks the passive solution.

Outputs include `sweep_qc.csv`, `sweep_features.csv`, `cell_features.csv`, `passive_sweeps.csv`, `reference_checks.csv`, `synthetic_checks.csv`, `source_manifest.json`, `cells.json`, and `report.html`. Missing numerical results use JSON `null`. Invalid input exits with status 2; failed numerical/reference checks exit with status 1 after saving outputs.

`membrane_report.hh.simulate_hh` also implements a separate single-compartment squid-axon Hodgkin–Huxley model. It supports current steps, paired pulses, and temperature-scaled gate kinetics. Its temperature parameter does not describe the mouse recording conditions.

## Verification and limits

The default test selection excludes the network download test. Three empirical-data tests skip when the pinned NWB files are absent; fetch the recordings to run them. Tests cover NWB metadata conversions, segmentation, quality-control failure paths, passive fitting, synthetic features, and Hodgkin–Huxley numerical behavior.

These five protocol-selected cells support descriptive within-recording comparisons. Recording temperature, access resistance, and electrode resistance are absent from the source metadata. The linear passive fit does not identify individual channel conductances or represent amplitude-dependent responses. Its bootstrap omits between-sweep and between-cell variability. Population inference and ATP-demand estimation are outside this analysis.

## Sources and licensing

Code uses [BSD-3-Clause](LICENSE). Source recordings are Allen Institute for Brain Science (2021), *Patch-seq recordings from mouse visual cortex*, [DANDI 000020 version 0.210913.1639](https://doi.org/10.48324/dandi.000020/0.210913.1639), [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). The selection log, dataset attribution, asset identifiers, checksums, and transformation details are retained in [data_manifest.json](data_manifest.json) and [data/selection_log.csv](data/selection_log.csv). Installed dependencies retain their own licenses.

The related [Model Atlas](https://github.com/seris10/model-atlas) displays selected voltage traces and the separate Hodgkin–Huxley simulation at [atlas.relay70.dev](https://atlas.relay70.dev/#membrane).
