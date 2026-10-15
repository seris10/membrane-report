import json
from pathlib import Path

import pytest

from membrane_report.cli import main

PROJECT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((PROJECT / "examples" / "cells.json").read_text())


def test_validate_command_passes():
    assert main(["validate"]) == 0


@pytest.mark.slow
@pytest.mark.data
def test_analyze_one_cell(tmp_path):
    first = CONFIG["cells"][0]
    if not (PROJECT / first["path"]).exists():
        pytest.skip("pinned NWB not present")
    cfg = dict(CONFIG, cells=[first], analysis={"n_boot": 5, "seed": 1})
    cfg["selection"] = dict(CONFIG["selection"], log=str(PROJECT / CONFIG["selection"]["log"]))
    examples = tmp_path / "examples"
    examples.mkdir()
    (tmp_path / "data").symlink_to(PROJECT / "data")
    p = examples / "one.json"
    p.write_text(json.dumps(cfg))
    out = tmp_path / "out"
    assert main(["analyze", "--config", str(p), "--out", str(out)]) == 0
    for name in ("sweep_qc.csv", "cell_features.csv", "passive_sweeps.csv", "reference_checks.csv",
                 "source_manifest.json", "report.html", "synthetic_checks.csv"):
        assert (out / name).exists(), name
    manifest = json.loads((out / "source_manifest.json").read_text())
    assert manifest["assets"][0]["sha256"] == first["sha256"]
    assert "<svg" in (out / "report.html").read_text()


@pytest.mark.network
def test_fetch_verifies_checksum(tmp_path):
    from membrane_report.dandi import fetch_asset, sha256_file

    first = CONFIG["cells"][0]
    path, meta = fetch_asset(first["asset_id"], tmp_path, expected_sha256=first["sha256"])
    assert sha256_file(path) == first["sha256"]
    assert meta["digest"]["dandi:sha2-256"] == first["sha256"]


def test_json_outputs_preserve_boolean_types_and_use_null_for_missing_results(tmp_path):
    import numpy as np
    from membrane_report.run import write

    write([{"cell": {}, "bootstrap": {"valid": np.bool_(True), "missing": np.nan,
                                      "count": np.int64(2), "interval": [np.inf, -np.inf]}}], {}, {}, tmp_path)
    def reject(token):
        raise ValueError(token)
    record = json.loads((tmp_path / "cells.json").read_text(), parse_constant=reject)[0]["bootstrap"]
    assert record == {"valid": True, "missing": None, "count": 2, "interval": [None, None]}


def test_missing_reference_and_unestimated_passive_model_can_be_reported():
    import pandas as pd
    from membrane_report.report import render
    from membrane_report.run import compare_with_references

    cell = {"cell": {"cell_id": "fixture/cell", "source_asset_id": "fixture"}, "qc": [], "features": [],
            "passive_sweeps": [], "_sweeps": {}, "passive_status": "not estimated", "interpretation": []}
    refs = compare_with_references(cell)
    assert refs[0]["verdict"] == "NOT TESTED"
    tables = {"cell_features": pd.DataFrame([{"cell_id": "fixture/cell", "passive_status": "not estimated"}]),
              "reference_checks": pd.DataFrame(refs)}
    manifest = {"dataset": {"dandiset": "fixture", "version": "1", "citation": "synthetic fixture", "license": "test"},
                "assets": []}
    html = render([cell], tables, manifest)
    assert "No eligible passive fit" in html and "Not evaluated" in html


def test_reference_failure_changes_the_analyze_exit_status(tmp_path, monkeypatch):
    import pandas as pd
    import membrane_report.run as run_module
    import membrane_report.report as report_module
    import membrane_report.synthetic as synthetic_module

    cfg = tmp_path / "cells.json"
    cfg.write_text(json.dumps({"cells": [{}]}))
    tables = {"reference_checks": pd.DataFrame([{"verdict": "CONTRADICTED"}])}
    monkeypatch.setattr(run_module, "run", lambda *a: ([], tables, {}))
    monkeypatch.setattr(run_module, "write", lambda *a: a[-1].mkdir())
    monkeypatch.setattr(report_module, "render", lambda *a: "synthetic failure-path fixture")
    monkeypatch.setattr(synthetic_module, "synthetic_checks", lambda: [{"passed": True}])
    assert main(["analyze", "--config", str(cfg), "--out", str(tmp_path / "out")]) == 1
