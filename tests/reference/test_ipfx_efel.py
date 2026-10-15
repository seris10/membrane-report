"""Agreement with IPFX and eFEL on the first selected DANDI 000020 cell (needs the pinned NWB file)."""

import json
import warnings
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((PROJECT / "examples" / "cells.json").read_text())
FIRST = CONFIG["cells"][0]
NWB = PROJECT / FIRST["path"]

pytestmark = [pytest.mark.data, pytest.mark.slow,
              pytest.mark.skipif(not NWB.exists(), reason="pinned NWB not present; run membrane-report fetch")]


def test_ipfx_ontology_classes_our_codes_as_long_squares():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ipfx.stimulus import StimulusOntology
        import ipfx

        onto_path = Path(ipfx.__file__).parent / "defaults" / "stimulus_ontology.json"
        onto = StimulusOntology(json.loads(onto_path.read_text()))
    from membrane_report.selection import LONG_SQUARE_CODES

    for code in LONG_SQUARE_CODES:
        stim = onto.find_one(code, tag_type="code")
        assert stim.has_tag("Long Square", tag_type="name"), code


def test_first_cell_agrees_with_references():
    from membrane_report.analysis import analyze_cell
    from membrane_report.dandi import sha256_file
    from membrane_report.run import compare_with_references

    assert sha256_file(NWB) == FIRST["sha256"]
    res = analyze_cell(NWB, FIRST["asset_id"], n_boot=10, seed=1)
    rows = compare_with_references(res)
    tested = [r for r in rows if r["verdict"] in ("SUPPORTED", "CONTRADICTED", "NOT TESTED")]
    assert len(tested) >= 10
    bad = [r for r in tested if r["verdict"] != "SUPPORTED"]
    assert not bad, bad
    assert res["passive_status"] == "estimated"
    assert res["ode_check"]["max_abs_difference_v"] < 1e-8
