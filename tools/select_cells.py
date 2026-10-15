"""Download DANDI 000020 assets newest session first and log which cells meet the selection criteria.

Usage: python tools/select_cells.py CACHE_DIR LOG_CSV [--want 5] [--max-assets 60]
"""

import argparse
import csv
import sys
from pathlib import Path

from membrane_report.dandi import fetch_asset, iter_assets
from membrane_report.recording import read_nwb
from membrane_report.selection import LONG_SQUARE_CODES, evaluate_cell, session_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir")
    ap.add_argument("log_csv")
    ap.add_argument("--want", type=int, default=5)
    ap.add_argument("--max-assets", type=int, default=60)
    a = ap.parse_args()
    fields = ["order", "asset_id", "path", "size", "sha256", "subject", "qualifies", "selected", "reasons",
              "subthreshold_amplitudes_pa", "depolarizing_amplitudes_pa", "n_incomplete_long_square_sweeps"]
    selected_subjects = set()
    n_sel = 0
    with open(a.log_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        assets = sorted(iter_assets(), key=lambda a: session_id(a["path"]), reverse=True)
        print(f"{len(assets)} assets listed", file=sys.stderr)
        for k, asset in enumerate(assets):
            if k >= a.max_assets or n_sel >= a.want:
                break
            subject = asset["path"].split("/")[0]
            row = {"order": k, "asset_id": asset["asset_id"], "path": asset["path"], "size": asset["size"],
                   "subject": subject}
            if subject in selected_subjects:
                row.update({"qualifies": "", "selected": False, "reasons": "C4: subject already selected"})
                w.writerow(row)
                fh.flush()
                continue
            path, meta = fetch_asset(asset["asset_id"], a.cache_dir)
            row["sha256"] = meta["digest"]["dandi:sha2-256"]
            try:
                cell, sweeps = read_nwb(path, asset["asset_id"], set(LONG_SQUARE_CODES))
                ev = evaluate_cell(sweeps, cell["stimulus_codes"])
            except Exception as exc:  # C1
                ev = {"qualifies": False, "reasons": [f"C1: {type(exc).__name__}: {exc}"],
                      "subthreshold_amplitudes_pa": [], "depolarizing_amplitudes_pa": [],
                      "n_incomplete_long_square_sweeps": ""}
            row.update({"qualifies": ev["qualifies"], "selected": ev["qualifies"], "reasons": "; ".join(ev["reasons"]),
                        "subthreshold_amplitudes_pa": " ".join(map(str, ev["subthreshold_amplitudes_pa"])),
                        "depolarizing_amplitudes_pa": " ".join(map(str, ev["depolarizing_amplitudes_pa"])),
                        "n_incomplete_long_square_sweeps": ev["n_incomplete_long_square_sweeps"]})
            if ev["qualifies"]:
                selected_subjects.add(subject)
                n_sel += 1
            w.writerow(row)
            fh.flush()
            print(k, asset["path"], "qualifies" if ev["qualifies"] else row["reasons"], flush=True)
    print(f"selected {n_sel}", file=sys.stderr)
    return 0 if n_sel else 1


if __name__ == "__main__":
    sys.exit(main())
