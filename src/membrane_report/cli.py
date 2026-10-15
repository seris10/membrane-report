"""Command-line interface: membrane-report {fetch,analyze,validate}."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def cmd_fetch(args):
    from .dandi import fetch_asset
    from .run import load_config

    cfg = load_config(args.config)
    root = Path(args.config).resolve().parent.parent
    for c in cfg["cells"]:
        dest = (root / c["path"]).parent
        path, _ = fetch_asset(c["asset_id"], dest, expected_sha256=c["sha256"])
        print(f"{path} ok")
    return 0


def cmd_analyze(args):
    from .data_io import read_selection_log
    from .report import render
    from .run import load_config, run, write

    cfg = load_config(args.config)
    root = Path(args.config).resolve().parent.parent
    if args.n_boot is not None:
        if args.n_boot < 2:
            raise ValueError("--n-boot must be at least two")
        cfg["analysis"]["n_boot"] = args.n_boot
    import pandas as pd

    from .synthetic import synthetic_checks

    cells, tables, manifest = run(cfg, root)
    tables["synthetic_checks"] = pd.DataFrame(synthetic_checks())
    out = Path(args.out)
    write(cells, tables, manifest, out)
    log = read_selection_log(root / cfg["selection"]["log"]) if cfg.get("selection", {}).get("log") else None
    (out / "report.html").write_text(render(cells, tables, manifest, log, __version__))
    for name in ("sweep_qc.csv", "sweep_features.csv", "cell_features.csv", "passive_sweeps.csv",
                 "reference_checks.csv", "source_manifest.json", "cells.json", "report.html"):
        print(out / name)
    contradicted = tables["reference_checks"].get("verdict", pd.Series(dtype=str)).eq("CONTRADICTED").any()
    return 1 if contradicted or not tables["synthetic_checks"]["passed"].all() else 0


def cmd_validate(args):
    from .synthetic import validate_synthetic

    ok = validate_synthetic(verbose=True)
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="membrane-report", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fetch", help="download the configured NWB files from DANDI and verify their sha256")
    s.add_argument("--config", required=True)
    s.set_defaults(func=cmd_fetch)
    s = sub.add_parser("analyze", help="QC, features, passive fits, reference checks and report")
    s.add_argument("--config", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-boot", type=int, default=None, help="override the bootstrap size")
    s.set_defaults(func=cmd_analyze)
    s = sub.add_parser("validate", help="run the synthetic fixtures with known answers")
    s.set_defaults(func=cmd_validate)
    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError, KeyError) as exc:
        print(f"{p.prog}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
