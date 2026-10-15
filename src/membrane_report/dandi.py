"""Read-only access to one published DANDI dandiset version: asset listing, metadata, verified download."""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.dandiarchive.org/api"
DANDISET = "000020"
VERSION = "0.210913.1639"
DOI = "10.48324/dandi.000020/0.210913.1639"
CITATION = ("Allen Institute for Brain Science (2021) Patch-seq recordings from mouse visual cortex "
            "(Version 0.210913.1639) [Data set]. DANDI archive. https://doi.org/10.48324/dandi.000020/0.210913.1639")
LICENSE = "CC-BY-4.0"


def _get_json(url: str, timeout=60) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "membrane-report/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def iter_assets(dandiset=DANDISET, version=VERSION, page_size=100):
    """Assets of the published version in path order: dicts with asset_id, path, size."""
    q = urllib.parse.urlencode({"order": "path", "page_size": page_size})
    url = f"{API}/dandisets/{dandiset}/versions/{version}/assets/?{q}"
    while url:
        page = _get_json(url)
        yield from page["results"]
        url = page.get("next")


def asset_metadata(asset_id: str) -> dict:
    return _get_json(f"{API}/assets/{asset_id}/")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_asset(asset_id: str, dest_dir: str | Path, expected_sha256: str | None = None) -> tuple[Path, dict]:
    """Download and SHA-256-verify one asset unless a verified copy exists; returns (path, metadata)."""
    meta = asset_metadata(asset_id)
    digest = expected_sha256 or meta["digest"]["dandi:sha2-256"]
    dest = Path(dest_dir) / Path(meta["path"]).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_file(dest) == digest:
        return dest, meta
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = f"{API}/assets/{asset_id}/download/"
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    got = sha256_file(tmp)
    if got != digest:
        tmp.unlink()
        raise ValueError(f"asset {asset_id}: sha256 {got} does not match recorded {digest}")
    tmp.replace(dest)
    return dest, meta
