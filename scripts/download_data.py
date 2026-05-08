"""Download ROOT files for all SAMPLES into ``data/raw/``.

Uses ``atlasopenmagic`` to discover the correct HTTPS URL for each DSID in the
2to4lep skim of the 2025e-13tev-beta release, then streams each file to disk
with a progress bar. Files are named ``mc_<DSID>.root`` for stable lookup by
``src/data_loading.py``.

Usage:
    python scripts/download_data.py [--force]

``--force`` re-downloads even if the file already exists. Otherwise existing
files are skipped (idempotent re-runs are cheap).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import atlasopenmagic as atom
import requests
from tqdm import tqdm

# Make src.* importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sample_info import (  # noqa: E402
    ATLAS_OPENDATA_RELEASE,
    ATLAS_OPENDATA_SKIM,
    SAMPLES,
)

CHUNK_SIZE = 1024 * 1024  # 1 MB


def _get_url(dsid: int) -> str:
    """Resolve the HTTPS download URL for one DSID."""
    samples = atom.build_dataset({str(dsid): {"dids": [dsid]}}, skim=ATLAS_OPENDATA_SKIM, protocol="https")
    files = samples[str(dsid)]["list"]
    if not files:
        raise RuntimeError(f"atlasopenmagic returned no files for DSID {dsid}")
    url = files[0]
    # Strip simplecache:: prefix - we manage our own on-disk cache via data/raw
    if url.startswith("simplecache::"):
        url = url[len("simplecache::"):]
    return str(url)


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with a tqdm progress bar."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=dest.name, leave=True
        ) as bar:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the destination file exists",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
        help="Destination directory (default: data/raw/)",
    )
    args = parser.parse_args()

    print(f"Release: {ATLAS_OPENDATA_RELEASE}  Skim: {ATLAS_OPENDATA_SKIM}")
    atom.set_release(ATLAS_OPENDATA_RELEASE)

    failures = 0
    for key, info in SAMPLES.items():
        dest = args.raw_dir / f"mc_{info.dsid}.root"
        if dest.exists() and not args.force:
            size_mb = dest.stat().st_size / 1e6
            print(f"  [skip] {key:25s} DSID {info.dsid}  ({size_mb:.0f} MB cached)")
            continue
        try:
            url = _get_url(info.dsid)
        except Exception as exc:
            print(f"  [fail] {key:25s} DSID {info.dsid}  URL lookup error: {exc}")
            failures += 1
            continue
        try:
            print(f"  [get ] {key:25s} DSID {info.dsid}")
            _download(url, dest)
        except Exception as exc:
            print(f"  [fail] {key:25s} DSID {info.dsid}  download error: {exc}")
            failures += 1
            continue

    if failures:
        print(f"\n{failures} sample(s) failed.")
        return 1
    print("\nAll samples downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
