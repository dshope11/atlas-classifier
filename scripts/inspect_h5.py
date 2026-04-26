"""Print the structure and field dtypes of an HDF5 file.

Usage:
    python scripts/inspect_h5.py data/processed/events.h5

Walks the HDF5 hierarchy and prints groups, datasets, and (for structured-array
datasets) the field names and dtypes. Assumes at most one level of group nesting
(sufficient for events.h5 and split.h5); deeper groups are not recursed into.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py


def inspect_dataset(name: str, dataset: h5py.Dataset) -> None:
    print(f"  {name}  shape={dataset.shape}  dtype={dataset.dtype}")
    if dataset.dtype.names is not None:
        print("    fields:")
        for field_name in dataset.dtype.names:
            field_dtype = dataset.dtype[field_name]
            print(f"      {field_name:<30s} {field_dtype}")


def inspect_file(path: Path) -> int:
    if not path.exists():
        print(f"Error: {path} does not exist", file=sys.stderr)
        return 1
    with h5py.File(path, "r") as f:
        print(f"File: {path}")
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Group):
                print(f"Group: {key}")
                for subkey in obj.keys():
                    sub = obj[subkey]
                    if isinstance(sub, h5py.Dataset):
                        inspect_dataset(f"{key}/{subkey}", sub)
            elif isinstance(obj, h5py.Dataset):
                inspect_dataset(key, obj)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to HDF5 file")
    args = parser.parse_args()
    return inspect_file(args.path)


if __name__ == "__main__":
    sys.exit(main())
