#!/usr/bin/env python3
"""Repack an extracted PyTorch checkpoint archive into a single file.

Some download/extraction tools unpack a `.pth` / `.pth.tar` file into a
directory that contains an `archive/` folder with `data.pkl`, `version`, and a
large `data/` directory. PyTorch expects those files to live inside a zip-style
checkpoint file, not as plain directories on disk.

Example:
    python scripts/repack_torch_checkpoint.py ^
        "C:\\path\\to\\00000189-checkpoint.pth" ^
        --output "C:\\path\\to\\pretrained_ckpts\\facevid2vid\\00000189-checkpoint.pth.tar"
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path


def resolve_source_root(source: Path) -> Path:
    """Return the directory whose top-level contents should be archived."""
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    if source.is_dir() and (source / "archive").is_dir():
        return source

    if source.is_dir() and source.name == "archive":
        return source.parent

    raise ValueError(
        "Source must be either:\n"
        "  1. A directory that contains an 'archive' folder, or\n"
        "  2. The 'archive' directory itself."
    )


def validate_source(source_root: Path) -> list[Path]:
    archive_dir = source_root / "archive"
    required = [
        archive_dir / "data.pkl",
        archive_dir / "version",
        archive_dir / "data",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required archive contents:\n{missing_text}")

    files = sorted(path for path in source_root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"No files found under: {source_root}")
    return files


def build_output_path(source_root: Path, output: Path | None) -> Path:
    if output is not None:
        return output
    return source_root.parent / f"{source_root.name}.tar"


def repack(source_root: Path, output_path: Path) -> int:
    files = validate_source(source_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    os.close(fd)
    temp_output_path = Path(temp_name)

    total_size = 0
    try:
        with zipfile.ZipFile(
            temp_output_path,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
            strict_timestamps=False,
        ) as zf:
            for file_path in files:
                arcname = file_path.relative_to(source_root).as_posix()
                zf.write(file_path, arcname)
                total_size += file_path.stat().st_size
        temp_output_path.replace(output_path)
    except Exception:
        temp_output_path.unlink(missing_ok=True)
        raise

    return total_size


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repack an extracted PyTorch checkpoint into a .pth/.pth.tar file."
    )
    parser.add_argument(
        "source",
        help="Directory containing the extracted archive, or the archive directory itself.",
    )
    parser.add_argument(
        "--output",
        help="Destination checkpoint file path. Defaults to '<source_dir>.tar'.",
    )
    args = parser.parse_args()

    source_root = resolve_source_root(Path(args.source).expanduser())
    output_path = build_output_path(
        source_root, Path(args.output).expanduser() if args.output else None
    )

    if output_path.exists():
        print(f"Refusing to overwrite existing file: {output_path}", file=sys.stderr)
        return 2

    total_size = repack(source_root, output_path)
    print(f"Created: {output_path}")
    print(f"Packed {total_size / (1024 * 1024):.1f} MiB from {source_root}")
    print("Next step: point the repo at this new .pth.tar file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
