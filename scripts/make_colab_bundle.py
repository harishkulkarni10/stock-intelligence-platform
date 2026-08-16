"""Build a Colab-safe source+parquet ZIP with POSIX paths."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "stock-intelligence-colab.zip"
INCLUDE = (
    "src",
    "backend",
    "logger",
    "feature_store",
    "tests",
    "pyproject.toml",
    "README.md",
)
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".ruff_cache", "egg-info"}


def _should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS or part.endswith(".egg-info") or part.endswith(".pyc") for part in path.parts)


def build_bundle(output: Path = DEFAULT_OUT) -> Path:
    if output.exists():
        output.unlink()
    entries: list[str] = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in INCLUDE:
            path = ROOT / item
            if not path.exists():
                raise FileNotFoundError(f"Missing required path: {path}")
            if path.is_file():
                archive.write(path, path.relative_to(ROOT).as_posix())
                entries.append(path.relative_to(ROOT).as_posix())
                continue
            for file in path.rglob("*"):
                if not file.is_file() or _should_skip(file):
                    continue
                relative = file.relative_to(ROOT).as_posix()
                archive.write(file, relative)
                entries.append(relative)

    parquet = "feature_store/data/features.parquet"
    if parquet not in entries:
        raise FileNotFoundError(
            f"{parquet} missing from bundle — run `sip-data build` before packaging"
        )
    if any("\\" in name for name in entries):
        raise RuntimeError("Bundle contains Windows path separators")
    print(f"wrote {output} ({len(entries)} entries)")
    print(f"parquet: {parquet}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build_bundle(args.output)


if __name__ == "__main__":
    main()
