#!/usr/bin/env python3
"""Build the Blender install ZIP and GitHub source ZIP using only the stdlib."""

import argparse
from pathlib import Path
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "mesh2scan"


def build_extension(destination):
    manifest = tomllib.loads((RUNTIME / "blender_manifest.toml").read_text())
    assert manifest["id"] == "mesh2scan"
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"mesh2scan-{manifest['version']}.zip"
    # Explicit allowlist keeps tests, developer environments, personal profiles
    # and source ZIPs out of Blender's extension archive.
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(RUNTIME.glob("*.py")):
            archive.write(path, path.name)
        archive.write(RUNTIME / "blender_manifest.toml", "blender_manifest.toml")
        archive.write(ROOT / "LICENSE", "LICENSE")
    return output


def build_source(destination):
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "mesh2scan-repository.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if any(
                p in {".git", "__pycache__", ".venv", "dist", ".ruff_cache"} for p in relative.parts
            ):
                continue
            if path.suffix in {".pyc", ".zip", ".blend", ".blend1"}:
                continue
            archive.write(path, Path("mesh2scan") / relative)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument(
        "--source", action="store_true", help="Also create a GitHub-ready source archive"
    )
    args = parser.parse_args()
    print(build_extension(args.output_dir))
    if args.source:
        print(build_source(args.output_dir))


if __name__ == "__main__":
    main()
