"""Create a source archive without local caches or generated outputs."""

import argparse
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
OUTPUT_DIRECTORIES = {"results", "figures", "tables", "separability"}
EXCLUDED_DIRECTORIES = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", "third_party"}


def release_files():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if relative.parts[0] in OUTPUT_DIRECTORIES:
            continue
        if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
            continue
        if path.suffix.lower() in {".pyc", ".pyo", ".zip"}:
            continue
        if path.name in {".DS_Store", "Thumbs.db"}:
            continue
        if path.is_symlink():
            raise ValueError(f"Symlinks are not supported: {relative}")
        if path.is_file():
            yield path, relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "paper-code-anonymous.zip")
    args = parser.parse_args()
    files = list(release_files())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, relative in files:
            info = zipfile.ZipInfo(f"paper-code/{relative.as_posix()}")
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
        for path, relative in files:
            if archive.read(f"paper-code/{relative.as_posix()}") != path.read_bytes():
                raise RuntimeError(f"Archive content mismatch: {relative}")
    print(f"Created {args.output.name}: {len(files)} files")


if __name__ == "__main__":
    main()
