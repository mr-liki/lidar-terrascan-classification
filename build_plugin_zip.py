"""Build the AI Model 3.0 Naksha plugin ZIP (Plugins > Manager > Install from ZIP).

    python ai3_app/build_plugin_zip.py --runtime-bundle <folder> [--out <folder>]

The ZIP holds one folder with manifest.json, the plugin modules (naksha_plugin/) and the
naksha_ai3 package with its engine and model (src/naksha_ai3/). Bytecode caches are left out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN_SRC = HERE / "naksha_plugin"
PACKAGE_SRC = HERE / "src" / "naksha_ai3"
FOLDER = "ai_model_3_0"
PLUGIN_FILES = ("manifest.json", "plugin.py", "naksha_ai3_host.py", "naksha_ai3_window.py",
                "README.txt", "ai_model_3.svg")
FIXED_TIME = (2026, 1, 1, 0, 0, 0)       # reproducible archives


def _skip(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo")


def build(out_dir: Path, runtime_bundle: Path) -> Path:
    """Require a validated conda-pack runtime; never produce a runtime-less plugin."""
    runtime_bundle = Path(runtime_bundle)
    descriptor = json.loads((runtime_bundle / "bundle.json").read_text(encoding="utf-8"))
    runtime_archive = runtime_bundle / "environment.zip"
    digest = hashlib.sha256()
    with runtime_archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(chunk)
    if descriptor.get("schema") != 1 or descriptor.get("sha256") != digest.hexdigest():
        raise ValueError("Runtime bundle failed its SHA-256 check")
    with zipfile.ZipFile(runtime_archive) as archive:
        names = set(archive.namelist())
        if not {"python.exe", "Scripts/conda-unpack-script.py"} <= names:
            raise ValueError("A relocatable Windows conda-pack runtime is required")
        if sum(entry.file_size for entry in archive.infolist()) != descriptor.get("unpacked_bytes"):
            raise ValueError("Runtime bundle unpacked size does not match")
    manifest = json.loads((PLUGIN_SRC / "manifest.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(PACKAGE_SRC.parent))
    import naksha_ai3
    if manifest["version"] != naksha_ai3.__version__:
        raise SystemExit(f"manifest version {manifest['version']} != naksha_ai3 "
                         f"{naksha_ai3.__version__}")
    entries = [(PLUGIN_SRC / name, f"{FOLDER}/{name}") for name in PLUGIN_FILES]
    for path in sorted(PACKAGE_SRC.rglob("*")):
        if path.is_file() and not _skip(path.relative_to(PACKAGE_SRC)):
            entries.append((path, f"{FOLDER}/naksha_ai3/{path.relative_to(PACKAGE_SRC).as_posix()}"))
    entries.extend([(runtime_bundle / "bundle.json", f"{FOLDER}/runtime/bundle.json"),
                    (runtime_archive, f"{FOLDER}/runtime/environment.zip")])
    missing = [str(src) for src, _ in entries if not src.is_file()]
    if missing:
        raise SystemExit("Missing plugin files: " + ", ".join(missing))

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"AI_Model_3_Plugin_v{manifest['version']}.zip"
    tmp = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for src, name in entries:
            info = zipfile.ZipInfo(name, FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            # Stream multi-GB runtimes, including ZIP64 entries, without loading into RAM.
            info._compresslevel = 1
            with src.open("rb") as source, archive.open(info, "w", force_zip64=True) as dest:
                shutil.copyfileobj(source, dest, 8*1024*1024)
    tmp.replace(target)
    digest_hash = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            digest_hash.update(chunk)
    digest = digest_hash.hexdigest()
    print(f"{target}  ({len(entries)} files, {target.stat().st_size / 1e6:.1f} MB, sha256 {digest[:16]})")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE.parent / "plugins_zip")
    parser.add_argument("--runtime-bundle", type=Path, required=True,
                        help="Folder containing conda-pack environment.zip and bundle.json")
    args = parser.parse_args()
    build(args.out, args.runtime_bundle)
