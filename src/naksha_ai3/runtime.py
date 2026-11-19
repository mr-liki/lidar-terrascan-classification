"""Locate, verify and describe the Python environment that runs the AI3 engine.

The engine needs PDAL, torch 2.4, torch-geometric and numpy 1.26, so it runs in its own
environment. Resolution order for its interpreter:

1. ``NAKSHA_AI3_PYTHON``        an explicit python.exe (development machines, managed installs)
2. ``<app>/runtime/python.exe``  the runtime bundled with the installed AI3 application
3. ``<runtime root>/runtime.json`` field ``python``, else ``<runtime root>/env/python.exe``,
   where the runtime root is ``NAKSHA_AI3_RUNTIME`` or ``%LOCALAPPDATA%\\NakshaAI\\ai3_runtime``
   (the environment earlier Naksha versions set up per user)

Standard library only: safe to import before any heavy dependency is loaded.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PACKAGE_DIR = Path(__file__).resolve().parent
ENGINE_DIR = PACKAGE_DIR / "engine"
ENGINE_SCRIPT = ENGINE_DIR / "naksha_ai_engine.py"
# Installed layout: <app>/src/naksha_ai3 and <app>/runtime.
APP_DIR = PACKAGE_DIR.parents[1]
BUNDLED_RUNTIME = APP_DIR / "runtime"

ENV_RUNTIME = "NAKSHA_AI3_RUNTIME"
ENV_PYTHON = "NAKSHA_AI3_PYTHON"

RUNTIME_SCHEMA = 1
# Files whose content decides whether a recorded self-test is still valid.
_FINGERPRINT_FILES = (
    "naksha_ai_engine.py",
    "step1_preprocess.py",
    "step2_inference.py",
    "step3_rule_engine.py",
    "las_compat.py",
    "preserve_dimensions.py",
    os.path.join("myria3d", "myria3d", "models", "bounded_interpolation.py"),
    os.path.join("myria3d", "myria3d", "models", "model.py"),
    os.path.join("config", "contract_spec.json"),
    os.path.join("models", "inference_config.yaml"),
)
_ENGINE_FILES = (
    "naksha_ai_engine.py", "step1_preprocess.py", "step2_inference.py", "step3_rule_engine.py",
    "las_compat.py", "preserve_dimensions.py", os.path.join("config", "contract_spec.json"),
    os.path.join("models", "fractal_lidarhd_myria3d.ckpt"),
    os.path.join("models", "inference_config.yaml"), os.path.join("myria3d", "run.py"),
)


def no_window_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def user_data_dir() -> Path:
    """Per-user state of the AI3 application (self-test record, logs)."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "Naksha-AI3"


def runtime_root() -> Path:
    override = os.environ.get(ENV_RUNTIME)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "NakshaAI" / "ai3_runtime"


def _python_in(prefix: Path) -> Path:
    return prefix / ("python.exe" if os.name == "nt" else "bin/python")


def read_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def python_path() -> Optional[Path]:
    """The interpreter that should run the engine, or None when there is no candidate."""
    # A packaged plugin always uses its own runtime, independent of host overrides.
    from .offline_runtime import descriptor, installed_python
    packaged = descriptor()
    if packaged is not None:
        return installed_python(packaged)
    override = os.environ.get(ENV_PYTHON)
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None
    bundled = _python_in(BUNDLED_RUNTIME)
    if bundled.is_file():
        return bundled
    root = runtime_root()
    recorded = read_json(root / "runtime.json").get("python")
    if recorded and Path(recorded).is_file():
        return Path(recorded)
    candidate = _python_in(root / "env")
    return candidate if candidate.is_file() else None


def process_env(python) -> Dict[str, str]:
    """Environment for a child process of ``python``: only its own packages and libraries."""
    root = Path(python).parent
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(key, None)
    if 'gui.numba_frozen_cache.' in env.get('NUMBA_CACHE_LOCATOR_CLASSES', ''):
        env.pop('NUMBA_CACHE_LOCATOR_CLASSES', None)
    for key in ('GDAL_DATA', 'PROJ_DATA', 'PROJ_LIB'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    # PDAL/GDAL of the runtime must be found before any other GDAL on PATH.
    env["PATH"] = os.pathsep.join(map(str, (root, root / "Library" / "bin", root / "Scripts"))) \
        + os.pathsep + env.get("PATH", "")
    if (root / "Library" / "share" / "gdal").is_dir():
        env["GDAL_DATA"] = str(root / "Library" / "share" / "gdal")
    if (root / "Library" / "share" / "proj").is_dir():
        env["PROJ_DATA"] = str(root / "Library" / "share" / "proj")
    return env


def engine_fingerprint() -> str:
    """Hash of the engine files and the model checkpoint's size."""
    digest = hashlib.sha256()
    for rel in _FINGERPRINT_FILES:
        digest.update(rel.encode())
        try:
            digest.update((ENGINE_DIR / rel).read_bytes())
        except OSError:
            digest.update(b"<missing>")
    try:
        digest.update(str((ENGINE_DIR / "models" / "fractal_lidarhd_myria3d.ckpt").stat().st_size).encode())
    except OSError:
        digest.update(b"<missing>")
    return digest.hexdigest()


def engine_files_missing() -> List[str]:
    return [str(ENGINE_DIR / rel) for rel in _ENGINE_FILES if not (ENGINE_DIR / rel).is_file()]


def _selftest_record_path() -> Path:
    return user_data_dir() / "runtime_check.json"


@dataclass
class RuntimeStatus:
    ok: bool = False
    state: str = "missing"          # missing | unverified | ready | broken
    message: str = ""
    python: Optional[Path] = None
    engine_ok: bool = True
    details: dict = field(default_factory=dict)


def runtime_status(verify_cached: bool = True) -> RuntimeStatus:
    """Cheap status check (no subprocess): is there an interpreter, and was it verified?"""
    status = RuntimeStatus()
    missing = engine_files_missing()
    if missing:
        status.engine_ok = False
        status.state = "broken"
        status.message = "AI model 3.0 engine files are missing: " + ", ".join(missing)
        return status
    status.python = python_path()
    if status.python is None:
        from .offline_runtime import descriptor
        if descriptor() is not None:
            status.state = "setup_required"
            status.message = "The bundled AI3 runtime will be installed automatically on the first run (offline)."
        else:
            status.message = ("No AI model 3.0 runtime was found. Install the full plugin ZIP, or "
                              f"set {ENV_PYTHON} to the python.exe of an AI3 runtime.")
        return status
    record = read_json(_selftest_record_path()).get(str(status.python).lower(), {})
    status.details = record
    status.state = "unverified"
    status.message = "Runtime found; not verified for this engine version yet."
    if verify_cached and record.get("selftest_ok") and \
            record.get("engine_fingerprint") == engine_fingerprint():
        status.state = "ready"
        status.ok = True
        status.message = "AI model 3.0 runtime is ready."
    return status


def verify_runtime(timeout: float = 600.0, device: str = "auto", python=None) -> RuntimeStatus:
    """Run ``naksha_ai_engine.py --selftest`` and record the outcome per interpreter."""
    status = runtime_status(verify_cached=False)
    if python is not None:
        status.python = Path(python)
    if status.python is None or not status.engine_ok:
        return status
    cmd = [str(status.python), str(ENGINE_SCRIPT), "--selftest", "--device", device]
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(ENGINE_DIR), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout,
                              env=process_env(status.python), creationflags=no_window_flags())
        out, err, code = proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired:
        status.state = "broken"
        status.message = f"The AI model 3.0 self-test did not finish within {timeout:.0f} s."
        return status
    except OSError as exc:
        status.state = "broken"
        status.message = f"Could not start the AI model 3.0 runtime: {exc}"
        return status
    details = {}
    for line in out.splitlines():
        if line.startswith("SELFTEST ") and "=" in line:
            key, _, value = line[len("SELFTEST "):].partition("=")
            details[key.strip()] = value.strip()
    ok = code == 0 and any(line.strip() == "SELFTEST OK" for line in out.splitlines())
    record = {"schema": RUNTIME_SCHEMA, "selftest_ok": bool(ok),
              "selftest_seconds": round(time.time() - started, 1),
              "selftest_details": details,
              "engine_fingerprint": engine_fingerprint() if ok else ""}
    try:
        records = read_json(_selftest_record_path())
        records[str(status.python).lower()] = record
        write_json(_selftest_record_path(), records)
    except OSError:
        pass                                  # a read-only location must not break a good run
    status.details = record
    if ok:
        status.ok, status.state = True, "ready"
        status.message = "AI model 3.0 runtime is ready."
    else:
        status.state = "broken"
        reason = [ln[len("ERROR "):] for ln in err.splitlines() if ln.startswith("ERROR ")]
        status.message = ("The AI model 3.0 runtime failed its self-test: "
                          + ("; ".join(reason) if reason else (err.strip()[-500:] or f"exit {code}")))
    return status


def gpu_description(status: Optional[RuntimeStatus] = None) -> str:
    details = (status.details if status else {}).get("selftest_details", {})
    if details.get("cuda_available", "").lower() == "true":
        return details.get("cuda_device", "CUDA GPU")
    return "CPU (no CUDA GPU found - classification will be slow)"


if __name__ == "__main__":      # python -m naksha_ai3.runtime [--verify]
    st = verify_runtime() if "--verify" in sys.argv else runtime_status()
    print(f"{st.state}: {st.message}")
    print(f"python={st.python}")
    sys.exit(0 if st.ok else 1)
