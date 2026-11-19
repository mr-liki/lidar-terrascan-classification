"""Plugin installation must be offline, private, relocatable and recoverable."""
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import zipfile
from types import SimpleNamespace

import pytest
from naksha_ai3 import offline_runtime as offline, runtime
from naksha_ai3.controller import Ai3Controller, Ai3Cancelled
from test_plugin_package import runtime_bundle


@pytest.fixture
def packaged(monkeypatch, tmp_path, runtime_bundle):
    monkeypatch.setattr(offline, "BUNDLE_DIR", runtime_bundle)
    monkeypatch.setattr(offline, "runtime_candidates", lambda: iter(()))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path/"user"))
    monkeypatch.setenv("NAKSHA_AI3_PYTHON", str(tmp_path/"host/python.exe"))
    monkeypatch.setenv("NAKSHA_AI3_RUNTIME", str(tmp_path/"host"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""))
    return runtime_bundle


def test_first_run_uses_only_own_runtime_and_reuses_it(packaged, monkeypatch):
    assert runtime.python_path() is None
    assert runtime.runtime_status().state == "setup_required"
    calls = []
    def relocate(command, **kwargs):
        calls.append(command)
        assert command[0].endswith("python.exe")
        assert command[2] == offline.WINDOWS_UNPACK
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", relocate)
    python = offline.install()
    assert "plugin-runtimes" in python.parts
    assert runtime.python_path() == python
    assert offline.install() == python
    assert len(calls) == 1


def test_corrupt_archive_cannot_install(packaged):
    (packaged/"environment.zip").write_bytes(b"damaged")
    with pytest.raises(RuntimeError, match="damaged"):
        offline.install()
    assert runtime.python_path() is None


def test_failed_relocation_is_cleaned_and_can_retry(packaged, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stdout="", stderr="failed"))
    with pytest.raises(RuntimeError, match="configuration failed"):
        offline.install()
    assert not (offline.install_root()/"env").exists()
    assert runtime.python_path() is None
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""))
    assert offline.install().is_file()


def test_cancellation_is_reported_by_controller(packaged):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Ai3Cancelled, match="cancelled"):
        Ai3Controller().ensure_ready(cancel=cancel)
    assert runtime.python_path() is None


def test_cancel_during_extraction_removes_partial_environment(packaged):
    cancel = threading.Event()
    def progress(percent, message):
        if percent >= 5:
            cancel.set()
    with pytest.raises(offline.SetupCancelled):
        offline.install(progress, cancel)
    assert not (offline.install_root()/"env").exists()


@pytest.mark.parametrize("name", ["../escape", "C:/escape", "C:escape", "\\\\server/escape", "a/../../escape"])
def test_reject_archive_path_escape(tmp_path, name):
    with pytest.raises(RuntimeError, match="Unsafe"):
        offline.checked_target(tmp_path, zipfile.ZipInfo(name))


def test_reject_symlinks(tmp_path):
    info = zipfile.ZipInfo("link")
    info.external_attr = 0o120777 << 16
    with pytest.raises(RuntimeError, match="Unsafe"):
        offline.checked_target(tmp_path, info)


def test_process_environment_removes_host_paths(monkeypatch, tmp_path):
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "GDAL_DATA", "PROJ_DATA", "PROJ_LIB"):
        monkeypatch.setenv(key, "host-only")
    monkeypatch.setenv("NUMBA_CACHE_LOCATOR_CLASSES", "gui.numba_frozen_cache.Locator")
    result = runtime.process_env(tmp_path/"python.exe")
    assert not any(result.get(key) == "host-only" for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "GDAL_DATA", "PROJ_DATA", "PROJ_LIB"))
    assert "NUMBA_CACHE_LOCATOR_CLASSES" not in result


def test_windows_relocation_changes_prefix_without_touching_unchanged_file(tmp_path):
    env = tmp_path/"env"
    scripts = env/"Scripts"
    scripts.mkdir(parents=True)
    payload = env/"prefix.txt"
    payload.write_text("old-prefix", encoding="utf-8")
    unpack = scripts/"conda-unpack-script.py"
    unpack.write_text("_prefix_records = [('prefix.txt', 'old-prefix', 'text')]\ndef replace_prefix(data, mode, placeholder, new_prefix):\n    return data.replace(placeholder.encode(), new_prefix.encode())\n", encoding="utf-8")
    import sys
    result = subprocess.run([sys.executable, "-c", offline.WINDOWS_UNPACK, str(unpack)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert payload.read_text(encoding="utf-8") == env.as_posix()


def test_verified_existing_runtime_skips_install(packaged, monkeypatch, tmp_path):
    existing = tmp_path/"existing"/"python.exe"
    existing.parent.mkdir()
    existing.write_bytes(b"existing compatible python")
    monkeypatch.setattr(offline, "runtime_candidates", lambda: iter([existing]))
    def verify(**kwargs):
        assert kwargs["python"] == existing
        return runtime.RuntimeStatus(ok=True, details={"selftest_details": {
            "torch": "2.4.1+cu121", "numpy": "1.26.4", "torch_geometric": "2.5.3"}})
    monkeypatch.setattr(runtime, "verify_runtime", verify)
    # No bundle read/extraction should be needed when the existing runtime passes.
    (packaged/"environment.zip").unlink()
    assert offline.install() == existing
    assert runtime.python_path() == existing
    assert not (offline.install_root()/"env").exists()


def test_incompatible_existing_runtime_installs_bundled_copy(packaged, monkeypatch, tmp_path):
    existing = tmp_path/"python.exe"
    existing.touch()
    monkeypatch.setattr(offline, "runtime_candidates", lambda: iter([existing]))
    monkeypatch.setattr(runtime, "verify_runtime", lambda **kwargs: runtime.RuntimeStatus(ok=False))
    result = offline.install()
    assert result != existing
    assert "plugin-runtimes" in result.parts


def test_forced_bundled_fallback_replaces_selected_external(packaged, tmp_path):
    existing = tmp_path/"external"/"python.exe"
    existing.parent.mkdir()
    existing.touch()
    data = offline.descriptor()
    root = offline.install_root(data)
    root.mkdir(parents=True)
    runtime.write_json(root/"runtime.json", dict(schema=1, source="existing", setup_complete=True,
                       archive_sha256=data["sha256"], python=str(existing)))
    assert runtime.python_path() == existing
    result = offline.install(force_bundled=True)
    assert result != existing and result.is_file()
    assert runtime.python_path() == result
