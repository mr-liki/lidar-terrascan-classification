"""The Naksha plugin ZIP: layout the Plugins Manager accepts, and a self-contained package."""
import json
import hashlib

import pytest
import zipfile
from pathlib import Path

import build_plugin_zip  # noqa: E402  (ai3_app on sys.path via conftest's parent)


@pytest.fixture
def runtime_bundle(tmp_path):
    folder = tmp_path / "bundle"
    folder.mkdir()
    archive = folder / "environment.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("python.exe", b"test interpreter")
        z.writestr("Scripts/conda-unpack-script.py", b"test relocation")
    with zipfile.ZipFile(archive) as z:
        size = sum(e.file_size for e in z.infolist())
    data = dict(schema=1, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), unpacked_bytes=size)
    (folder / "bundle.json").write_text(json.dumps(data), encoding="utf-8")
    return folder


def test_plugin_zip_layout(tmp_path, runtime_bundle):
    target = build_plugin_zip.build(tmp_path, runtime_bundle)
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        manifest_name = next(name for name in names if name.endswith("manifest.json"))
        assert manifest_name == "ai_model_3_0/manifest.json"
        manifest = json.loads(archive.read(manifest_name))
        assert {"name", "entry", "class"} <= set(manifest)
        assert f"ai_model_3_0/{manifest['entry']}" in names
        for required in ("naksha_ai3/job.py", "naksha_ai3/engine/naksha_ai_engine.py",
                         "naksha_ai3/engine/models/fractal_lidarhd_myria3d.ckpt",
                         "naksha_ai3_host.py", "naksha_ai3_window.py", "runtime/environment.zip",
                         "runtime/bundle.json", "naksha_ai3/offline_runtime.py"):
            assert f"ai_model_3_0/{required}" in names
        assert not [name for name in names if "__pycache__" in name or name.endswith(".pyc")]
        entry = archive.read(f"ai_model_3_0/{manifest['entry']}").decode()
        assert f"class {manifest['class']}(" in entry


def test_builder_rejects_corrupt_runtime(tmp_path, runtime_bundle):
    (runtime_bundle/"environment.zip").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="SHA-256"):
        build_plugin_zip.build(tmp_path, runtime_bundle)
