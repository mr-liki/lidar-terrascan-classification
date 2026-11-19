"""Offline runtime owned by the plugin; no host imports or downloads."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import stat
import sys
import subprocess
import zipfile

BUNDLE_DIR = Path(__file__).resolve().parent.parent / "runtime"

# Avoid rewriting a loaded Windows extension unless its prefix actually changes.
WINDOWS_UNPACK = r"""
from pathlib import Path
import runpy, sys
script = Path(sys.argv[1]).resolve()
prefix = script.parent.parent
namespace = runpy.run_path(str(script), run_name='naksha_conda_relocation')
for relative, placeholder, mode in namespace['_prefix_records']:
    path = (prefix / relative).resolve()
    if not path.is_relative_to(prefix):
        raise RuntimeError('Unsafe conda relocation record: ' + relative)
    original = path.read_bytes()
    replacement = namespace['replace_prefix'](original, mode, placeholder, prefix.as_posix())
    if replacement != original:
        path.write_bytes(replacement)
"""

class SetupCancelled(RuntimeError):
    pass


def descriptor():
    path = BUNDLE_DIR / 'bundle.json'
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    digest = data.get('sha256', '')
    if (data.get('schema') != 1 or len(digest) != 64 or
            any(c not in '0123456789abcdef' for c in digest) or
            not isinstance(data.get('unpacked_bytes'), int) or data['unpacked_bytes'] <= 0):
        raise RuntimeError('Invalid AI3 plugin runtime descriptor. Reinstall the plugin.')
    return data


def install_root(data=None):
    from .runtime import user_data_dir
    data = data or descriptor()
    return user_data_dir() / 'plugin-runtimes' / data['sha256'][:16]


def installed_python(data=None):
    data = data or descriptor()
    root = install_root(data)
    try:
        record = json.loads((root/'runtime.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    python = Path(record.get('python', '')) if record.get('source') == 'existing' else root/'env'/'python.exe'
    if record.get('archive_sha256') == data['sha256'] and record.get('setup_complete') and python.is_file():
        return python
    return None



def runtime_candidates():
    """Known AI3 environments and configured interpreters; never import host AI code."""
    from .runtime import runtime_root, read_json, user_data_dir
    candidates = []
    if os.environ.get('NAKSHA_AI3_PYTHON'):
        candidates.append(Path(os.environ['NAKSHA_AI3_PYTHON']).expanduser())
    local = user_data_dir().parent
    roots = [runtime_root(), local/'NakshaAI'/'ai3_runtime']
    versions = local/'NakshaAI-LiDAR'/'runtimes'/'ai3'
    if versions.is_dir():
        roots.extend(sorted(versions.iterdir()))
    private = user_data_dir()/'plugin-runtimes'
    if private.is_dir():
        roots.extend(sorted(private.iterdir()))
    for root in roots:
        record = read_json(root/'runtime.json')
        if record.get('python'):
            candidates.append(Path(record['python']))
        candidates.append(root/'env'/'python.exe')
    if os.environ.get('CONDA_PREFIX'):
        candidates.append(Path(os.environ['CONDA_PREFIX'])/'python.exe')
    if not getattr(sys, 'frozen', False):
        candidates.append(Path(sys.executable))
    located = shutil.which('python.exe')
    if located:
        candidates.append(Path(located))
    seen = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate.resolve()))
        if key not in seen and candidate.is_file():
            seen.add(key)
            yield candidate


def reuse_existing(root, data, progress, check_cancel):
    from .runtime import verify_runtime, write_json
    for candidate in runtime_candidates():
        check_cancel()
        progress(0, f'Checking existing Python runtime: {candidate}')
        status = verify_runtime(python=candidate, timeout=120)
        details = status.details.get('selftest_details', {})
        # Native PyG extensions require the same torch/numpy ABI as the bundled engine.
        compatible = (str(details.get('torch', '')).split('+')[0] == '2.4.1' and
                      details.get('numpy') == '1.26.4' and details.get('torch_geometric') == '2.5.3')
        if status.ok and compatible:
            check_cancel()
            write_json(root/'runtime.json', dict(schema=1, setup_complete=True,
                source='existing', archive_sha256=data['sha256'], python=str(candidate.resolve())))
            progress(100, 'Compatible AI3 runtime found. Installation skipped.')
            return candidate.resolve()
    return None


@contextmanager
def setup_lock(root):
    with (root/'setup.lock').open('a+b') as handle:
        handle.seek(0)
        handle.write(b'0')
        handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('AI3 runtime setup is already running. Wait, then retry.') from exc
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def checked_target(destination, entry):
    name = entry.filename
    windows = PureWindowsPath(name)
    target = (destination / name).resolve()
    if (windows.drive or windows.root or '..' in windows.parts or
            not target.is_relative_to(destination.resolve()) or
            stat.S_ISLNK(entry.external_attr >> 16)):
        raise RuntimeError('Unsafe AI3 runtime archive entry: ' + name)
    return target


def install(progress=None, cancel=None, detect_existing=True, force_bundled=False):
    from .runtime import process_env, no_window_flags, write_json
    progress = progress or (lambda *_: None)
    data = descriptor()
    if data is None:
        raise RuntimeError('This plugin is missing its offline runtime. Reinstall the full plugin ZIP.')
    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise SetupCancelled('AI3 runtime setup was cancelled. Run again to retry.')
    ready = installed_python(data)
    if ready and not force_bundled:
        return ready
    root = install_root(data).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with setup_lock(root):
        ready = installed_python(data)
        if ready and not force_bundled:
            return ready
        check_cancel()
        if detect_existing and not force_bundled:
            existing = reuse_existing(root, data, progress, check_cancel)
            if existing is not None:
                return existing
        destination = root/'env'
        # Only this plugin's content-addressed environment may be recovered/deleted.
        if destination.resolve().parent != root or destination.is_symlink():
            raise RuntimeError('Unsafe AI3 runtime destination')
        needed = data['unpacked_bytes'] + 1024**3
        if shutil.disk_usage(root).free < needed:
            raise RuntimeError(f'AI3 runtime setup needs {needed/1024**3:.1f} GiB free in {root}.')
        progress(0, 'Checking the bundled AI3 runtime (first run only)...')
        digest = hashlib.sha256()
        archive = BUNDLE_DIR/'environment.zip'
        with archive.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8*1024*1024), b''):
                check_cancel()
                digest.update(chunk)
        if digest.hexdigest() != data['sha256']:
            raise RuntimeError('The AI3 runtime archive is damaged. Reinstall the plugin ZIP.')
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir()
        try:
            with zipfile.ZipFile(archive) as source:
                entries = source.infolist()
                for entry in entries:
                    checked_target(destination, entry)
                if sum(entry.file_size for entry in entries) != data['unpacked_bytes']:
                    raise RuntimeError('AI3 runtime archive size does not match its descriptor.')
                done = 0
                for index, entry in enumerate(entries):
                    check_cancel()
                    target = checked_target(destination, entry)
                    if entry.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(entry) as src, target.open('wb') as dst:
                        while True:
                            check_cancel()
                            chunk = src.read(8*1024*1024)
                            if not chunk:
                                break
                            dst.write(chunk)
                    done += entry.file_size
                    if index % 100 == 0:
                        progress(5 + int(80*done/data['unpacked_bytes']), 'Installing the bundled AI3 runtime (first run only)...')
            python = destination/'python.exe'
            unpack = destination/'Scripts'/'conda-unpack-script.py'
            if not python.is_file() or not unpack.is_file():
                raise RuntimeError('AI3 requires a Windows conda-pack runtime archive.')
            check_cancel()
            progress(88, 'Configuring the plugin Python runtime...')
            result = subprocess.run([str(python), '-c', WINDOWS_UNPACK, str(unpack)],
                env=process_env(python), cwd=str(destination), capture_output=True,
                text=True, encoding='utf-8', errors='replace', timeout=600,
                creationflags=no_window_flags())
            if result.returncode:
                raise RuntimeError('AI3 runtime configuration failed: ' + (result.stdout + result.stderr)[-4000:])
            check_cancel()
            write_json(root/'runtime.json', dict(schema=1, setup_complete=True,
                archive_sha256=data['sha256'], python=str(python)))
            progress(100, 'Plugin runtime installed. Checking the AI engine...')
            return python
        except BaseException:
            shutil.rmtree(destination, ignore_errors=True)
            raise
