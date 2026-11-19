"""Run the AI model 3.0 engine as an isolated process and report its progress.

The caller never imports the pipeline: this starts
``engine/naksha_ai_engine.py`` with the runtime's own interpreter and reads the engine's stdout
protocol (see the module docstring of the engine):

    PROGRESS <0-100> <message>     RESULT {json}     WARNING <text>     ERROR <text> (stderr)

Exit codes: 0 ok, 1 bad arguments, 2 pipeline step failed, 3 environment, 130 cancelled.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import runtime as rt

ProgressCallback = Callable[[int, str], None]
_RUNTIME_LOCK = threading.Lock()


class Ai3Error(RuntimeError):
    """Engine failure with a message fit to show the user."""


class Ai3RuntimeMissing(Ai3Error):
    """The runtime environment is absent or unusable; the GUI can offer to (re)build it."""

    def __init__(self, message: str, state: str = "missing"):
        super().__init__(message)
        self.state = state


class Ai3Cancelled(Ai3Error):
    pass


@dataclass
class Ai3Result:
    output_path: Path
    report: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _log(message: str) -> None:
    """print() that cannot raise: a console that cannot encode "→" (cp1252), or no console at all
    (a windowed build), must never be able to kill a classification run over a log line."""
    try:
        print(message, flush=True)
    except Exception:
        try:
            print(message.encode("ascii", "replace").decode("ascii"), flush=True)
        except Exception:
            pass


def _no_window() -> int:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    # Own process group, so a console CTRL_BREAK can be sent to the engine alone.
    flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    return flags


class Ai3Controller:
    """Starts the engine, forwards its progress, and can cancel it."""

    def __init__(self, device: str = "auto"):
        self.device = device if device in ("auto", "cuda", "cpu") else "auto"
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._cancelled = False

    def spawn(self) -> "Ai3Controller":
        """A new controller with the same settings (one per engine run in parallel)."""
        return Ai3Controller(device=self.device)

    # ------------------------------------------------------------------ environment
    def ensure_ready(self, progress=None, cancel=None) -> rt.RuntimeStatus:
        """Serialize automatic install and first verification across plugin workers."""
        while not _RUNTIME_LOCK.acquire(timeout=0.2):
            if cancel is not None and cancel.is_set():
                raise Ai3Cancelled("AI3 runtime setup was cancelled.")
        try:
            return self._ensure_ready(progress, cancel)
        finally:
            _RUNTIME_LOCK.release()

    def _ensure_ready(self, progress=None, cancel=None) -> rt.RuntimeStatus:
        """Return a verified runtime, or raise Ai3RuntimeMissing with an actionable message."""
        from .offline_runtime import descriptor, install, SetupCancelled
        if descriptor() is not None:
            try:
                install(progress=progress, cancel=cancel)
            except SetupCancelled as exc:
                raise Ai3Cancelled(str(exc)) from exc
            except (OSError, RuntimeError, ValueError) as exc:
                raise Ai3RuntimeMissing(str(exc), "broken") from exc
        status = rt.runtime_status()
        if status.ok:
            return status
        if status.python is None or not status.engine_ok:
            raise Ai3RuntimeMissing(status.message, status.state)
        status = rt.verify_runtime(device=self.device)
        if not status.ok and descriptor() is not None:
            from .offline_runtime import install_root
            record = rt.read_json(install_root()/"runtime.json")
            if record.get("source") == "existing":
                # An external environment may have been changed since it was selected.
                try:
                    install(progress=progress, cancel=cancel, detect_existing=False, force_bundled=True)
                except SetupCancelled as exc:
                    raise Ai3Cancelled(str(exc)) from exc
                except (OSError, RuntimeError, ValueError) as exc:
                    raise Ai3RuntimeMissing(str(exc), "broken") from exc
                status = rt.verify_runtime(device=self.device)
        if not status.ok:
            raise Ai3RuntimeMissing(status.message, status.state)
        return status

    # ------------------------------------------------------------------ command
    def build_command(self, python: Path, input_path: Path, output_path: Path,
                      schema_path: Path, work_dir: Path, power_lines: bool,
                      schema_is_ptc: bool = True,
                      ground_unclass_pct: Optional[float] = None,
                      extra: Optional[List[str]] = None) -> List[str]:
        cmd = [str(python), str(rt.ENGINE_SCRIPT),
               "--input", str(input_path),
               "--output", str(output_path),
               "--ptc" if schema_is_ptc else "--mapping", str(schema_path),
               "--work-dir", str(work_dir),
               "--device", self.device]
        if not power_lines:
            cmd.append("--no-power-lines")
        if ground_unclass_pct is not None:
            cmd += ["--ground-unclass-pct", f"{float(ground_unclass_pct):g}"]
        return cmd + list(extra or [])

    @staticmethod
    def _engine_env(python) -> Dict[str, str]:
        return rt.process_env(python)

    # ------------------------------------------------------------------ run / cancel
    def run(self, input_path: Path, output_path: Path, ptc_path, work_dir: Path,
            power_lines: bool = True,
            progress_callback: Optional[ProgressCallback] = None,
            mapping: Optional[dict] = None,
            ground_unclass_pct: Optional[float] = None,
            overrides: Optional[dict] = None, keep_step2: Optional[Path] = None,
            step3_from: Optional[Path] = None,
            grid_origin: Optional[tuple] = None) -> Ai3Result:
        """Classify ``input_path``. ``ptc_path`` (the .ptc class-definition file)
        is the delivered class schema; ``mapping`` is a JSON-dict alternative used by tests.

        A tiled run passes file-wide contract ``overrides``, can keep the step 2 output
        (``keep_step2``) and later re-run only step 3 on it (``step3_from``)."""
        status = self.ensure_ready()
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_path)
        if mapping is not None:
            schema_path = work_dir / "ai3_mapping.json"
            with open(schema_path, "w", encoding="utf-8") as handle:
                json.dump(mapping, handle, indent=2, sort_keys=True)
        else:
            if not ptc_path or not Path(ptc_path).is_file():
                raise Ai3Error(f"AI model 3.0 needs the class-definition .ptc file: {ptc_path}")
            # a private copy: the run is then unaffected by the user touching the file later
            schema_path = work_dir / "input_classes.ptc"
            shutil.copyfile(str(ptc_path), str(schema_path))
        try:
            if output_path.exists():
                output_path.unlink()
        except OSError:
            pass

        extra: List[str] = []
        if overrides:
            overrides_path = work_dir / "contract_overrides.json"
            with open(overrides_path, "w", encoding="utf-8") as handle:
                json.dump(overrides, handle, indent=2, sort_keys=True)
            extra += ["--contract-overrides", str(overrides_path)]
        if keep_step2:
            extra += ["--keep-step2", str(keep_step2)]
        if step3_from:
            extra += ["--step3-from", str(step3_from)]
        cmd = self.build_command(status.python, Path(input_path), output_path, schema_path,
                                 work_dir / "engine_work", power_lines,
                                 schema_is_ptc=mapping is None,
                                 ground_unclass_pct=ground_unclass_pct, extra=extra)
        _log("[AI model 3.0] " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        engine_env = self._engine_env(status.python)
        if grid_origin is not None:
            # anchors the model's 50 m receptive fields on the whole file's grid (tiled runs)
            engine_env["NAKSHA_AI3_GRID_ORIGIN"] = f"{grid_origin[0]:.3f},{grid_origin[1]:.3f}"

        with self._lock:
            self._cancelled = False
            self._process = subprocess.Popen(
                cmd, cwd=str(rt.ENGINE_DIR), env=engine_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", creationflags=_no_window())
            process = self._process

        tail: deque = deque(maxlen=60)
        report: dict = {}
        warnings: List[str] = []
        percent = 0
        try:
            assert process.stdout is not None
            for raw in process.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                tail.append(line)
                if line.startswith("PROGRESS "):
                    parts = line.split(" ", 2)
                    try:
                        percent = max(percent, min(100, int(parts[1])))
                    except (IndexError, ValueError):
                        continue
                    if progress_callback:
                        progress_callback(percent, parts[2] if len(parts) > 2 else "")
                elif line.startswith("RESULT "):
                    try:
                        report = json.loads(line[len("RESULT "):])
                    except ValueError:
                        report = {}
                elif line.startswith("WARNING "):
                    warnings.append(line[len("WARNING "):])
                    if progress_callback:
                        progress_callback(percent, "Warning: " + line[len("WARNING "):])
                else:
                    _log(f"[AI model 3.0] {line}")
            code = process.wait()
        finally:
            if process.poll() is None:
                self._kill(process)
            with self._lock:
                self._process = None

        if self._cancelled or code == 130:
            raise Ai3Cancelled("AI model 3.0 classification was cancelled.")
        if code != 0:
            raise Ai3Error(self._explain(code, tail))
        if not output_path.is_file():
            raise Ai3Error("The AI model 3.0 engine finished but wrote no output file.")
        return Ai3Result(output_path=output_path, report=report, warnings=warnings)

    @staticmethod
    def _explain(code: int, tail) -> str:
        errors = [ln[len("ERROR "):] for ln in tail if ln.startswith("ERROR ")]
        reason = errors[-1] if errors else (list(tail)[-1] if tail else "no output")
        what = {1: "invalid input", 2: "the classification pipeline failed",
                3: "the runtime environment is not usable"}.get(code, f"exit code {code}")
        detail = "\n".join(list(tail)[-12:])
        return f"AI model 3.0: {what}: {reason}\n\nLast engine output:\n{detail}"

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            process = self._process
        if process is not None and process.poll() is None:
            self._kill(process)

    @staticmethod
    def _kill(process: subprocess.Popen) -> None:
        """End the engine and everything it started (step 2 runs a Myria3D child process)."""
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            else:
                process.kill()
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
