"""``naksha-ai3`` command line.

Examples::

    naksha-ai3 --input tile.laz --ptc classes.ptc
    naksha-ai3 --input a.laz b.laz c.laz --ptc classes.ptc --output-dir D:/out      (batch)
    naksha-ai3 --input tile.laz --ptc classes.ptc --fences fences.json --learn
    naksha-ai3 --job job.json        (a launcher's job file; see ``job_from_file``)
    naksha-ai3 --check               (verify the runtime)

Machine-readable lines on stdout (for launchers), next to the readable log:

    PROGRESS <0-100> <message>       progress of the current file
    FILE <index> <count> <path>      start of a file in a batch
    RESULT {json}                    one per file: its summary, or {"status": "FAILED", ...}

Exit codes: 0 all files done, 1 bad arguments, 2 a file failed, 3 runtime unusable,
130 cancelled.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import __version__, runtime
from .controller import Ai3Controller
from .fences import FenceError, load_polygons
from .job import Ai3Job, run_batch

EXIT_OK, EXIT_ARGS, EXIT_FAILED, EXIT_RUNTIME, EXIT_CANCELLED = 0, 1, 2, 3, 130


def _say(line: str) -> None:
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)


def job_from_file(path: Path) -> List[Ai3Job]:
    """Jobs described by a launcher's JSON job file.

    ``{"inputs": [...], "ptc": ..., "output_dir": ..., "fences": <fences.json> | "polygons": [...],
    "targets": <.npy of point indices>, "classes": <.npy of the current classification>,
    "learn": bool, "power_lines": bool, "ground_pct": float, "class_mapping": {...},
    "power_mapping": {...}, "optional_codes": {...}, "device": "auto", "origin": "..."}``

    ``targets`` and ``classes`` apply to a single input only.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    inputs = data.get("inputs") or ([data["input"]] if data.get("input") else [])
    if not inputs:
        raise ValueError("The job file names no input LAS/LAZ.")
    base = Path(path).parent

    def resolve(value):
        return None if value in (None, "") else (Path(value) if Path(value).is_absolute()
                                                   else base / value)

    polygons = data.get("polygons")
    if polygons is None and data.get("fences"):
        polygons = load_polygons(resolve(data["fences"]))
    targets = np.load(resolve(data["targets"])) if data.get("targets") else None
    classes = np.load(resolve(data["classes"])) if data.get("classes") else None
    if (targets is not None or classes is not None) and len(inputs) != 1:
        raise ValueError("'targets' and 'classes' can only be given for a single input.")

    def int_keys(mapping):
        return None if mapping is None else {int(k): int(v) for k, v in mapping.items()}

    return [Ai3Job(source=resolve(item), ptc=resolve(data["ptc"]),
                   output_dir=resolve(data.get("output_dir")), polygons=polygons,
                   targets=targets, classes=classes, learn=bool(data.get("learn", False)),
                   power_lines=bool(data.get("power_lines", True)),
                   ground_pct=float(data.get("ground_pct", 20.0)),
                   class_mapping=int_keys(data.get("class_mapping")),
                   power_mapping=int_keys(data.get("power_mapping")),
                   optional_codes=data.get("optional_codes"),
                   device=str(data.get("device", "auto")),
                   origin=str(data.get("origin", "Naksha AI model 3.0")))
            for item in inputs]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="naksha-ai3",
                                     description="Naksha AI model 3.0 LiDAR classification")
    parser.add_argument("--version", action="version", version=f"naksha-ai3 {__version__}")
    parser.add_argument("--input", nargs="+", type=Path, help="LAS/LAZ file(s) to classify")
    parser.add_argument("--ptc", type=Path, help="class-definition .ptc file")
    parser.add_argument("--output-dir", type=Path,
                        help="output folder (default: AI3_Classified next to each input)")
    parser.add_argument("--fences", type=Path, help="fences JSON: classify only inside them")
    parser.add_argument("--learn", action="store_true",
                        help="learn from already classified areas inside the fences")
    parser.add_argument("--ground-pct", type=float, default=20.0,
                        help="share of Ground delivered as Uncategorized (default 20)")
    parser.add_argument("--no-power-lines", action="store_true", help="skip wire/pole extraction")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--job", type=Path, help="JSON job file written by a launcher")
    parser.add_argument("--check", action="store_true", help="verify the runtime and exit")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        status = runtime.verify_runtime(device=args.device)
        _say(f"{status.state}: {status.message}")
        _say(f"python={status.python}  device={runtime.gpu_description(status)}")
        return EXIT_OK if status.ok else EXIT_RUNTIME
    try:
        if args.job:
            jobs = job_from_file(args.job)
        else:
            if not args.input or not args.ptc:
                build_parser().error("--input and --ptc are required (or use --job)")
            if args.learn and not args.fences:
                build_parser().error("--learn needs --fences")
            polygons = load_polygons(args.fences) if args.fences else None
            jobs = [Ai3Job(source=path, ptc=args.ptc, output_dir=args.output_dir,
                           polygons=polygons, learn=args.learn,
                           power_lines=not args.no_power_lines, ground_pct=args.ground_pct,
                           device=args.device)
                    for path in args.input]
    except (OSError, ValueError, KeyError, FenceError) as exc:
        _say(f"ERROR {exc}")
        return EXIT_ARGS

    cancel = threading.Event()
    controller = Ai3Controller(device=jobs[0].device)

    def interrupt(*_):
        cancel.set()
        controller.cancel()
    signal.signal(signal.SIGINT, interrupt)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, interrupt)

    def progress(index, count, percent, message):
        _say(f"PROGRESS {percent} {message}")

    current = {"index": -1}

    def tracked(index, count, percent, message):
        if index != current["index"]:
            current["index"] = index
            _say(f"FILE {index + 1} {count} {jobs[index].source}")
        progress(index, count, percent, message)

    items = run_batch(jobs, controller, tracked, cancel)
    failed = False
    for item in items:
        if item.result is not None:
            _say("RESULT " + json.dumps(item.result.summary()))
        else:
            failed = True
            _say("RESULT " + json.dumps({"status": "FAILED", "source": str(item.job.source),
                                         "error": item.error}))
    if cancel.is_set():
        return EXIT_CANCELLED
    if failed and any("runtime" in (item.error or "").lower() for item in items):
        return EXIT_RUNTIME
    return EXIT_FAILED if failed else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
