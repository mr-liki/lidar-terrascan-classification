"""One AI3 classification job (full file or fences), and batches of jobs.

Full-file mode
    The engine classifies the whole source LAS/LAZ; the deliverable is the source file with
    only the classification replaced.

Fence mode
    The pipeline needs surrounding points (ground height, 50 m inference tiles), so it runs on a
    LOCAL cropped copy: the bounding box of the fence points plus a context buffer. Only the
    points inside the fences are merged back; every other point keeps its current class. A fence
    result is therefore not guaranteed identical to a full-file run.

    With ``learn`` set, already classified areas inside the fences are kept unchanged and used
    as reference for the rest (see ``reference.py``).

The current classification normally comes from the source file. A host application that holds
unsaved edits (Naksha) passes them as ``classes``, and may pass its resident ``xyz`` to avoid
reading the file twice.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from . import __version__, lasio
from .controller import Ai3Cancelled, Ai3Controller, Ai3Error
from .fences import FenceError, normalize_polygons, points_in_fences
from .mapping import build_engine_mapping, mappings_from_ptc
from .reference import FenceLearning
from . import tiling

ProgressCallback = Callable[[int, str], None]

FENCE_CONTEXT_BUFFER_M = float(os.environ.get("NAKSHA_AI3_FENCE_BUFFER_M", "50.0"))
MIN_ENGINE_POINTS = 5_000          # below this the pipeline has too little to work with


@dataclass
class Ai3Job:
    source: Path
    ptc: Path
    output_dir: Optional[Path] = None          # default: <source folder>/AI3_Classified
    polygons: Optional[Sequence] = None        # fence polygons in the cloud's XY
    targets: Optional[np.ndarray] = None       # or fence point indices (source order)
    classes: Optional[np.ndarray] = None       # current classification (default: the file's)
    xyz: Optional[np.ndarray] = None           # source-order XYZ (default: read from the file)
    learn: bool = False                        # learn from classified areas inside the fences
    power_lines: bool = True
    ground_pct: float = 20.0                   # Ground -> Uncategorized share when not learned
    class_mapping: Optional[Dict[int, int]] = None   # default: resolved from the .ptc
    power_mapping: Optional[Dict[int, int]] = None
    optional_codes: Optional[Dict[str, int]] = None
    fence_buffer_m: float = FENCE_CONTEXT_BUFFER_M
    device: str = "auto"
    origin: str = "Naksha AI model 3.0"
    work_root: Optional[Path] = None           # scratch folder (default: a local work folder);
                                               # always removed at the end of the run
    tile_points: int = tiling.TILE_TARGET_POINTS   # full file: points per tile
    tile_workers: int = tiling.TILE_WORKERS        # full file: tiles classified at once
    keep_final_classes: bool = True            # full file: also return every class in memory
                                               # (1 byte per point); off for huge files

    @property
    def fence_mode(self) -> bool:
        return self.polygons is not None or self.targets is not None


@dataclass
class Ai3JobResult:
    source: Path
    output_path: Path
    report_path: Path
    fence_mode: bool
    points_total: int
    points_processed: int
    engine_points: int
    ground_pct: float
    final_classes: Optional[np.ndarray]
    target_indices: Optional[np.ndarray] = None
    target_classes: Optional[np.ndarray] = None
    warnings: List[str] = field(default_factory=list)
    engine_report: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"status": "COMPLETED", "source": str(self.source), "output": str(self.output_path),
                "report": str(self.report_path), "fence_mode": self.fence_mode,
                "points_total": self.points_total, "points_processed": self.points_processed,
                "engine_input_points": self.engine_points, "ground_pct": self.ground_pct,
                "warnings": list(self.warnings)}


class _Run:
    """State of one job run: cancellation and progress."""

    def __init__(self, controller, progress: Optional[ProgressCallback],
                 cancel: Optional[threading.Event]):
        self.controller = controller
        self.progress = progress or (lambda percent, message: None)
        self.cancel_event = cancel or threading.Event()

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise Ai3Cancelled("AI model 3.0 classification was cancelled.")

    def scaled(self, lo: float, hi: float) -> ProgressCallback:
        def emit(percent: int, message: str) -> None:
            self.progress(int(lo + (hi - lo) * max(0, min(100, percent)) / 100.0), message)
        return emit


def _codes(job: Ai3Job):
    class_mapping, power_mapping, optional = job.class_mapping, job.power_mapping, job.optional_codes
    if class_mapping is None or power_mapping is None or optional is None:
        ptc_classes, ptc_power, ptc_optional, _ = mappings_from_ptc(job.ptc)
        class_mapping = ptc_classes if class_mapping is None else class_mapping
        power_mapping = ptc_power if power_mapping is None else power_mapping
        optional = ptc_optional if optional is None else optional
    config = {"_ptc_optional_semantic_codes": dict(optional)}
    return dict(class_mapping), dict(power_mapping), config


def _write_report(path: Path, **fields) -> None:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(fields, handle, indent=2, sort_keys=True, default=str)
    except OSError:
        pass


def run_job(job: Ai3Job, controller: Optional[Ai3Controller] = None,
            progress: Optional[ProgressCallback] = None,
            cancel: Optional[threading.Event] = None) -> Ai3JobResult:
    """Classify one file. Raises Ai3Error / Ai3Cancelled / FenceError with a user message."""
    run = _Run(controller or Ai3Controller(device=job.device), progress, cancel)
    source = Path(job.source).resolve()
    if not source.is_file():
        raise Ai3Error(f"Source LAS/LAZ not found: {source}")
    if not job.ptc or not Path(job.ptc).is_file():
        raise Ai3Error(f"AI model 3.0 needs the class-definition .ptc file: {job.ptc}")
    run.progress(1, "Preparing AI model 3.0...")
    fallback_pct = min(max(float(job.ground_pct), 0.0), 100.0)
    total = lasio.point_count(source)

    final_path = lasio.output_path_for(source, job.output_dir)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    work_root = (Path(job.work_root) if job.work_root else
                 tiling.local_work_root() / f"job_{os.getpid()}_{int(time.time() * 1000)}")
    shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)
    engine_out = work_root / "engine_classified.laz"   # fence mode
    warnings: List[str] = []
    try:
        xyz, original = job.xyz, job.classes
        if job.fence_mode and (xyz is None or original is None):
            run.progress(1, "Reading the point cloud...")
            file_xyz, file_classes = lasio.read_xyz_classes(source, run.check_cancel)
            xyz = file_xyz if xyz is None else xyz
            original = file_classes if original is None else original
        if original is not None:
            original = np.asarray(original, dtype=np.uint8).copy()
            if len(original) != total:
                raise Ai3Error(f"Classification/source point-count mismatch: {len(original):,} "
                               f"classes for {total:,} points.")
        if xyz is not None:
            xyz = np.asarray(xyz)
            if xyz.ndim != 2 or xyz.shape[1] < 3 or len(xyz) != total:
                raise Ai3Error(f"XYZ has an unexpected shape {xyz.shape} for {total:,} points.")

        if job.fence_mode:
            if job.targets is not None:
                targets = np.asarray(job.targets, dtype=np.int64).ravel()
                targets = np.unique(targets[(targets >= 0) & (targets < total)])
            else:
                targets = points_in_fences(xyz, normalize_polygons(job.polygons), run.check_cancel)
            if targets.size == 0:
                raise FenceError("No points are inside the selected fences.")
            # Only the fence post-processing needs the class codes (the engine reads the .ptc).
            class_mapping, power_mapping, config = _codes(job)
            mapping = build_engine_mapping(class_mapping, power_mapping, config,
                                           job.power_lines)["classes"]
            learning = FenceLearning(xyz, original, targets, mapping, enabled=job.learn,
                                     fallback_pct=fallback_pct, check_cancel=run.check_cancel)
            local_in = work_root / f"{source.stem}_fence_local.laz"
            buffer_m = max(float(job.fence_buffer_m), 0.0)
            run.progress(2, f"Cropping the fence area: {len(targets):,} target points + "
                            f"{buffer_m:.0f} m context...")
            subset = lasio.crop_to_targets(source, local_in,
                                           np.asarray(xyz[targets, :2], dtype=np.float64),
                                           buffer_m, run.check_cancel)
            if subset.size < MIN_ENGINE_POINTS:
                raise Ai3Error(f"The fence area holds only {subset.size:,} points; AI model 3.0 "
                               f"needs at least {MIN_ENGINE_POINTS:,}. Select a larger fence.")
            target_local = np.searchsorted(subset, targets)
            if np.any(target_local >= len(subset)) or not np.array_equal(subset[target_local], targets):
                raise Ai3Error("The fence crop lost some of the selected target points.")
            ground_pct = learning.ground_percentage()
            run.progress(8, f"Fence mode: {len(targets):,} target points, engine input "
                            f"{subset.size:,}/{total:,} points.")
            result = run.controller.run(local_in, engine_out, job.ptc, work_root,
                                        power_lines=job.power_lines, ground_unclass_pct=ground_pct,
                                        progress_callback=run.scaled(8, 88))
            run.check_cancel()
            run.progress(90, "Applying the result inside the selected fence(s)...")
            local_classes = lasio.read_engine_classes(local_in, engine_out, run.check_cancel)
            target_classes = learning.refine(local_classes[target_local])
            final_classes = original.copy()
            final_classes[targets] = target_classes
            processed, engine_points = int(len(targets)), int(subset.size)
            notes = [learning.ground_ratio_note, learning.reference_note]
        else:
            ground_pct = fallback_pct
            final_classes, result = _run_tiled(job, source, total, ground_pct, final_path,
                                               work_root, run)
            targets = target_classes = None
            processed = engine_points = total
            notes = []

        warnings = list(result.warnings) + [note for note in notes if note]
        if job.fence_mode:              # the tiled full-file path wrote it while merging
            run.progress(95, "Writing the classified copy of the source file...")
            lasio.write_classified_copy(source, final_classes, final_path, run.check_cancel)
        run.check_cancel()
        report_path = final_path.parent / f"{source.stem}_ai3_report.json"
        outcome = Ai3JobResult(
            source=source, output_path=final_path, report_path=report_path,
            fence_mode=job.fence_mode, points_total=total, points_processed=processed,
            engine_points=engine_points, ground_pct=float(ground_pct),
            final_classes=final_classes, target_indices=targets, target_classes=target_classes,
            warnings=warnings, engine_report=result.report)
        _write_report(report_path, integration=job.origin, app_version=__version__,
                      ptc=str(job.ptc), learn=job.learn, engine_report=result.report,
                      **outcome.summary())
        run.progress(100, "AI model 3.0 classification complete.")
        return outcome
    except lasio.LasIoError as exc:
        raise Ai3Error(str(exc)) from exc
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def _run_tiled(job: Ai3Job, source: Path, total: int, ground_pct: float, final_path: Path,
               work_root: Path, run: "_Run"):
    """Full file: split into tiles, classify them (in parallel), merge into the output."""
    with lasio.laspy.open(str(source), mode="r") as reader:
        grid = tiling.plan_grid(reader.header, job.tile_points,
                                max(float(job.fence_buffer_m), 0.0))
    workers = max(1, min(int(job.tile_workers), grid.count))
    spawn = getattr(run.controller, "spawn", None)
    handed_out = [False]

    def factory():
        # The first engine run uses the caller's controller, so the caller's cancel() still
        # reaches it; further parallel runs get their own.
        if not handed_out[0] or spawn is None:
            handed_out[0] = True
            return run.controller
        return spawn()
    if spawn is None:                   # a host controller that cannot be duplicated
        workers = 1
    try:
        class_mapping, power_mapping, config = _codes(job)
        unclassified = build_engine_mapping(class_mapping, power_mapping, config,
                                            job.power_lines)["classes"]["unclassified"]
    except (OSError, ValueError, KeyError):
        unclassified = 0
    tiles_dir = work_root / "tiles"
    # Tiles (LAZ, with overlap) take about the source's size; each running engine also holds
    # uncompressed intermediate files of roughly 120 bytes per point of its tile.
    needed = int(source.stat().st_size * 1.4 + workers * job.tile_points * 1.6 * 120)
    free = shutil.disk_usage(work_root).free
    if free < needed:
        raise Ai3Error(f"AI model 3.0 needs about {needed / 1e9:.1f} GB free in {work_root} for "
                       f"this file's tiles, but only {free / 1e9:.1f} GB is free. Free up space "
                       "or set NAKSHA_AI3_WORK_DIR to a folder on a larger local disk.")
    run.progress(2, f"Full-file mode: {total:,} points in {grid.count} tile(s) of "
                    f"{grid.sx:.0f} x {grid.sy:.0f} m, {workers} at a time.")
    tiles, file_stats = tiling.split(source, grid, tiles_dir, run.check_cancel,
                                     run.scaled(2, 10))
    results, warnings, reports = tiling.classify_tiles(
        tiles, grid, job.ptc, tiles_dir, power_lines=job.power_lines, ground_pct=ground_pct,
        device=job.device, workers=workers, cancel=run.cancel_event,
        check_cancel=run.check_cancel, progress=run.scaled(10, 92),
        controller_factory=factory, unclassified_code=unclassified, source=source,
        file_stats=file_stats, tile_options=tiling.takes_tile_options(run.controller))
    run.check_cancel()
    histogram = tiling.merge(source, grid, results, final_path, run.check_cancel,
                             run.scaled(92, 99))
    final_classes = lasio.read_classes(final_path, run.check_cancel) \
        if job.keep_final_classes else None
    report = {"tiles": len(tiles), "tile_grid": [grid.nx, grid.ny],
              "tile_size_m": [round(grid.sx, 1), round(grid.sy, 1)], "tile_workers": workers,
              "class_histogram": {str(c): int(n) for c, n in enumerate(histogram) if n},
              "tile_reports": {str(t): r for t, r in sorted(reports.items())}}
    return final_classes, SimpleNamespace(warnings=warnings, report=report)


@dataclass
class BatchItem:
    job: Ai3Job
    result: Optional[Ai3JobResult] = None
    error: Optional[str] = None


def run_batch(jobs: Sequence[Ai3Job], controller: Optional[Ai3Controller] = None,
              progress: Optional[Callable[[int, int, int, str], None]] = None,
              cancel: Optional[threading.Event] = None) -> List[BatchItem]:
    """Run jobs one after the other; a failed file is reported and the batch continues.

    ``progress(index, count, percent, message)`` reports the current file's progress.
    A cancellation stops the batch.
    """
    cancel = cancel or threading.Event()
    controller = controller or Ai3Controller(device=jobs[0].device if jobs else "auto")
    items = [BatchItem(job) for job in jobs]
    for index, item in enumerate(items):
        if cancel.is_set():
            item.error = "cancelled"
            continue
        per_file = (lambda p, m, i=index: progress(i, len(items), p, m)) if progress else None
        try:
            item.result = run_job(item.job, controller, per_file, cancel)
        except Ai3Cancelled as exc:
            item.error = str(exc)
            cancel.set()
        except (Ai3Error, FenceError, OSError, ValueError) as exc:
            item.error = str(exc)
    return items
