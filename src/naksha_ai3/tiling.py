"""Tiled full-file classification: bounded memory and parallel engine runs for very large clouds.

The engine loads its whole input into memory (PDAL ground filter, the model, the rule engine),
so a single pass over a file with hundreds of millions or billions of points is slow or
impossible. A full file is therefore classified in square tiles:

1. **Split** - one streaming pass over the source writes every point into the tile that owns
   it (its *core* tile) and into the neighbouring tiles whose ``buffer_m`` margin it falls in.
   Tiles are written in source order to a local work folder (never the network share).
2. **Classify** - the engine runs on each tile; ``workers`` tiles run at the same time. The
   buffer gives the ground filter, the 50 m inference tiles and the neighbourhood rules the same
   context they would have in a single pass.
3. **Keep the core** - of each tile's result only the core points' classes are kept, in source
   order, as a small ``uint8`` file.
4. **Merge** - one more streaming pass over the source writes the classified copy: every point
   takes the next class of its core tile. Point order is never changed.

Memory is bounded by the tile size, not the file size. Tile sides follow the point density so a
tile holds about ``target_points`` core points.
"""
from __future__ import annotations

import math
import os
import shutil
import threading
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import laspy
import numpy as np

from . import lasio
from .controller import Ai3Cancelled, Ai3Controller, Ai3Error

TILE_TARGET_POINTS = int(os.environ.get("NAKSHA_AI3_TILE_POINTS", "8000000"))
TILE_WORKERS = int(os.environ.get("NAKSHA_AI3_TILE_WORKERS", "2"))
MIN_TILE_SIDE_M = 100.0
MAX_TILE_SIDE_M = 2000.0
MIN_ENGINE_POINTS = 5_000

ProgressCallback = Callable[[int, str], None]


@dataclass
class TileGrid:
    x0: float
    y0: float
    sx: float
    sy: float
    nx: int
    ny: int
    buffer_m: float

    @property
    def count(self) -> int:
        return self.nx * self.ny

    def core_ids(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ix = np.clip(np.floor((x - self.x0) / self.sx), 0, self.nx - 1).astype(np.int64)
        iy = np.clip(np.floor((y - self.y0) / self.sy), 0, self.ny - 1).astype(np.int64)
        return iy * self.nx + ix

    def bounds(self, tile: int):
        ix, iy = tile % self.nx, tile // self.nx
        return (self.x0 + ix * self.sx, self.y0 + iy * self.sy,
                self.x0 + (ix + 1) * self.sx, self.y0 + (iy + 1) * self.sy)


def plan_grid(header, target_points: int = TILE_TARGET_POINTS, buffer_m: float = 50.0) -> TileGrid:
    """A grid over the file's extent whose tiles hold about ``target_points`` points each."""
    total = max(int(header.point_count), 1)
    x0, y0 = float(header.mins[0]), float(header.mins[1])
    dx = max(float(header.maxs[0]) - x0, 1.0)
    dy = max(float(header.maxs[1]) - y0, 1.0)
    density = total / (dx * dy)
    side = math.sqrt(max(target_points, 1) / density)
    side = min(max(side, MIN_TILE_SIDE_M, 2.5 * buffer_m), MAX_TILE_SIDE_M)
    nx, ny = max(1, math.ceil(dx / side)), max(1, math.ceil(dy / side))
    # Equal tiles over the exact extent: no slivers at the far edges.
    return TileGrid(x0, y0, dx / nx, dy / ny, nx, ny, float(buffer_m))


class _TileWriters:
    """Lazily opened LAZ writers, one per tile, all with the source's header."""

    def __init__(self, folder: Path, header):
        self.folder, self.header = folder, header
        self.writers: Dict[int, laspy.LasWriter] = {}
        self.counts: Dict[int, int] = {}

    def path(self, tile: int) -> Path:
        return self.folder / f"tile_{tile:05d}.laz"

    def write(self, tile: int, points) -> None:
        writer = self.writers.get(tile)
        if writer is None:
            writer = laspy.open(str(self.path(tile)), mode="w", header=self.header.copy(),
                                do_compress=True)
            self.writers[tile] = writer
            self.counts[tile] = 0
        writer.write_points(points)
        self.counts[tile] += len(points)

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()


def split(source: Path, grid: TileGrid, folder: Path, check_cancel: Callable[[], None],
          progress: ProgressCallback):
    """Write the tiles (core + buffer, source order).

    Returns ({tile: {points, core, path}}, FileStats) - the file-wide values every tile uses.
    """
    folder.mkdir(parents=True, exist_ok=True)
    core_counts = np.zeros(grid.count, dtype=np.int64)
    intensity_hist = np.zeros(65536, dtype=np.int64)
    has_intensity = False
    sampled_cells = []
    done = 0
    with laspy.open(str(source), mode="r") as reader:
        total = int(reader.header.point_count)
        dims = set(reader.header.point_format.dimension_names)
        writers = _TileWriters(folder, reader.header)
        try:
            for points in reader.chunk_iterator(lasio.STREAM_CHUNK):
                check_cancel()
                x = np.asarray(points.x, dtype=np.float64)
                y = np.asarray(points.y, dtype=np.float64)
                core = grid.core_ids(x, y)
                core_counts += np.bincount(core, minlength=grid.count)
                if "intensity" in dims:
                    has_intensity = True
                    intensity_hist += np.bincount(np.asarray(points.intensity, dtype=np.int64),
                                                  minlength=65536)[:65536]
                sampled_cells.append(_sample_cells(x, y))
                b = grid.buffer_m
                ix_lo = np.clip(np.floor((x - b - grid.x0) / grid.sx), 0, grid.nx - 1).astype(np.int64)
                ix_hi = np.clip(np.floor((x + b - grid.x0) / grid.sx), 0, grid.nx - 1).astype(np.int64)
                iy_lo = np.clip(np.floor((y - b - grid.y0) / grid.sy), 0, grid.ny - 1).astype(np.int64)
                iy_hi = np.clip(np.floor((y + b - grid.y0) / grid.sy), 0, grid.ny - 1).astype(np.int64)
                # buffer < tile side, so a point lies in at most 2 x 2 buffered tiles
                member = np.stack([iy_lo * grid.nx + ix_lo, iy_lo * grid.nx + ix_hi,
                                   iy_hi * grid.nx + ix_lo, iy_hi * grid.nx + ix_hi], axis=1)
                # a tile listed twice for the same point (ix_lo == ix_hi) counts once
                member[:, 1][member[:, 1] == member[:, 0]] = -1
                member[:, 2][member[:, 2] == member[:, 0]] = -1
                member[:, 3][(member[:, 3] == member[:, 0]) | (member[:, 3] == member[:, 1])
                             | (member[:, 3] == member[:, 2])] = -1
                in_tile = np.zeros(len(points), dtype=bool)
                for tile in np.unique(member[member >= 0]):
                    np.any(member == tile, axis=1, out=in_tile)
                    # boolean selection keeps source order inside every tile
                    writers.write(int(tile), points[in_tile])
                done += len(points)
                progress(int(100 * done / max(total, 1)),
                         f"Splitting into tiles: {done:,}/{total:,} points")
        finally:
            writers.close()
    if done != total:
        raise Ai3Error(f"Tile split read {done:,} of {total:,} points.")
    tiles = {tile: {"points": writers.counts[tile], "core": int(core_counts[tile]),
                    "path": writers.path(tile)}
             for tile in writers.counts if core_counts[tile] > 0}
    stats = FileStats(intensity_hist=intensity_hist if has_intensity else None,
                      point_density=_density_from_sample(sampled_cells))
    return tiles, stats


# A deterministic 1-in-SAMPLE_EVERY subset of the 1 m cells gives the file's median point
# density without holding a cell table for the whole survey.
SAMPLE_EVERY = 16


def _sample_cells(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cx = np.floor(x).astype(np.int64)
    cy = np.floor(y).astype(np.int64)
    key = cx * 1_000_003 + cy
    keep = ((cx * 73_856_093) ^ (cy * 19_349_663)) % SAMPLE_EVERY == 0
    return key[keep]


def _density_from_sample(chunks) -> Optional[float]:
    keys = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
    if keys.size == 0:
        return None
    _, counts = np.unique(keys, return_counts=True)
    return float(np.median(counts))


@dataclass
class FileStats:
    """File-wide values the rule engine would otherwise measure on each tile."""
    intensity_hist: Optional[np.ndarray]       # 65536 bins
    point_density: Optional[float]             # median points per occupied 1 m2 cell

    def intensity_percentile(self, pct: float) -> Optional[float]:
        hist = self.intensity_hist
        if hist is None or hist.sum() == 0:
            return None
        cumulative = np.cumsum(hist)
        rank = pct / 100.0 * (cumulative[-1] - 1)
        return float(np.searchsorted(cumulative, rank, side="right"))

    def overrides(self, intensity_percentile: float = 50.0) -> dict:
        result: dict = {}
        threshold = self.intensity_percentile(intensity_percentile)
        if threshold is not None:
            result["building_recovery"] = {"intensity_threshold": threshold}
        if self.point_density:
            result["roof_consolidation"] = {"point_density": self.point_density}
        return result


def _pooled_gate(reports) -> Optional[str]:
    """The intensity-gate decision a single pass would take: pooled candidate median >=
    pooled ground median (from the tiles' 256-bin histograms). None when unknown."""
    cand = ground = None
    for report in reports:
        stats = (report or {}).get("building_recovery") or {}
        c, g = stats.get("intensity_candidates_hist"), stats.get("intensity_ground_hist")
        if c is None or g is None:
            continue
        cand = np.asarray(c, dtype=np.int64) if cand is None else cand + np.asarray(c)
        ground = np.asarray(g, dtype=np.int64) if ground is None else ground + np.asarray(g)
    if cand is None or ground is None or cand.sum() == 0 or ground.sum() == 0:
        return None

    def median_bin(hist):
        return int(np.searchsorted(np.cumsum(hist), (hist.sum() - 1) / 2.0, side="right"))
    return "on" if median_bin(cand) >= median_bin(ground) else "off"


def takes_tile_options(controller) -> bool:
    """Whether a controller's run() accepts overrides / keep_step2 / step3_from. A host may
    supply an older controller; its tiles then run without the file-wide decisions."""
    import inspect
    try:
        parameters = inspect.signature(controller.run).parameters
    except (TypeError, ValueError, AttributeError):
        return False
    return "step3_from" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def _merge_overrides(*parts) -> dict:
    merged: dict = {}
    for part in parts:
        for section, values in (part or {}).items():
            merged.setdefault(section, {}).update(values)
    return merged


def _core_classes(grid: TileGrid, tile: int, tile_in: Path, tile_out: Path,
                  check_cancel: Callable[[], None]) -> np.ndarray:
    """Classes of the tile's core points, in source order."""
    classes = lasio.read_engine_classes(tile_in, tile_out, check_cancel)
    keep = np.empty(len(classes), dtype=bool)
    pos = 0
    with laspy.open(str(tile_in), mode="r") as reader:
        for points in reader.chunk_iterator(lasio.STREAM_CHUNK):
            n = len(points)
            keep[pos:pos + n] = grid.core_ids(np.asarray(points.x, dtype=np.float64),
                                              np.asarray(points.y, dtype=np.float64)) == tile
            pos += n
    return classes[keep]


def classify_tiles(tiles: Dict[int, dict], grid: TileGrid, ptc, folder: Path, *,
                   power_lines: bool, ground_pct: float, device: str, workers: int,
                   cancel: threading.Event, check_cancel: Callable[[], None],
                   progress: ProgressCallback, controller_factory=None,
                   unclassified_code: int = 0, source: Optional[Path] = None,
                   file_stats: Optional[FileStats] = None, tile_options: bool = True):
    """Run the engine on every tile with the file-wide values a single pass would use.

    The rule engine measures a few values on its whole input (the building intensity
    threshold and whether to apply it, the point density used for eaves). A tile would measure
    them on itself, so a building could be kept in one tile and dropped in the next. Instead:

    * the intensity threshold and the point density come from the split pass (``file_stats``);
    * the gate decision is pooled from the first batch of tiles (the largest, run together),
      then fixed for every other tile; a first-batch tile that decided differently re-runs
      step 3 only, from its kept step 2 output.

    Returns ({tile: .npy of its core classes}, warnings, {tile: engine report}).
    """
    controller_factory = controller_factory or (lambda: Ai3Controller(device=device))
    order = sorted(tiles, key=lambda t: -tiles[t]["points"])      # biggest first
    total_points = sum(tiles[t]["points"] for t in order) or 1
    finished_points = [0]
    lock = threading.Lock()
    active: List[Ai3Controller] = []
    results: Dict[int, Path] = {}
    warnings: List[str] = []
    reports: Dict[int, dict] = {}
    tile_percent: Dict[int, float] = {}
    base = file_stats.overrides() if file_stats is not None else {}
    file_points = sum(tiles[t]["core"] for t in order)
    workers = max(1, int(workers))

    def report(message: str) -> None:
        with lock:
            running = sum(tiles[t]["points"] * p / 100.0 for t, p in tile_percent.items())
            share = (finished_points[0] + running) / total_points
        progress(int(100 * min(share, 1.0)), message)

    def engine(tile: int, out: Path, work: Path, overrides: dict, keep_step2=None,
               step3_from=None, label: str = ""):
        controller = controller_factory()
        with lock:
            active.append(controller)
        try:
            def tile_progress(percent: int, message: str) -> None:
                with lock:
                    tile_percent[tile] = percent
                report(f"Tile {len(results) + 1}/{len(order)}{label}: {message}")
            extra = {}
            if tile_options:
                extra = {"overrides": overrides or None, "keep_step2": keep_step2,
                         "step3_from": step3_from, "grid_origin": (grid.x0, grid.y0)}
            outcome = controller.run(tiles[tile]["path"], out, ptc, work,
                                     power_lines=power_lines, ground_unclass_pct=ground_pct,
                                     progress_callback=tile_progress, **extra)
            with lock:
                reports[tile] = getattr(outcome, "report", {}) or {}
                for note in getattr(outcome, "warnings", []) or []:
                    if note not in warnings:
                        warnings.append(note)
            return outcome
        finally:
            with lock:
                if controller in active:
                    active.remove(controller)

    def finish(tile: int, out: Path) -> None:
        info = tiles[tile]
        core = _core_classes(grid, tile, info["path"], out, check_cancel)
        if len(core) != info["core"]:
            raise Ai3Error(f"Tile {tile}: {len(core):,} core classes for "
                           f"{info['core']:,} core points.")
        target = folder / f"tile_{tile:05d}_classes.npy"
        np.save(target, core)
        with lock:
            results[tile] = target
            tile_percent.pop(tile, None)
            finished_points[0] += info["points"]
        report(f"Classified tile {len(results)}/{len(order)}")

    def cleanup(tile: int) -> None:
        for name in (f"tile_{tile:05d}_classified.laz", f"tile_{tile:05d}_step2.las"):
            try:
                (folder / name).unlink()
            except OSError:
                pass
        try:
            tiles[tile]["path"].unlink()
        except OSError:
            pass
        shutil.rmtree(folder / f"tile_{tile:05d}_work", ignore_errors=True)

    def too_small(tile: int) -> bool:
        info = tiles[tile]
        if info["points"] < MIN_ENGINE_POINTS and source is not None:
            # A sparse corner of the grid: re-cut it from the source with a wider margin until
            # the engine has enough context. Only the tile's own points are kept, as always.
            info["points"] = _widen_small_tile(source, grid, tile, info["path"], check_cancel)
        if info["points"] >= MIN_ENGINE_POINTS or info["points"] >= file_points:
            return False                # enough context, or the whole file: as a single pass
        target = folder / f"tile_{tile:05d}_classes.npy"
        np.save(target, np.full(info["core"], unclassified_code, dtype=np.uint8))
        with lock:
            warnings.append(f"Only {info['points']:,} points around tile {tile}; its "
                            f"{info['core']:,} points were left Unclassified.")
            results[tile] = target
            finished_points[0] += info["points"]
        cleanup(tile)
        return True

    def first_batch(tile: int) -> None:
        """Run with the gate decided on the tile itself; keep step 2 for a possible redo."""
        check_cancel()
        if too_small(tile):
            return
        out = folder / f"tile_{tile:05d}_classified.laz"
        keep = (folder / f"tile_{tile:05d}_step2.las"
                if len(order) > 1 and tile_options else None)
        engine(tile, out, folder / f"tile_{tile:05d}_work", base, keep_step2=keep)

    def later(tile: int, gate: Optional[str]) -> None:
        check_cancel()
        try:
            if too_small(tile):
                return
            out = folder / f"tile_{tile:05d}_classified.laz"
            overrides = _merge_overrides(
                base, {"building_recovery": {"intensity_gate": gate}} if gate else None)
            engine(tile, out, folder / f"tile_{tile:05d}_work", overrides)
            check_cancel()
            finish(tile, out)
        finally:
            cleanup(tile)

    def stop_all() -> None:
        with lock:
            running = list(active)
        for controller in running:
            controller.cancel()

    def run_all(pool, function, items) -> None:
        futures = [pool.submit(function, *item) for item in items]
        done, pending = wait(futures, return_when=FIRST_EXCEPTION)
        failed = [f for f in done if f.exception() is not None]
        if failed:
            cancel.set()
            stop_all()
            for future in pending:
                future.cancel()
            wait(pending)
            error = failed[0].exception()
            if isinstance(error, (Ai3Cancelled, Ai3Error)):
                raise error
            raise Ai3Error(f"A tile failed: {error}") from error

    watcher_done = threading.Event()

    def watch_cancel() -> None:
        while not watcher_done.wait(0.5):
            if cancel.is_set():
                stop_all()
                return

    watcher = threading.Thread(target=watch_cancel, daemon=True)
    watcher.start()
    batch = order[:workers]
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # 1. the largest tiles, deciding the intensity gate themselves
            run_all(pool, first_batch, [(t,) for t in batch])
            check_cancel()
            gate = _pooled_gate(reports.get(t) for t in batch if t in reports)

            # 2. first-batch tiles that decided otherwise redo step 3 with the pooled decision
            def settle(tile: int) -> None:
                try:
                    if tile in results:                 # too small: already settled
                        return
                    out = folder / f"tile_{tile:05d}_classified.laz"
                    applied = ((reports.get(tile) or {}).get("building_recovery") or {}) \
                        .get("intensity_applied")
                    redo_from = folder / f"tile_{tile:05d}_step2.las"
                    if gate is not None and applied is not None and len(order) > 1 \
                            and bool(applied) != (gate == "on") and redo_from.is_file():
                        engine(tile, out, folder / f"tile_{tile:05d}_work3",
                               _merge_overrides(base, {"building_recovery":
                                                       {"intensity_gate": gate}}),
                               step3_from=redo_from,
                               label=" (step 3 with the file-wide building decision)")
                    check_cancel()
                    finish(tile, out)
                finally:
                    cleanup(tile)
            run_all(pool, settle, [(t,) for t in batch])

            # 3. every other tile with the decision fixed
            run_all(pool, later, [(t, gate) for t in order[workers:]])
    finally:
        watcher_done.set()
        for tile in batch:
            cleanup(tile)
    if gate is not None and len(order) > 1:
        warnings.append(f"Building intensity gate {gate} for the whole file "
                        f"(decided from the {len(batch)} largest tile(s)).")
    return results, warnings, reports


def _widen_small_tile(source: Path, grid: TileGrid, tile: int, path: Path,
                      check_cancel: Callable[[], None]) -> int:
    """Rewrite a tile with growing margins until it holds MIN_ENGINE_POINTS; returns its size."""
    x0, y0, x1, y1 = grid.bounds(tile)
    corners = np.array([[x0, y0], [x1, y1]], dtype=np.float64)
    count = 0
    for margin in (grid.buffer_m * 4, grid.buffer_m * 16, 4 * max(grid.sx, grid.sy)):
        count = int(lasio.crop_to_targets(source, path, corners, margin, check_cancel).size)
        if count >= MIN_ENGINE_POINTS:
            break
    return count


def merge(source: Path, grid: TileGrid, tile_classes: Dict[int, Path], out_path: Path,
          check_cancel: Callable[[], None], progress: ProgressCallback) -> np.ndarray:
    """Write the classified copy; returns the class histogram (256 bins)."""
    arrays = {tile: np.load(path, mmap_mode="r") for tile, path in tile_classes.items()}
    cursor = {tile: 0 for tile in arrays}
    histogram = np.zeros(256, dtype=np.int64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)
    max_code = max((int(np.max(a)) if len(a) else 0) for a in arrays.values()) if arrays else 0
    done = 0
    with laspy.open(str(source), mode="r") as reader:
        total = int(reader.header.point_count)
        source_pf = int(reader.header.point_format.id)
        header = (lasio.extended_header(reader.header) if (max_code > 31 and source_pf < 6)
                  else reader.header.copy())
        with laspy.open(str(out_path), mode="w", header=header,
                        do_compress=out_path.suffix.lower() == ".laz") as writer:
            for points in reader.chunk_iterator(lasio.STREAM_CHUNK):
                check_cancel()
                core = grid.core_ids(np.asarray(points.x, dtype=np.float64),
                                     np.asarray(points.y, dtype=np.float64))
                classes = np.empty(len(points), dtype=np.uint8)
                for tile in np.unique(core):
                    rows = np.flatnonzero(core == tile)
                    start = cursor[int(tile)]
                    block = arrays[int(tile)][start:start + len(rows)]
                    if len(block) != len(rows):
                        raise Ai3Error(f"Tile {tile} ran out of classes while merging.")
                    classes[rows] = block
                    cursor[int(tile)] = start + len(rows)
                histogram += np.bincount(classes, minlength=256)
                if header.point_format.id != reader.header.point_format.id:
                    points = lasio.promote_points(points, header.point_format)
                points.classification = classes
                writer.write_points(points)
                done += len(points)
                progress(int(100 * done / max(total, 1)),
                         f"Writing the classified copy: {done:,}/{total:,} points")
    unused = [t for t, a in arrays.items() if cursor[t] != len(a)]
    if unused or done != total:
        raise Ai3Error(f"Tile merge mismatch ({done:,}/{total:,} points; tiles {unused[:5]}).")
    return histogram


def local_work_root() -> Path:
    """Work folder on a local disk (tiles must not be written to a network share)."""
    from .runtime import user_data_dir
    override = os.environ.get("NAKSHA_AI3_WORK_DIR")
    return Path(override) if override else user_data_dir() / "work"
