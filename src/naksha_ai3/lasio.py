"""LAS/LAZ input and output for AI3 jobs: reading, fence cropping and the classified copy.

The classified copy is always built from the SOURCE file with only the classification replaced:
the engine's own output goes through colour handling for the model and is not a faithful copy of
the user's data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import laspy
import numpy as np

from .engine import las_compat  # noqa: F401  (installs the opaque extra-bytes fix on import)

STREAM_CHUNK = 1_000_000
ORDER_CHECK_STEP = 997             # every n-th point is compared when checking point order
ORDER_CHECK_TOL = 0.011            # metres: the LAS coordinate resolution is 0.01 m


class LasIoError(RuntimeError):
    """A LAS/LAZ consistency failure with a message fit to show the user."""


def _noop() -> None:
    return None


def point_count(path) -> int:
    with laspy.open(str(path), mode="r") as reader:
        return int(reader.header.point_count)


def read_xyz_classes(path, check_cancel: Callable[[], None] = _noop) -> Tuple[np.ndarray, np.ndarray]:
    """(xyz float64 Nx3, classification uint8) of a LAS/LAZ, streamed in chunks."""
    total = point_count(path)
    xyz = np.empty((total, 3), dtype=np.float64)
    classes = np.empty(total, dtype=np.uint8)
    pos = 0
    with laspy.open(str(path), mode="r") as reader:
        for points in reader.chunk_iterator(STREAM_CHUNK):
            check_cancel()
            n = len(points)
            xyz[pos:pos + n, 0] = points.x
            xyz[pos:pos + n, 1] = points.y
            xyz[pos:pos + n, 2] = points.z
            classes[pos:pos + n] = np.asarray(points.classification, dtype=np.uint8)
            pos += n
    if pos != total:
        raise LasIoError(f"Read {pos:,} of {total:,} points from {path}")
    return xyz, classes


def read_classes(path, check_cancel: Callable[[], None] = _noop) -> np.ndarray:
    """Classification of a LAS/LAZ (1 byte per point), streamed."""
    total = point_count(path)
    classes = np.empty(total, dtype=np.uint8)
    pos = 0
    with laspy.open(str(path), mode="r") as reader:
        for points in reader.chunk_iterator(STREAM_CHUNK):
            check_cancel()
            n = len(points)
            classes[pos:pos + n] = np.asarray(points.classification, dtype=np.uint8)
            pos += n
    if pos != total:
        raise LasIoError(f"Read {pos:,} of {total:,} classes from {path}")
    return classes


def extended_header(source_header):
    """Promote legacy records to the corresponding 8-bit classification format."""
    format_id = (6, 6, 7, 7, 9, 10)[source_header.point_format.id]
    header = source_header.copy()
    point_format = laspy.PointFormat(format_id)
    point_format.dimensions.extend(header.point_format.extra_dimensions)
    header.set_version_and_point_format(laspy.header.Version(1, 4), point_format)
    return header


def promote_points(points, point_format):
    result = laspy.PackedPointRecord.from_point_record(points, point_format)
    for name in points.point_format.extra_dimension_names:
        result.array[name] = points.array[name]
    result["scan_angle"] = np.rint(np.asarray(points["scan_angle_rank"]) / 0.006).astype(np.int16)
    return result


def crop_to_targets(source: Path, subset_path: Path, target_xy: np.ndarray, buffer_m: float,
                    check_cancel: Callable[[], None] = _noop) -> np.ndarray:
    """Write the source points inside the targets' bounding box + buffer; return their indices."""
    min_x, max_x = float(target_xy[:, 0].min() - buffer_m), float(target_xy[:, 0].max() + buffer_m)
    min_y, max_y = float(target_xy[:, 1].min() - buffer_m), float(target_xy[:, 1].max() + buffer_m)
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    subset_path.unlink(missing_ok=True)
    parts, source_pos = [], 0
    with laspy.open(str(source), mode="r") as reader:
        header = reader.header.copy()
        total = int(reader.header.point_count)
        with laspy.open(str(subset_path), mode="w", header=header,
                        do_compress=subset_path.suffix.lower() == ".laz") as writer:
            for points in reader.chunk_iterator(STREAM_CHUNK):
                check_cancel()
                x = np.asarray(points.x, dtype=np.float64)
                y = np.asarray(points.y, dtype=np.float64)
                mask = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
                if np.any(mask):
                    parts.append(np.flatnonzero(mask).astype(np.int64) + source_pos)
                    writer.write_points(points[mask])
                source_pos += len(points)
    if source_pos != total:
        raise LasIoError(f"Fence crop scan mismatch: {source_pos:,}/{total:,}")
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def read_engine_classes(reference: Path, engine_output: Path,
                        check_cancel: Callable[[], None] = _noop) -> np.ndarray:
    """Classification of the engine output, after checking it kept the point order.

    Everything downstream merges by index, so the order is verified rather than assumed:
    every ORDER_CHECK_STEP-th point of the reference file is compared with the output.
    """
    n_ref = point_count(reference)
    n_out = point_count(engine_output)
    if n_ref != n_out:
        raise LasIoError(f"The engine returned {n_out:,} points for {n_ref:,} input points.")
    classes = np.empty(n_out, dtype=np.uint8)
    pos = 0
    with laspy.open(str(reference), mode="r") as ref, \
            laspy.open(str(engine_output), mode="r") as out:
        for ref_pts, out_pts in zip(ref.chunk_iterator(STREAM_CHUNK),
                                    out.chunk_iterator(STREAM_CHUNK)):
            check_cancel()
            count = len(out_pts)
            if len(ref_pts) != count:
                raise LasIoError("The engine output could not be aligned with its input.")
            step = slice(0, count, ORDER_CHECK_STEP)
            for axis in "xyz":
                delta = np.abs(np.asarray(getattr(ref_pts, axis))[step]
                               - np.asarray(getattr(out_pts, axis))[step])
                if delta.size and float(delta.max()) > ORDER_CHECK_TOL:
                    raise LasIoError("The engine changed the point order; refusing to apply "
                                     "its classification to the wrong points.")
            classes[pos:pos + count] = np.asarray(out_pts.classification, dtype=np.uint8)
            pos += count
    if pos != n_out:
        raise LasIoError(f"Engine output read mismatch: {pos:,}/{n_out:,}")
    return classes


def write_classified_copy(source: Path, final_classes: np.ndarray, out_path: Path,
                          check_cancel: Callable[[], None] = _noop) -> None:
    """The source file with only the classification replaced (LAS 1.4 when codes need it)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)
    need_ext = int(final_classes.max(initial=0)) > 31
    start = 0
    with laspy.open(str(source), mode="r") as reader:
        source_pf = int(reader.header.point_format.id)
        # legacy point formats store classification in 5 bits: only those need LAS 1.4
        header = (extended_header(reader.header) if (need_ext and source_pf < 6)
                  else reader.header.copy())
        with laspy.open(str(out_path), mode="w", header=header,
                        do_compress=out_path.suffix.lower() == ".laz") as writer:
            for points in reader.chunk_iterator(STREAM_CHUNK):
                check_cancel()
                count = len(points)
                if header.point_format.id != reader.header.point_format.id:
                    points = promote_points(points, header.point_format)
                points.classification = final_classes[start:start + count]
                writer.write_points(points)
                start += count
    if start != len(final_classes):
        raise LasIoError(f"Classified copy write mismatch: {start:,}/{len(final_classes):,}")


def output_path_for(source: Path, out_dir: Optional[Path] = None) -> Path:
    """``<out_dir or source folder/AI3_Classified>/<stem>_ai3_classified<suffix>``."""
    folder = Path(out_dir) if out_dir else source.parent / "AI3_Classified"
    return folder / f"{source.stem}_ai3_classified{source.suffix}"
