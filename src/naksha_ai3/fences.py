"""Fence polygons: reading them and finding the points inside.

A fences file is JSON: ``{"polygons": [[[x, y], [x, y], ...], ...]}`` (a polygon may also be
given as ``{"coords": [...]}``). Coordinates are in the point cloud's own XY. A point is inside
when it is inside any polygon (even-odd rule); rings need not be closed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import numpy as np


class FenceError(ValueError):
    """Unusable fence input, with a message fit to show the user."""


def _ring(coords) -> np.ndarray:
    ring = np.asarray(coords, dtype=np.float64)
    if ring.ndim != 2 or ring.shape[0] < 3 or ring.shape[1] < 2:
        raise FenceError("A fence polygon needs at least three XY vertices.")
    ring = ring[:, :2]
    if not np.isfinite(ring).all():
        raise FenceError("A fence polygon has a non-finite coordinate.")
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    if ring.shape[0] < 3:
        raise FenceError("A fence polygon needs at least three distinct vertices.")
    return ring


def normalize_polygons(polygons: Iterable) -> List[np.ndarray]:
    rings = []
    for polygon in polygons:
        coords = polygon.get("coords") if isinstance(polygon, dict) else polygon
        rings.append(_ring(coords))
    if not rings:
        raise FenceError("No fence polygons were given.")
    return rings


def load_polygons(path) -> List[np.ndarray]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FenceError(f"Cannot read the fences file {path}: {exc}") from exc
    polygons = data.get("polygons") if isinstance(data, dict) else data
    if not isinstance(polygons, list):
        raise FenceError(f"{path} has no 'polygons' list.")
    return normalize_polygons(polygons)


def save_polygons(path, polygons: Sequence) -> None:
    rings = normalize_polygons(polygons)
    Path(path).write_text(json.dumps({"polygons": [ring.tolist() for ring in rings]}),
                          encoding="utf-8")


def inside_polygon(xy: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test for many points against one ring."""
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), dtype=bool)
    x0, y0 = ring[:, 0], ring[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for ax, ay, bx, by in zip(x0, y0, x1, y1):
        if ay == by:
            continue
        crosses = (ay > y) != (by > y)
        if not crosses.any():
            continue
        x_at = ax + (y[crosses] - ay) * (bx - ax) / (by - ay)
        hit = np.zeros(len(xy), dtype=bool)
        hit[np.flatnonzero(crosses)[x[crosses] < x_at]] = True
        inside ^= hit
    return inside


def points_in_fences(xyz: np.ndarray, polygons: Sequence[np.ndarray],
                     check_cancel: Callable[[], None] = lambda: None,
                     chunk: int = 2_000_000) -> np.ndarray:
    """Sorted indices of the points inside any of the polygons."""
    rings = normalize_polygons(polygons)
    found = []
    for start in range(0, len(xyz), chunk):
        check_cancel()
        xy = np.asarray(xyz[start:start + chunk, :2], dtype=np.float64)
        selected = np.zeros(len(xy), dtype=bool)
        for ring in rings:
            lo, hi = ring.min(axis=0), ring.max(axis=0)
            near = np.flatnonzero((xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0])
                                  & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1]) & ~selected)
            if near.size:
                selected[near[inside_polygon(xy[near], ring)]] = True
        found.append(np.flatnonzero(selected) + start)
    return np.concatenate(found) if found else np.empty(0, dtype=np.int64)
