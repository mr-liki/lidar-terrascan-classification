"""Learning from already classified areas inside the selected fences.

A fence can hold the user's finished classification next to unfinished points. The finished
part is kept unchanged and used as reference:

* whether Uncategorized is a deliberate result (thinned ground) is read from the classified
  data around the fences; if it is not, every Uncategorized fence point is unfinished;
* the Ground/Uncategorized percentage passed to the engine is estimated from the reference;
* unfinished points take the class the reference shows for their engine class and their height
  above the local lowest surface (2 cm bands), which reproduces the user's height bands
  between Ground, thinned ground and low vegetation.

Points that already carry a class other than Uncategorized are never changed.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np


class FenceLearning:
    """Reference analysis and post-processing for one fence job.

    ``xyz_all`` and ``original`` cover the whole cloud in source order (the ring around the
    fences is read from them); ``targets`` are the fence point indices; ``mapping`` holds the
    delivered codes by name (``ground``, ``low_veg``, ``unclassified``...).
    """

    def __init__(self, xyz_all, original, targets, mapping: Dict[str, int], *,
                 enabled: bool = True, fallback_pct: float = 20.0,
                 check_cancel: Optional[Callable[[], None]] = None):
        self.xyz_all = xyz_all
        self.original = original
        self.targets = np.asarray(targets)
        self.mapping = dict(mapping)
        self.enabled = bool(enabled)
        self.fallback_pct = float(fallback_pct)
        self.check_cancel = check_cancel or (lambda: None)
        self.ground_ratio_note: Optional[str] = None
        self.reference_note: Optional[str] = None
        self._fence_reference_cache = None
        self._surrounding_cache = "unset"
        self._cell_m = None

    # Columns must hold enough points to judge; on sparse clouds they grow beyond 2 m so that
    # an average column holds CELL_TARGET_POINTS.
    CELL_TARGET_POINTS = 20

    @property
    def cell_m(self) -> float:
        """Column size for this cloud: REFERENCE_CELL_M, larger on sparse clouds."""
        if self._cell_m is None:
            xy = np.asarray(self.xyz_all[self.targets, :2], dtype=np.float64)
            xy = xy[np.isfinite(xy).all(axis=1)]
            size = self.REFERENCE_CELL_M
            if len(xy) >= 3:
                span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1.0)
                density = len(xy) / float(span[0] * span[1])
                size = max(size, float(np.sqrt(self.CELL_TARGET_POINTS / density)))
            self._cell_m = size
        return self._cell_m

    # Fence reference areas: 2 m columns where at least this share of the points already
    # carries a class other than Uncategorized are treated as finished by the user.
    REFERENCE_CELL_M = 2.0
    REFERENCE_MIN_LABELED_SHARE = 0.30
    REFERENCE_MIN_CELL_POINTS = 5
    # A column on the edge of an unfinished area holds both kinds of points and can pass the
    # share test, which would leave its unfinished points Uncategorized. Uncategorized points
    # this close to an unfinished column are therefore classified too, and nothing this close is
    # used as reference (its Uncategorized points are no longer known to be deliberate).
    UNFINISHED_EDGE_M = 2.0
    # Whether the user's classification uses Uncategorized deliberately (thinned ground) is read
    # from the classified columns in a ring around the fences. Below this share of Uncategorized
    # among Ground-or-Uncategorized there, every Uncategorized fence point is unfinished.
    SURROUND_RING_M = 15.0
    DELIBERATE_UNCAT_MIN_SHARE = 0.05
    SURROUND_MIN_POINTS = 200

    def surrounding_uncategorized(self):
        """(share, ground, uncategorized) in classified columns around the fences, or None."""
        cached = getattr(self, "_surrounding_cache", "unset")
        if cached != "unset":
            return cached
        mapping = self.mapping
        xyz_all = self.xyz_all
        target_xy = np.asarray(xyz_all[self.targets, :2], dtype=np.float64)
        lo = target_xy.min(axis=0) - self.SURROUND_RING_M
        hi = target_xy.max(axis=0) + self.SURROUND_RING_M
        inside = ((xyz_all[:, 0] >= lo[0]) & (xyz_all[:, 0] <= hi[0])
                  & (xyz_all[:, 1] >= lo[1]) & (xyz_all[:, 1] <= hi[1]))
        inside[self.targets] = False
        ring = np.flatnonzero(inside)
        result = None
        if ring.size:
            labels = np.asarray(self.original[ring])
            xy = np.asarray(xyz_all[ring, :2], dtype=np.float64)
            finite = np.isfinite(xy).all(axis=1)
            labels, xy = labels[finite], xy[finite]
        if ring.size and labels.size:
            uncat = labels == mapping["unclassified"]
            cells = np.floor((xy - xy.min(axis=0)) / self.cell_m).astype(np.int64)
            _, cell = np.unique(cells, axis=0, return_inverse=True)
            cell = cell.ravel()
            total = np.bincount(cell)
            labeled = np.bincount(cell, weights=(~uncat).astype(float))
            done = ((total >= self.REFERENCE_MIN_CELL_POINTS)
                    & (labeled >= self.REFERENCE_MIN_LABELED_SHARE * total))[cell]
            ground = int(np.count_nonzero(done & (labels == mapping["ground"])))
            uncategorized = int(np.count_nonzero(done & uncat))
            if ground + uncategorized >= self.SURROUND_MIN_POINTS:
                result = (uncategorized / (ground + uncategorized), ground, uncategorized)
        self._surrounding_cache = result
        return result

    def uncategorized_is_deliberate(self) -> bool:
        surrounding = self.surrounding_uncategorized()
        return surrounding is None or surrounding[0] >= self.DELIBERATE_UNCAT_MIN_SHARE

    def fence_reference(self):
        """(labels, xyz, classified, reference) for the fence targets.

        Uncategorized is both the state of unfinished points and a deliberate result in finished
        areas (thinned ground), so a single label cannot tell them apart; the neighbourhood can.
        A 2 m column with enough points of which too few carry a class is unfinished; its
        Uncategorized points, and those within one column (at least UNFINISHED_EDGE_M) of them, are
        classified.
        ``classified`` marks the points kept as they are, ``reference`` the subset away from
        the unfinished edge that the ratio and the learned table are taken from.
        """
        cached = getattr(self, "_fence_reference_cache", None)
        if cached is not None:
            return cached
        from scipy.spatial import cKDTree
        unclassified = self.mapping["unclassified"]
        labels = np.asarray(self.original[self.targets])
        xyz = np.asarray(self.xyz_all[self.targets], dtype=np.float64)
        finite = np.isfinite(xyz).all(axis=1)
        is_uncat = labels == unclassified
        if not self.uncategorized_is_deliberate():
            # The surrounding classification does not use Uncategorized: every Uncategorized
            # fence point is unfinished and every labelled one is reference.
            self._fence_reference_cache = (labels, xyz, ~is_uncat, ~is_uncat & finite)
            return self._fence_reference_cache
        classified = np.ones(len(labels), dtype=bool)
        near_edge = np.zeros(len(labels), dtype=bool)
        rows = np.flatnonzero(finite)
        if rows.size:
            xy = xyz[rows, :2]
            cells = np.floor((xy - xy.min(axis=0)) / self.cell_m).astype(np.int64)
            _, cell = np.unique(cells, axis=0, return_inverse=True)
            cell = cell.ravel()
            total = np.bincount(cell)
            labeled = np.bincount(cell, weights=(~is_uncat[rows]).astype(float))
            # Near-empty columns (fence edges) decide nothing: they follow their neighbours.
            unfinished_cell = ((total >= self.REFERENCE_MIN_CELL_POINTS)
                               & (labeled < self.REFERENCE_MIN_LABELED_SHARE * total))
            core = rows[unfinished_cell[cell] & is_uncat[rows]]
            if core.size:
                tree = cKDTree(xyz[core, :2])
                for start in range(0, rows.size, 200_000):
                    self.check_cancel()
                    chunk = rows[start:start + 200_000]
                    distance, _ = tree.query(xyz[chunk, :2], k=1,
                                             distance_upper_bound=max(self.UNFINISHED_EDGE_M, self.cell_m))
                    near_edge[chunk[np.isfinite(distance)]] = True
        else:
            classified[:] = False
        classified &= ~(near_edge & is_uncat)
        reference = classified & ~near_edge & finite
        self._fence_reference_cache = (labels, xyz, classified, reference)
        return self._fence_reference_cache

    def ground_percentage(self):
        """Ground/Uncategorized ratio of the classified areas inside the selected fences."""
        fallback = self.fallback_pct
        if not self.enabled:
            return fallback
        mapping = self.mapping
        if not self.uncategorized_is_deliberate():
            share, ground_n, uncat_n = self.surrounding_uncategorized()
            percentage = 100.0 * share
            self.ground_ratio_note = (
                f"Estimated Ground to Uncategorized: {percentage:.2f}% from {ground_n:,} Ground "
                f"and {uncat_n:,} Uncategorized points in the classified areas around the "
                "selected fences; every Unclassified point inside them is classified.")
            return percentage
        labels, xyz, _, reference_area = self.fence_reference()
        ground = reference_area & (labels == mapping["ground"])
        if np.count_nonzero(ground) < 100:
            self.ground_ratio_note = (
                f"No classified Ground reference area in the selected fences; using {fallback:g}%.")
            return fallback
        # Small 3D cells around existing Ground: Uncategorized points at ground level there are
        # the user's thinned ground, not elevated or unfinished points.
        sel = np.flatnonzero(reference_area)
        cells = np.floor((xyz[sel] - xyz[sel].min(axis=0)) / [2.0, 2.0, 0.25]).astype(np.int64)
        _, cell = np.unique(cells, axis=0, return_inverse=True)
        cell = cell.ravel()
        is_ground = labels[sel] == mapping["ground"]
        ground_counts = np.bincount(cell[is_ground], minlength=int(cell.max()) + 1)
        reference = ground_counts[cell] >= 10
        ground_count = int(np.count_nonzero(reference & is_ground))
        uncategorized_count = int(np.count_nonzero(
            reference & (labels[sel] == mapping["unclassified"])))
        if ground_count < 100:
            self.ground_ratio_note = (
                f"Insufficient classified Ground reference area; using {fallback:g}%.")
            return fallback
        percentage = 100.0 * uncategorized_count / (ground_count + uncategorized_count)
        self.ground_ratio_note = (
            f"Estimated Ground to Uncategorized: {percentage:.2f}% from "
            f"{ground_count:,} Ground and {uncategorized_count:,} Uncategorized "
            "points in the classified areas inside the selected fences.")
        return percentage

    # Learned refinement: the user's labels in the classified areas are tabulated against the
    # engine's broad class and the point's height above the local lowest surface; unfinished
    # points take the reference majority of their (class, height) cell.
    LEARN_HEIGHT_BIN_M = 0.02
    LEARN_NEIGHBORS = 8
    LEARN_MIN_SAMPLES = 20

    def local_heights(self, xyz: np.ndarray) -> np.ndarray:
        """Height above the second-lowest of the nearest points in plan view.

        Users separate Ground, thinned (Uncategorized) ground and low vegetation by a few
        centimetres, so this has to follow the local lowest surface; the second-lowest point
        keeps a single low outlier from defining it.
        """
        from scipy.spatial import cKDTree
        heights = np.full(len(xyz), np.nan)
        finite = np.flatnonzero(np.isfinite(xyz).all(axis=1))
        if finite.size < self.LEARN_NEIGHBORS:
            return heights
        tree = cKDTree(xyz[finite, :2])
        for start in range(0, finite.size, 200_000):
            self.check_cancel()
            rows = finite[start:start + 200_000]
            _, neighbors = tree.query(xyz[rows, :2], k=self.LEARN_NEIGHBORS)
            floor = np.sort(xyz[finite[neighbors], 2], axis=1)[:, 1]
            heights[rows] = xyz[rows, 2] - floor
        return heights

    def refine(self, predicted):
        """Keep the user's classified areas and fit the AI result of the rest to them."""
        if not self.enabled:
            return predicted
        mapping = self.mapping
        labels, xyz, classified, reference_area = self.fence_reference()
        kept = int(classified.sum())
        if kept == 0:
            self.reference_note = ("No classified reference area in the selected fences; "
                                    "normal AI3 output used.")
            return predicted
        result = np.asarray(predicted).copy()
        # The classified areas are the user's answer: every label stays, Uncategorized included.
        result[classified] = labels[classified]
        heights = self.local_heights(xyz)
        usable = np.isfinite(heights)
        open_rows = np.flatnonzero(~classified & usable)
        reference_rows = np.flatnonzero(reference_area & usable)
        changed = 0
        if open_rows.size and reference_rows.size:
            # Unclassified, Ground and Low vegetation form one group: telling them apart is
            # exactly what the height bands learned from the reference are for.
            ground_like = np.isin(predicted, [mapping["unclassified"], mapping["ground"],
                                              mapping["low_veg"]])
            group = np.where(ground_like, 256, np.asarray(predicted, dtype=np.int64))
            height_bin = np.clip(np.floor(heights[usable] / self.LEARN_HEIGHT_BIN_M),
                                 -50, 500).astype(np.int64) + 50
            key = np.full(len(labels), -1, dtype=np.int64)
            key[usable] = group[usable] * 1000 + height_bin
            ref_keys, ref_labels = key[reference_rows], labels[reference_rows].astype(np.int64)
            pairs, counts = np.unique(ref_keys * 256 + ref_labels, return_counts=True)
            pair_key, pair_label = pairs // 256, pairs % 256
            # Majority label per key: sort by key, then count descending, take the first.
            order = np.lexsort((-counts, pair_key))
            pair_key, pair_label, counts = pair_key[order], pair_label[order], counts[order]
            first = np.r_[True, pair_key[1:] != pair_key[:-1]]
            totals = np.add.reduceat(counts, np.flatnonzero(first))
            table_key, table_label = pair_key[first], pair_label[first]
            trusted = totals >= self.LEARN_MIN_SAMPLES
            table_key, table_label = table_key[trusted], table_label[trusted]
            pos = np.searchsorted(table_key, key[open_rows])
            pos = np.minimum(pos, max(len(table_key) - 1, 0))
            if len(table_key):
                hit = table_key[pos] == key[open_rows]
            else:
                hit = np.zeros(open_rows.size, dtype=bool)
            rows, new = open_rows[hit], table_label[pos[hit]].astype(result.dtype)
            changed = int(np.count_nonzero(result[rows] != new))
            result[rows] = new
        # A point the user already gave a class keeps it, even at the edge of an unfinished area.
        labeled = labels != mapping["unclassified"]
        result[labeled] = labels[labeled]
        self.reference_note = (
            f"Kept {kept:,} points of already classified areas unchanged; AI3 classified "
            f"{len(labels) - kept:,} unfinished points, {changed:,} of them adjusted to the "
            "classes and height bands learned from the classified areas.")
        return result
