"""Everything the AI Model 3.0 plugin needs from the Naksha host.

The classification itself is ``naksha_ai3`` (bundled with the plugin). This module only:

* reads the open native (Vulkan/LNT) point cloud in source order, with its current classes
  including unsaved edits, through Naksha's LNT AI bridge;
* lists the completed fences the user can classify inside;
* writes the result back into the open view, with an undo step, and refreshes the statistics.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

ProgressCallback = Callable[[int, str], None]

FENCE_TYPES = {"rectangle", "polygon", "freehand", "polyline", "orthopolygon", "ortho", "circle"}
_READ_ONLY_SOURCES = {"snt_readonly", "block_creation", "cell_creation", "measure_selection"}


class HostError(RuntimeError):
    """A host-side problem with a message fit to show the user."""


# ---------------------------------------------------------------------- runtime / PTC
def configure_runtime():
    """Report the standalone runtime status."""
    from naksha_ai3 import runtime

    return runtime.runtime_status()


def applied_ptc(app) -> Optional[str]:
    """The .ptc applied in Display Mode, if any."""
    try:
        if not getattr(app, "active_ptc_schema", None):
            return None
        path = getattr(app, "active_ptc_path", None)
    except Exception:                   # noqa: BLE001
        return None
    return str(path) if path and os.path.isfile(str(path)) else None


# ---------------------------------------------------------------------- fences
def _fence_coords(drawing) -> list:
    kind = str(drawing.get("type", "")).lower()
    if kind == "circle" and drawing.get("center") is not None and drawing.get("radius") is not None:
        center = np.asarray(drawing["center"], dtype=np.float64)
        radius = float(drawing["radius"])
        angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        return [(float(center[0] + radius * np.cos(a)), float(center[1] + radius * np.sin(a)))
                for a in angles]
    coords = []
    for point in drawing.get("coords", []) or []:
        if len(point) >= 2:
            coords.append((float(point[0]), float(point[1])))
    return coords


def fence_choices(app) -> List[dict]:
    """Completed fences in the view: ``{"label", "coords", "drawing"}`` each."""
    try:
        from gui.lnt_ai_bridge import native_fence_drawings
        from gui.native_byclass_fences import _valid_ring
        drawings = native_fence_drawings(app)
    except Exception:                   # noqa: BLE001
        return []
    choices = []
    for drawing in drawings:
        if not isinstance(drawing, dict):
            continue
        kind = str(drawing.get("type", "")).lower()
        if kind not in FENCE_TYPES or drawing.get("imported") or \
                drawing.get("source") in _READ_ONLY_SOURCES:
            continue
        try:
            ring = _valid_ring(_fence_coords(drawing))
        except (TypeError, ValueError, IndexError):
            ring = None
        if ring is None:
            continue
        xs, ys = [p[0] for p in ring], [p[1] for p in ring]
        name = drawing.get("name") or drawing.get("layer") or ""
        label = (f"{len(choices) + 1}. {kind} - {max(xs) - min(xs):.1f} x "
                 f"{max(ys) - min(ys):.1f} m" + (f" ({name})" if name else ""))
        choices.append({"label": label, "coords": ring, "drawing": drawing})
    return choices


def highlight_fences(app, choices: List[dict]) -> None:
    try:
        from gui.lnt_ai_bridge import show_native_highlights
        show_native_highlights(app, [{"coords": c["coords"], "color": "#ffd400", "width": 3}
                                     for c in choices])
    except Exception:                   # noqa: BLE001
        pass


def open_classified_copy(app, path) -> None:
    """Open a classified copy in the native view with Naksha's own loader (keeps drawings)."""
    from gui.lnt_open import open_with_lnt
    open_with_lnt(app, str(path), preserve_drawings=True)


# ---------------------------------------------------------------------- session
class HostSession:
    """One plugin run against the open native point cloud."""

    def __init__(self, app, view, bridge, source: Path):
        self.app, self.view, self.bridge, self.source = app, view, bridge, source
        self.dataset = None

    @classmethod
    def open(cls, app) -> "HostSession":
        """Check the view and claim the LNT bridge; call on the GUI thread."""
        view = getattr(app, "_lnt_main_view", None)
        if view is None or getattr(view, "native", None) is None:
            raise HostError("Open a LAS/LAZ point cloud in the native Vulkan view first.")
        from gui.lnt_ai_bridge import LntAiBridge, active_bridge
        if active_bridge(app) is not None or getattr(app, "_ai_preparation_worker", None):
            raise HostError("Another AI classification is open. Close it, then run "
                            "AI Model 3.0 again.")
        try:
            from gui.lnt_collection import is_collection
            if is_collection(view.lnt_path):
                raise HostError("This view shows a collection of several files. Open a single "
                                "LAS/LAZ file to classify it with the AI Model 3.0 plugin.")
        except ImportError:
            pass
        source = getattr(app, "_lnt_source_path", None)
        if not source or not os.path.isfile(source) or \
                os.path.splitext(source)[1].lower() not in (".las", ".laz"):
            raise HostError("The source LAS/LAZ of the open view was not found:\n"
                            f"{source or '(unknown)'}")
        bridge = getattr(app, "_lnt_ai_bridge", None)
        if bridge is None or bridge.view is not view:
            if bridge is not None:
                bridge.discard_cache()
                bridge.deleteLater()
            bridge = LntAiBridge(app, view)
            app._lnt_ai_bridge = bridge
        store = getattr(app, "_lnt_section_store", None)
        if store is not None and getattr(store, "view", None) is view:
            store.flush()               # the .lnt must hold every edit shown on screen
        return cls(app, view, bridge, Path(source))

    def prepare(self, progress: ProgressCallback) -> dict:
        """Source-order XYZ and current classes (runs on a worker thread, like Naksha's)."""
        self.dataset = self.bridge.prepare(str(self.source), progress, flush_store=False)
        if self.dataset is None:
            raise HostError("The point cloud could not be prepared for classification.")
        return self.dataset

    def apply(self, new_classes: np.ndarray, before: np.ndarray, label: str) -> int:
        """Write ``new_classes`` into the open view with one undo step; GUI thread only."""
        if getattr(self.app, "_lnt_main_view", None) is not self.view or self.view.native is None:
            raise HostError("The point cloud was closed while AI Model 3.0 was running; "
                            "the result was saved to the AI3_Classified folder only.")
        new_classes = np.asarray(new_classes, dtype=before.dtype)
        changed = new_classes != before
        count = int(changed.sum())
        if count == 0:
            return 0
        bridge, dataset = self.bridge, self.dataset
        bridge.begin(dataset)
        try:
            dataset["classification"] = new_classes.copy()
            bridge.commit()
            node, local = bridge.identity(changed)
        finally:
            bridge.end()
        self._push_undo(changed, before, new_classes, node, local, label)
        try:
            from gui.point_count_widget import refresh_point_statistics
            refresh_point_statistics(self.app)
        except Exception as exc:        # noqa: BLE001
            print(f"[AI Model 3.0 plugin] statistics refresh failed: {exc}")
        return count

    def _push_undo(self, changed, before, after, node, local, label) -> None:
        step = {"type": "ai_classification", "ai_mode": "ai3_plugin", "fence_mode": label == "fence",
                "old_classes": before[changed].copy(), "oldclasses": before[changed].copy(),
                "new_classes": after[changed].copy(), "newclasses": after[changed].copy(),
                "lnt_node": node, "lnt_local": local, "lnt_path": self.view.lnt_path}
        stack = getattr(self.app, "undo_stack", None)
        if stack is None:
            stack = getattr(self.app, "undostack", None)
        if stack is None:
            return
        stack.append(step)
        for name in ("redo_stack", "redostack"):
            redo = getattr(self.app, name, None)
            if redo is not None:
                redo.clear()
        try:
            from gui.memory_manager import trim_undo_stack
            trim_undo_stack(self.app)
        except Exception:               # noqa: BLE001
            pass
