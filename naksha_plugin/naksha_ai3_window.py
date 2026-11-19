"""The AI Model 3.0 plugin window: settings, progress, and applying the result."""
from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QDoubleSpinBox, QFileDialog,
                               QSpinBox,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QRadioButton, QVBoxLayout, QWidget)

import os
from concurrent.futures import ThreadPoolExecutor

from naksha_ai3 import __version__, lasio
from naksha_ai3.tiling import TILE_WORKERS
from naksha_ai3.controller import Ai3Cancelled, Ai3Controller, Ai3Error
from naksha_ai3.fences import FenceError, points_in_fences
from naksha_ai3.job import Ai3Job, run_job
from naksha_ai3.mapping import mappings_from_ptc

from naksha_ai3_host import (HostError, HostSession, applied_ptc, configure_runtime,
                             fence_choices, highlight_fences, open_classified_copy)

# Whole-cloud runs on clouds up to this size write the result into the open view (undo-able);
# that needs every point's XYZ in memory. Larger clouds open the classified copy instead.
INPLACE_MAX_POINTS = int(os.environ.get("NAKSHA_AI3_INPLACE_MAX_POINTS", "100000000"))


class _RunThread(QThread):
    """Prepare the cloud, classify it, and hand the result back to the GUI thread."""

    progress = Signal(int, str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, session: HostSession, settings: dict, parent=None):
        super().__init__(parent)
        self.session, self.settings = session, settings
        self.controller = Ai3Controller()
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()
        self.controller.cancel()

    def _job(self, **fields) -> Ai3Job:
        output_dir = self.settings.get("output_dir")
        return Ai3Job(source=self.session.source, ptc=Path(self.settings["ptc"]),
                      output_dir=Path(output_dir) if output_dir else None,
                      power_lines=self.settings["power_lines"],
                      ground_pct=self.settings["ground_pct"],
                      tile_workers=self.settings.get("tile_workers", 2),
                      origin=f"Naksha AI Model 3.0 plugin {__version__}", **fields)

    def run(self) -> None:
        try:
            self.controller.ensure_ready(lambda p, m: self.progress.emit(0, m), self.cancel_event)
            if self.cancel_event.is_set():
                raise Ai3Cancelled("AI model 3.0 classification was cancelled.")
            if self.settings["polygons"]:
                self._run_fences()
            else:
                self._run_full_file()
        except (Ai3Cancelled, Ai3Error, FenceError, HostError) as exc:
            self.failed.emit(str(exc))
        except Exception:               # noqa: BLE001
            self.failed.emit(traceback.format_exc())

    def _run_fences(self) -> None:
        emit = self.progress.emit
        dataset = self.session.prepare(lambda p, m: emit(int(p * 0.12), m))
        if self.cancel_event.is_set():
            raise Ai3Cancelled("AI model 3.0 classification was cancelled.")
        before = np.asarray(dataset["classification"]).astype(np.uint8)
        xyz = np.asarray(dataset["xyz"])
        emit(12, "Finding the points inside the selected fences...")
        targets = points_in_fences(xyz, self.settings["polygons"])
        if targets.size == 0:
            raise FenceError("No points are inside the selected fences.")
        job = self._job(targets=targets, classes=before, xyz=xyz, learn=self.settings["learn"])
        result = run_job(job, self.controller,
                         lambda p, m: emit(12 + int(p * 0.86), m), self.cancel_event)
        after = before.copy()
        after[result.target_indices] = result.target_classes
        self.succeeded.emit({"result": result, "before": before, "after": after})

    def _run_full_file(self) -> None:
        """Tiled engine run. Up to INPLACE_MAX_POINTS the result is written into the open view
        (undo-able); the view's own data is read while the engine runs. Larger clouds are not
        read into memory at all: the classified copy is opened in the view afterwards."""
        emit = self.progress.emit
        total = lasio.point_count(self.session.source)
        in_place = total <= INPLACE_MAX_POINTS
        pool = ThreadPoolExecutor(max_workers=1) if in_place else None
        try:
            pending = pool.submit(self.session.prepare, lambda p, m: None) if pool else None
            job = self._job(keep_final_classes=in_place)
            result = run_job(job, self.controller, lambda p, m: emit(int(p * 0.97), m),
                             self.cancel_event)
            if not in_place:
                self.succeeded.emit({"result": result, "open": result.output_path})
                return
            emit(97, "Matching the result to the open view...")
            dataset = pending.result()
            before = np.asarray(dataset["classification"]).astype(np.uint8)
            self.succeeded.emit({"result": result, "before": before,
                                 "after": np.asarray(result.final_classes, dtype=np.uint8)})
        finally:
            if pool is not None:
                pool.shutdown(wait=True)


class Ai3PluginWindow(QDialog):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self._thread: Optional[_RunThread] = None
        self._session: Optional[HostSession] = None
        self._fences: List[dict] = []
        self._started = 0.0
        self.setWindowTitle(f"AI Model 3.0 (plugin {__version__})")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setModal(False)
        self.resize(620, 640)
        self._build()
        self.refresh()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QVBoxLayout(self)

        self.source_label = QLabel()
        self.source_label.setWordWrap(True)
        self.runtime_label = QLabel()
        self.runtime_label.setWordWrap(True)
        root.addWidget(self.source_label)
        root.addWidget(self.runtime_label)

        ptc_box = QGroupBox("Class definition (.ptc)")
        ptc_row = QHBoxLayout(ptc_box)
        self.ptc_edit = QLineEdit()
        self.ptc_edit.setPlaceholderText("The .ptc applied in Display Mode, or choose one")
        self.ptc_edit.textChanged.connect(self._update_codes)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_ptc)
        ptc_row.addWidget(self.ptc_edit, 1)
        ptc_row.addWidget(browse)
        root.addWidget(ptc_box)
        self.codes_label = QLabel()
        self.codes_label.setWordWrap(True)
        root.addWidget(self.codes_label)

        scope_box = QGroupBox("What to classify")
        scope = QVBoxLayout(scope_box)
        self.whole_radio = QRadioButton("The whole point cloud")
        self.fence_radio = QRadioButton("Only inside the selected fences")
        group = QButtonGroup(self)
        group.addButton(self.whole_radio)
        group.addButton(self.fence_radio)
        self.whole_radio.toggled.connect(self._update_enabled)
        scope.addWidget(self.whole_radio)
        scope.addWidget(self.fence_radio)
        fence_row = QHBoxLayout()
        self.fence_list = QListWidget()
        self.fence_list.setMinimumHeight(110)
        self.fence_list.itemChanged.connect(self._highlight)
        reload_fences = QPushButton("Reload fences")
        reload_fences.clicked.connect(self.refresh)
        fence_row.addWidget(self.fence_list, 1)
        fence_row.addWidget(reload_fences, 0, Qt.AlignTop)
        scope.addLayout(fence_row)
        self.learn_check = QCheckBox("Learn from already classified areas inside the fences")
        self.learn_check.setToolTip(
            "Classified areas inside the fences are kept unchanged and used as reference: the "
            "unfinished points follow their classes and height bands, and the "
            "Ground/Uncategorized percentage is taken from them.")
        self.learn_check.setChecked(True)
        scope.addWidget(self.learn_check)
        root.addWidget(scope_box)

        options = QGroupBox("Options")
        opts = QVBoxLayout(options)
        self.power_check = QCheckBox("Detect power-line wires and poles (when the .ptc has them)")
        self.power_check.setChecked(True)
        opts.addWidget(self.power_check)
        pct_row = QHBoxLayout()
        pct_row.addWidget(QLabel("Ground delivered as Uncategorized (%):"))
        self.pct_spin = QDoubleSpinBox()
        self.pct_spin.setRange(0.0, 100.0)
        self.pct_spin.setDecimals(1)
        self.pct_spin.setValue(20.0)
        self.pct_spin.setToolTip("Used unless it is learned from classified areas in the fences.")
        pct_row.addWidget(self.pct_spin)
        pct_row.addStretch(1)
        opts.addLayout(pct_row)
        tiles_row = QHBoxLayout()
        tiles_row.addWidget(QLabel("Whole cloud: tiles classified at the same time:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 8)
        self.workers_spin.setValue(max(1, min(8, TILE_WORKERS)))
        self.workers_spin.setToolTip(
            "Large clouds are classified in tiles of about 8 million points. More tiles at once "
            "use more CPU cores, GPU memory and RAM; 2 suits a 6 GB GPU.")
        tiles_row.addWidget(self.workers_spin)
        tiles_row.addStretch(1)
        opts.addLayout(tiles_row)
        root.addWidget(options)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        root.addWidget(self.progress_bar)
        root.addWidget(self.status_label)
        root.addWidget(self.log, 1)

        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run AI Model 3.0")
        self.run_button.clicked.connect(self.start)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setEnabled(False)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        buttons.addStretch(1)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.close_button)
        root.addLayout(buttons)

    # ------------------------------------------------------------------ state
    def refresh(self) -> None:
        source = getattr(self.app, "_lnt_source_path", None)
        view = getattr(self.app, "_lnt_main_view", None)
        if view is not None and getattr(view, "native", None) is not None and source:
            self.source_label.setText(f"<b>Point cloud:</b> {source}")
        else:
            self.source_label.setText("<b>Point cloud:</b> none open - open a LAS/LAZ in the "
                                      "native view first.")
        status = configure_runtime()
        state = {"ready": "ready", "unverified": "found (checked on the first run)"}.get(
            status.state, status.message)
        self.runtime_label.setText(f"<b>AI3 runtime:</b> {state}"
                                   + (f" - {status.python}" if status.python else ""))
        if not self.ptc_edit.text():
            self.ptc_edit.setText(applied_ptc(self.app) or "")
        self._fences = fence_choices(self.app)
        self.fence_list.blockSignals(True)
        self.fence_list.clear()
        for choice in self._fences:
            item = QListWidgetItem(choice["label"])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if len(self._fences) == 1 else Qt.Unchecked)
            self.fence_list.addItem(item)
        if not self._fences:
            self.fence_list.addItem("No completed fences in the view. Draw one, then Reload.")
        self.fence_list.blockSignals(False)
        if self._fences and not self.whole_radio.isChecked():
            self.fence_radio.setChecked(True)
        elif not self._fences:
            self.whole_radio.setChecked(True)
        self._update_enabled()
        self._highlight()

    def _update_enabled(self) -> None:
        fences = self.fence_radio.isChecked()
        idle = self._thread is None
        self.fence_list.setEnabled(fences and idle and bool(self._fences))
        self.learn_check.setEnabled(fences and idle)
        for widget in (self.whole_radio, self.fence_radio, self.ptc_edit, self.power_check,
                       self.pct_spin, self.workers_spin, self.run_button):
            widget.setEnabled(idle)
        self.fence_radio.setEnabled(idle and bool(self._fences))
        self.cancel_button.setEnabled(not idle)

    def _update_codes(self) -> None:
        path = self.ptc_edit.text().strip()
        if not path:
            self.codes_label.setText("")
            return
        try:
            classes, power, optional, notes = mappings_from_ptc(path)
        except (OSError, ValueError) as exc:
            self.codes_label.setText(f"<span style='color:#e57373'>{exc}</span>")
            return
        names = ("Ground", "Low veg", "Medium veg", "High veg", "Building")
        text = ", ".join(f"{n} {classes[i]}" for i, n in enumerate(names))
        text += f"; wire {power[5]}, pole {power[6]}"
        if "uncategorized" in optional:
            text += f"; Uncategorized {optional['uncategorized']}"
        self.codes_label.setText(f"Codes: {text}")

    def _browse_ptc(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Class definition file",
                                              self.ptc_edit.text(), "PTC files (*.ptc)")
        if path:
            self.ptc_edit.setText(path)

    def _selected_fences(self) -> List[dict]:
        return [choice for index, choice in enumerate(self._fences)
                if self.fence_list.item(index).checkState() == Qt.Checked]

    def _highlight(self, *_):
        highlight_fences(self.app, self._selected_fences()
                         if self.fence_radio.isChecked() and self._fences else [])

    def _say(self, message: str) -> None:
        self.status_label.setText(message)
        self.log.appendPlainText(message)

    # ------------------------------------------------------------------ run
    def start(self) -> None:
        ptc = self.ptc_edit.text().strip()
        if not ptc or not Path(ptc).is_file():
            QMessageBox.warning(self, "AI Model 3.0", "Choose the class-definition .ptc file.")
            return
        polygons = None
        if self.fence_radio.isChecked():
            chosen = self._selected_fences()
            if not chosen:
                QMessageBox.warning(self, "AI Model 3.0", "Tick at least one fence.")
                return
            polygons = [choice["coords"] for choice in chosen]
        try:
            self._session = HostSession.open(self.app)
        except HostError as exc:
            QMessageBox.warning(self, "AI Model 3.0", str(exc))
            return
        settings = {"ptc": ptc, "polygons": polygons,
                    "learn": self.learn_check.isChecked(),
                    "power_lines": self.power_check.isChecked(),
                    "ground_pct": float(self.pct_spin.value()),
                    "tile_workers": int(self.workers_spin.value())}
        self.log.clear()
        self._say(f"Classifying {'inside ' + str(len(polygons)) + ' fence(s)' if polygons else 'the whole point cloud'} "
                  f"with {Path(ptc).name}...")
        self.progress_bar.setValue(0)
        self._started = time.time()
        self._thread = _RunThread(self._session, settings, self)
        self._thread.progress.connect(self._on_progress)
        self._thread.succeeded.connect(self._on_success)
        self._thread.failed.connect(self._on_failure)
        self._thread.finished.connect(self._on_thread_done)
        self._update_enabled()
        self._thread.start()

    def cancel(self) -> None:
        if self._thread is not None:
            self._say("Cancelling...")
            self._thread.cancel()

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(max(self.progress_bar.value(), int(percent)))
        if message:
            self.status_label.setText(message)

    def _on_success(self, payload: dict) -> None:
        result = payload["result"]
        if "open" in payload:
            self._open_result(result, payload["open"])
            return
        try:
            changed = self._session.apply(payload["after"], payload["before"],
                                          "fence" if result.fence_mode else "full")
        except Exception as exc:        # noqa: BLE001
            self._say(f"Classification finished, but it could not be applied: {exc}")
            QMessageBox.warning(self, "AI Model 3.0", str(exc))
            return
        self.progress_bar.setValue(100)
        minutes, seconds = divmod(int(time.time() - self._started), 60)
        lines = [f"Done in {minutes:02d}:{seconds:02d}. {changed:,} points changed "
                 f"({result.points_processed:,} classified). Undo with Ctrl+Z.",
                 f"Classified copy: {result.output_path}"]
        lines += [f"- {note}" for note in result.warnings]
        for line in lines:
            self._say(line)
        self.status_label.setText(lines[0])

    def _open_result(self, result, path) -> None:
        """Large cloud: show the classified copy in the view (the source stays unchanged)."""
        self.progress_bar.setValue(100)
        minutes, seconds = divmod(int(time.time() - self._started), 60)
        self._say(f"Done in {minutes:02d}:{seconds:02d}. {result.points_total:,} points "
                  f"classified in {result.engine_report.get('tiles', 1)} tile(s).")
        self._say(f"Classified copy: {path}")
        for note in result.warnings:
            self._say(f"- {note}")
        try:
            open_classified_copy(self.app, path)
            self._say("The classified copy is now open in the view; the original file is "
                      "unchanged.")
        except Exception as exc:        # noqa: BLE001
            self._say(f"Open the classified copy from File > Open: {exc}")

    def _on_failure(self, message: str) -> None:
        self._say(message)
        if "cancelled" not in message.lower():
            QMessageBox.warning(self, "AI Model 3.0", message[-1500:])

    def _on_thread_done(self) -> None:
        self._thread = None
        self._update_enabled()
        self._highlight()

    # ------------------------------------------------------------------ closing
    def shutdown(self) -> None:
        thread = self._thread
        if thread is not None:
            thread.cancel()
            thread.wait(15000)
        highlight_fences(self.app, [])

    def closeEvent(self, event) -> None:
        if self._thread is not None:
            if QMessageBox.question(self, "AI Model 3.0",
                                    "Classification is running. Cancel it and close?") \
                    != QMessageBox.Yes:
                event.ignore()
                return
        self.shutdown()
        super().closeEvent(event)
