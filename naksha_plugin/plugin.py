"""AI Model 3.0 - Naksha plugin entry point (see manifest.json).

The plugin ships its own copy of the ``naksha_ai3`` package (engine and model included) next to
this file and always uses that copy, so it works independently of the AI3 code built into a
given Naksha version.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
ICON_FILE = PLUGIN_DIR / "ai_model_3.svg"
BUTTON_LABEL = "AI Model 3.0"


def plugin_icon(size: int = 24):
    """The plugin's icon in the theme's AI colour, drawn like Naksha's own ribbon icons."""
    from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
    from PySide6.QtGui import QIcon, QPainter, QPixmap
    from PySide6.QtSvg import QSvgRenderer
    color = "#FF8A65"
    try:
        from gui.theme_manager import ThemeColors
        color = ThemeColors.get("icon_ai") or color
    except Exception:                   # noqa: BLE001
        pass
    try:
        svg = ICON_FILE.read_text(encoding="utf-8").replace("{color}", color)
    except OSError:
        return QIcon()
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    inner = size * 0.92                 # the same inset as the built-in ribbon icons
    renderer.render(painter, QRectF((size - inner) / 2, (size - inner) / 2, inner, inner))
    painter.end()
    return QIcon(pixmap)

try:                                    # the host's base class, when available
    from gui.naksha_plugin_api import NakshaPlugin
except Exception:                       # noqa: BLE001
    class NakshaPlugin:                 # minimal stand-in with the same interface
        def on_load(self, app_window):
            self.app_window = app_window

        def get_ribbon_button(self):
            return None

        def on_unload(self):
            pass


def _package_modules():
    return [name for name in sys.modules if name == "naksha_ai3" or name.startswith("naksha_ai3.")]


def _from_plugin(module) -> bool:
    locations = [getattr(module, "__file__", None)] + list(getattr(module, "__path__", []) or [])
    for location in locations:
        if location:
            path = Path(location).resolve()
            if path == PLUGIN_DIR or PLUGIN_DIR in path.parents:
                return True
    return False


def forget_package() -> None:
    """Drop every imported naksha_ai3 module (namespace subpackages included)."""
    for name in _package_modules():
        sys.modules.pop(name, None)


def use_bundled_package() -> None:
    """Make ``import naksha_ai3`` resolve to this plugin's copy."""
    plugin = str(PLUGIN_DIR)
    while plugin in sys.path:
        sys.path.remove(plugin)
    sys.path.insert(0, plugin)
    # Another copy (e.g. Naksha's built-in one, or an earlier install of this plugin) may be
    # imported already: drop it so the plugin runs with the engine and model it shipped with.
    if any(not _from_plugin(sys.modules[name]) for name in _package_modules()):
        forget_package()


class Ai3Plugin(NakshaPlugin):
    def __init__(self):
        self.app_window = None
        self._window = None
        self._setup_thread = None
        self._setup_cancel = threading.Event()

    def on_load(self, app_window):
        self.app_window = app_window
        use_bundled_package()
        import naksha_ai3
        print(f"[AI Model 3.0 plugin] loaded naksha_ai3 {naksha_ai3.__version__} from "
              f"{Path(naksha_ai3.__file__).parent}")
        # Plugin Manager calls on_load after installation and at application startup.
        # Setup stays off the GUI thread; Run joins the same serialized readiness check.
        self._setup_cancel.clear()
        self._setup_thread = threading.Thread(target=self._prepare_runtime,
                                              name="AI3 plugin setup", daemon=True)
        self._setup_thread.start()

    def _prepare_runtime(self):
        from naksha_ai3.controller import Ai3Controller
        try:
            status = Ai3Controller().ensure_ready(cancel=self._setup_cancel)
            print(f"[AI Model 3.0 plugin] {status.message} Python: {status.python}")
        except Exception as exc:
            print(f"[AI Model 3.0 plugin] Runtime setup: {exc}. Click Run to retry.")


    def get_ribbon_button(self):
        # The ribbon only draws icons it knows by name; the plugin puts its own on the button
        # once the ribbon has created it (this runs on every ribbon rebuild). "AI" is shown
        # until then, or if the icon cannot be drawn.
        try:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, self.decorate_button)
        except Exception:               # noqa: BLE001
            pass
        return {"label": BUTTON_LABEL, "emoji": "AI", "section": "AI",
                "callback": self.open_window}

    def decorate_button(self) -> int:
        """Put the plugin icon on its ribbon button(s); returns how many were found."""
        from PySide6.QtCore import QSize
        from PySide6.QtWidgets import QPushButton
        window = self.app_window
        if window is None:
            return 0
        icon = plugin_icon(24)
        if icon.isNull():
            return 0
        found = 0
        try:
            buttons = window.findChildren(QPushButton, "ribbonButton")
        except RuntimeError:
            return 0
        for button in buttons:
            if button.property("ribbonText") == BUTTON_LABEL:
                button.setText("")
                button.setIcon(icon)
                button.setIconSize(QSize(24, 24))
                found += 1
        return found

    def open_window(self):
        from naksha_ai3_window import Ai3PluginWindow
        try:
            if self._window is not None and self._window.isVisible():
                self._window.raise_()
                self._window.activateWindow()
                return
        except RuntimeError:            # the Qt object was deleted
            self._window = None
        self._window = Ai3PluginWindow(self.app_window)
        icon = plugin_icon(32)
        if not icon.isNull():
            self._window.setWindowIcon(icon)
        self._window.show()
        self.decorate_button()          # e.g. after a theme change redrew the ribbon

    def on_unload(self):
        self._setup_cancel.set()
        window, self._window = self._window, None
        if window is not None:
            try:
                window.shutdown()
                window.close()
                window.deleteLater()
            except RuntimeError:
                pass
        if self._setup_thread is not None and self._setup_thread.is_alive():
            return
        for name in ("naksha_ai3_window", "naksha_ai3_host"):
            sys.modules.pop(name, None)
        forget_package()
