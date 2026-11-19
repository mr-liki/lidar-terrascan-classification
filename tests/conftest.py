import sys
from pathlib import Path
from types import SimpleNamespace

import laspy
import numpy as np
import pytest

APP = Path(__file__).resolve().parents[1]
SRC = APP / "src"
for _path in (SRC, APP):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

PTC_TEXT = """0\t\tUnclassified
*\tZero length line\t0\t128,128,128\t1\t1.00

2\t\tGround
*\tZero length line\t2\t255,127,0\t1\t1.00

3\t\tLow vegetation
*\tZero length line\t3\t128,255,128\t1\t1.00

4\t\tMedium vegetation
*\tZero length line\t4\t0,255,0\t1\t1.00

5\t\tHigh vegetation
*\tZero length line\t5\t0,64,0\t1\t1.00

6\t\tBuilding
*\tZero length line\t6\t255,0,128\t1\t1.00

7\t\tLow point
*\tZero length line\t7\t255,0,255\t1\t1.00
"""


@pytest.fixture
def ptc(tmp_path):
    path = tmp_path / "classes.ptc"
    path.write_text(PTC_TEXT, encoding="utf-8")
    return path


def write_cloud(path: Path, xyz: np.ndarray, classes: np.ndarray, point_format=6):
    header = laspy.LasHeader(point_format=point_format, version="1.4")
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [float(xyz[:, 0].min()), float(xyz[:, 1].min()), 0.0]
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    cloud.classification = classes.astype(np.uint8)
    cloud.intensity = np.arange(len(xyz)) % 65535
    cloud.write(path)
    return path


@pytest.fixture
def grid_cloud(tmp_path):
    """100 m x 100 m, 1 point per m2, flat ground with a 0.1 m low-vegetation layer every 3rd row."""
    gx, gy = np.meshgrid(np.arange(100, dtype=float), np.arange(100, dtype=float))
    x, y = gx.ravel() + 500000.0, gy.ravel() + 4000000.0
    z = np.where(gy.ravel() % 3 == 0, 0.10, 0.0)
    xyz = np.c_[x, y, z]
    classes = np.where(z > 0.05, 3, 2).astype(np.uint8)
    path = write_cloud(tmp_path / "tile.laz", xyz, classes)
    return SimpleNamespace(path=path, xyz=xyz, classes=classes)


class FakeController:
    """Stands in for the engine: copies its input and sets every class to ``code``."""

    def __init__(self, code=2, fn=None):
        self.code, self.fn, self.calls = code, fn, []

    def run(self, input_path, output_path, ptc_path, work_dir, power_lines=True,
            progress_callback=None, mapping=None, ground_unclass_pct=None):
        self.calls.append(dict(input=Path(input_path), ground_pct=ground_unclass_pct,
                               power_lines=power_lines))
        cloud = laspy.read(input_path)
        cloud.classification = (self.fn(cloud) if self.fn else
                                np.full(len(cloud.points), self.code, dtype=np.uint8))
        cloud.write(output_path)
        if progress_callback:
            progress_callback(50, "fake engine")
        return SimpleNamespace(output_path=Path(output_path), report={"fake": True}, warnings=[])

    def cancel(self):
        pass


@pytest.fixture
def fake_controller():
    return FakeController
