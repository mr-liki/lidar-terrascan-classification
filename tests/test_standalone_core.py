"""The standalone AI3 package: independence from Naksha, PTC codes, fences, jobs and the CLI."""
import json
import subprocess
from types import SimpleNamespace
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

from conftest import SRC, write_cloud

from naksha_ai3 import cli, lasio
from naksha_ai3.fences import FenceError, load_polygons, points_in_fences, save_polygons
from naksha_ai3.job import Ai3Job, run_batch, run_job
from naksha_ai3.mapping import build_engine_mapping, mappings_from_ptc
from naksha_ai3.reference import FenceLearning


def test_package_imports_nothing_from_naksha():
    code = ("import sys; import naksha_ai3.job, naksha_ai3.cli, naksha_ai3.runtime;"
            "print(','.join(m for m in sys.modules if m == 'gui' or m.startswith('gui.')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=SRC, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""
    sources = "\n".join(p.read_text(encoding="utf-8") for p in (SRC / "naksha_ai3").rglob("*.py"))
    assert "from gui" not in sources and "import gui" not in sources


def test_ptc_codes(ptc):
    classes, power, optional, notes = mappings_from_ptc(ptc)
    assert classes == {0: 2, 1: 3, 2: 4, 3: 5, 4: 6}
    assert optional == {"uncategorized": 0, "low_point": 7}
    assert power == {5: 14, 6: 15} and "not in the PTC" in notes["wire"]
    mapping = build_engine_mapping(classes, power, {"_ptc_optional_semantic_codes": optional}, False)
    assert mapping["classes"]["unclassified"] == 0 and mapping["classes"]["low_noise"] == 7


def test_points_in_fences_and_file_round_trip(tmp_path):
    xyz = np.array([[0.5, 0.5, 0], [2, 2, 0], [4, 5, 0], [0.9, 0.1, 0], [5.5, 4, 0]])
    square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    triangle = [[3, 3], [6, 3], [6, 6], [3, 3]]                 # closed ring is accepted
    path = tmp_path / "fences.json"
    save_polygons(path, [square, {"coords": triangle}])
    rings = load_polygons(path)
    assert points_in_fences(xyz, rings).tolist() == [0, 3, 4]
    with pytest.raises(FenceError):
        load_polygons(tmp_path / "missing.json")


def test_full_file_job_writes_classified_copy(grid_cloud, ptc, fake_controller, tmp_path):
    controller = fake_controller(code=5)
    result = run_job(Ai3Job(source=grid_cloud.path, ptc=ptc, ground_pct=33),
                     controller=controller)
    assert result.output_path == grid_cloud.path.parent / "AI3_Classified" / "tile_ai3_classified.laz"
    out = laspy.read(result.output_path)
    assert (np.asarray(out.classification) == 5).all()
    np.testing.assert_array_equal(np.asarray(out.intensity), np.arange(10000) % 65535)
    assert controller.calls[0]["ground_pct"] == 33
    report = json.loads(result.report_path.read_text())
    assert report["status"] == "COMPLETED" and report["fence_mode"] is False
    assert not list(result.output_path.parent.glob("_ai3_work_*"))


def test_fence_job_changes_only_fence_points(grid_cloud, ptc, fake_controller):
    polygon = [[500010, 4000010], [500030, 4000010], [500030, 4000030], [500010, 4000030]]
    result = run_job(Ai3Job(source=grid_cloud.path, ptc=ptc, polygons=[polygon]),
                     controller=fake_controller(code=6))
    inside = points_in_fences(grid_cloud.xyz, [polygon])
    out = np.asarray(laspy.read(result.output_path).classification)
    assert (out[inside] == 6).all()
    outside = np.setdiff1d(np.arange(len(out)), inside)
    np.testing.assert_array_equal(out[outside], grid_cloud.classes[outside])
    assert result.points_processed == len(inside)


def test_fence_learning_follows_reference_and_keeps_labels(grid_cloud, ptc, fake_controller):
    """Half the fence is unfinished (0). The fake engine calls everything Ground; learning
    restores the 0.1 m low-vegetation band from the classified half, and never touches labels."""
    polygon = [[500000, 4000000], [500060, 4000000], [500060, 4000060], [500000, 4000060]]
    classes = grid_cloud.classes.copy()
    unfinished = np.flatnonzero((grid_cloud.xyz[:, 0] < 500030)
                                & (grid_cloud.xyz[:, 1] < 4000060))
    classes[unfinished] = 0
    result = run_job(Ai3Job(source=grid_cloud.path, ptc=ptc, polygons=[polygon], classes=classes,
                            learn=True), controller=fake_controller(code=2))
    out = result.final_classes
    np.testing.assert_array_equal(out[unfinished], grid_cloud.classes[unfinished])
    labelled = np.flatnonzero(classes != 0)
    np.testing.assert_array_equal(out[labelled], classes[labelled])
    assert result.ground_pct == 0.0                 # the surroundings use no Uncategorized
    assert any("Kept" in note for note in result.warnings)


def test_learning_is_off_by_default(grid_cloud, ptc, fake_controller):
    polygon = [[500000, 4000000], [500060, 4000000], [500060, 4000060], [500000, 4000060]]
    classes = grid_cloud.classes.copy()
    classes[:] = 0
    result = run_job(Ai3Job(source=grid_cloud.path, ptc=ptc, polygons=[polygon], classes=classes,
                            ground_pct=12.5), controller=fake_controller(code=2))
    inside = points_in_fences(grid_cloud.xyz, [polygon])
    assert (result.final_classes[inside] == 2).all()
    assert result.ground_pct == 12.5


def test_batch_continues_after_a_failed_file(grid_cloud, ptc, fake_controller, tmp_path):
    missing = tmp_path / "missing.laz"
    items = run_batch([Ai3Job(source=missing, ptc=ptc), Ai3Job(source=grid_cloud.path, ptc=ptc)],
                      controller=fake_controller(code=2))
    assert items[0].error and "not found" in items[0].error
    assert items[1].result is not None and items[1].result.output_path.is_file()


def test_cli_job_file(grid_cloud, ptc, tmp_path, monkeypatch, fake_controller, capsys):
    classes = grid_cloud.classes.copy()
    np.save(tmp_path / "classes.npy", classes)
    np.save(tmp_path / "targets.npy", np.arange(50, 7000))
    job_file = tmp_path / "job.json"
    job_file.write_text(json.dumps({
        "inputs": [str(grid_cloud.path)], "ptc": str(ptc), "output_dir": "out",
        "classes": "classes.npy", "targets": "targets.npy", "learn": True,
        "class_mapping": {"0": 2, "1": 3, "2": 4, "3": 5, "4": 6}}))
    jobs = cli.job_from_file(job_file)
    assert jobs[0].output_dir == tmp_path / "out" and jobs[0].learn
    assert jobs[0].class_mapping == {0: 2, 1: 3, 2: 4, 3: 5, 4: 6}
    monkeypatch.setattr(cli, "Ai3Controller", lambda device="auto": fake_controller(code=4))
    assert cli.main(["--job", str(job_file)]) == cli.EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    result = json.loads(next(line for line in lines if line.startswith("RESULT "))[7:])
    assert result["status"] == "COMPLETED" and Path(result["output"]).is_file()
    assert any(line.startswith("PROGRESS ") for line in lines)


def test_cli_rejects_learn_without_fences(ptc, grid_cloud):
    with pytest.raises(SystemExit):
        cli.main(["--input", str(grid_cloud.path), "--ptc", str(ptc), "--learn"])


def _class_from_xy(cloud):
    x = np.asarray(cloud.x)
    y = np.asarray(cloud.y)
    return (2 + (np.floor(x) + 3 * np.floor(y)).astype(np.int64) % 4).astype(np.uint8)


def test_full_file_tiles_merge_back_in_source_order(ptc, fake_controller, tmp_path):
    """A 4 x 4 tile split with overlaps, in shuffled source order, merges back exactly."""
    rng = np.random.default_rng(3)
    gx, gy = np.meshgrid(np.arange(0, 400, 2.0), np.arange(0, 400, 2.0))
    xyz = np.c_[gx.ravel() + 600000.0, gy.ravel() + 5000000.0, rng.random(gx.size)]
    xyz = xyz[rng.permutation(len(xyz))]                         # source order != spatial order
    source = write_cloud(tmp_path / "big.laz", xyz, np.zeros(len(xyz), np.uint8))
    controller = fake_controller(fn=_class_from_xy)
    controller.spawn = lambda: fake_controller(fn=_class_from_xy)
    result = run_job(Ai3Job(source=source, ptc=ptc, tile_points=2_500, tile_workers=3,
                            fence_buffer_m=20.0), controller=controller)
    assert result.engine_report["tiles"] == 16
    out = laspy.read(result.output_path)
    np.testing.assert_array_equal(np.asarray(out.classification), _class_from_xy(out))
    np.testing.assert_array_equal(result.final_classes, np.asarray(out.classification))
    np.testing.assert_allclose(np.asarray(out.x), xyz[:, 0], atol=0.006)


class _GateEngine:
    """Fake engine with the tile options: reports a building intensity gate per tile."""

    calls = []

    def __init__(self, applied_for):
        self.applied_for = applied_for

    def spawn(self):
        return _GateEngine(self.applied_for)

    def cancel(self):
        pass

    def run(self, input_path, output_path, ptc_path, work_dir, power_lines=True,
            progress_callback=None, mapping=None, ground_unclass_pct=None,
            overrides=None, keep_step2=None, step3_from=None, grid_origin=None):
        cloud = laspy.read(input_path)
        cloud.classification = _class_from_xy(cloud)
        cloud.write(output_path)
        if keep_step2:
            Path(keep_step2).write_bytes(b"step2")
        gate = ((overrides or {}).get("building_recovery") or {}).get("intensity_gate")
        x_mean = float(np.mean(cloud.x))
        applied = (gate == "on") if gate else self.applied_for(x_mean)
        bins = np.zeros(256, dtype=int)
        cand, ground = bins.copy(), bins.copy()
        # tiles left of x=600200 see bright candidates, the others dim ones
        cand[180 if x_mean < 600200 else 60] = 1000
        ground[120] = 1000
        _GateEngine.calls.append(dict(x=x_mean, overrides=overrides, step3_from=step3_from,
                                      keep=keep_step2))
        report = {"building_recovery": {"intensity_applied": applied,
                                        "intensity_candidates_hist": cand.tolist(),
                                        "intensity_ground_hist": ground.tolist()}}
        return SimpleNamespace(report=report, warnings=[])


def test_tiles_share_one_building_decision_and_file_values(ptc, tmp_path):
    rng = np.random.default_rng(5)
    gx, gy = np.meshgrid(np.arange(0, 400, 2.0), np.arange(0, 400, 2.0))
    xyz = np.c_[gx.ravel() + 600000.0, gy.ravel() + 5000000.0, rng.random(gx.size)]
    source = write_cloud(tmp_path / "gate.laz", xyz, np.zeros(len(xyz), np.uint8))
    _GateEngine.calls = []
    # a tile decides on its own: only tiles left of x=600200 apply the gate
    engine = _GateEngine(lambda x: x < 600200)
    result = run_job(Ai3Job(source=source, ptc=ptc, tile_points=10_000, tile_workers=2,
                            fence_buffer_m=20.0), controller=engine)
    calls = _GateEngine.calls
    first = [c for c in calls if c["keep"]]
    redo = [c for c in calls if c["step3_from"]]
    later = [c for c in calls if not c["keep"] and not c["step3_from"]]
    assert result.engine_report["tiles"] == 4 and len(first) == 2
    pooled = _GateEngine.calls[0]                     # the two first tiles: one bright, one dim
    assert later and all(c["overrides"]["building_recovery"]["intensity_gate"] in ("on", "off")
                         for c in later)
    gate = later[0]["overrides"]["building_recovery"]["intensity_gate"]
    assert all(c["overrides"]["building_recovery"]["intensity_gate"] == gate for c in later + redo)
    # every tile got the file-wide threshold and density from the split pass
    for c in calls:
        assert c["overrides"]["building_recovery"]["intensity_threshold"] is not None
        assert c["overrides"]["roof_consolidation"]["point_density"] > 0
    # first-batch tiles that decided otherwise redid step 3 from their kept step 2
    disagreeing = [c for c in first if (c["x"] < 600200) != (gate == "on")]
    assert len(redo) == len(disagreeing)
    assert any(gate in w for w in result.warnings)
    out = laspy.read(result.output_path)
    np.testing.assert_array_equal(np.asarray(out.classification), _class_from_xy(out))
    assert pooled


def test_classified_copy_promotes_legacy_format_for_high_codes(tmp_path):
    xyz = np.c_[np.arange(9.0), np.zeros(9), np.zeros(9)]
    source = write_cloud(tmp_path / "legacy.las", xyz, np.zeros(9, np.uint8), point_format=3)
    classes = np.arange(32, 41, dtype=np.uint8)
    out = tmp_path / "out.las"
    lasio.write_classified_copy(source, classes, out)
    result = laspy.read(out)
    assert result.point_format.id == 7
    np.testing.assert_array_equal(np.asarray(result.classification), classes)
