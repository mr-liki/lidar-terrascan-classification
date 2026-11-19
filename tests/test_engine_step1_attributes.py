"""Run with the AI3 runtime's Python (requires PDAL, laspy and scipy)."""
import contextlib
import io
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import laspy
import numpy as np

try:
    import pdal  # noqa: F401
except ImportError:                     # only the AI3 runtime has the engine's dependencies
    raise unittest.SkipTest("needs the AI3 runtime (pdal)")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "naksha_ai3" / "engine"))
from step1_preprocess import restore_source_attributes, run_step1_preprocessing
from step2_inference import prepare_model_input
from naksha_ai_engine import ensure_extended_classification


class SourceAttributeTests(unittest.TestCase):
    def test_saturated_rgb_nir_passes_actual_myria_transform(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "source.las", Path(folder) / "model.las"
            header = laspy.LasHeader(point_format=8, version="1.4")
            header.add_extra_dim(laspy.ExtraBytesParams(name="HeightAboveGround", type=np.float32))
            cloud = laspy.LasData(header)
            values = np.array([0, 255, 256, 32768, 65279, 65280, 65281, 65534, 65535], dtype=np.uint16)
            cloud.x = np.arange(len(values), dtype=float)
            cloud.y = np.arange(len(values), dtype=float) * 2
            cloud.z = np.arange(len(values), dtype=float) * 0.5
            cloud.return_number = np.ones(len(values), dtype=np.uint8)
            cloud.number_of_returns = np.ones(len(values), dtype=np.uint8)
            cloud.HeightAboveGround = np.linspace(0, 10, len(values), dtype=np.float32)
            for channel in ("red", "green", "blue", "nir"):
                cloud[channel] = values
            cloud.write(source)
            original_bytes = source.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_model_input(str(source), str(output))
            result = laspy.read(output)
            for channel in ("red", "green", "blue", "nir"):
                np.testing.assert_array_equal(result[channel], np.minimum(values, 65280))
            np.testing.assert_array_equal(result.HeightAboveGround, cloud.HeightAboveGround)
            self.assertEqual(source.read_bytes(), original_bytes)

            transform_path = Path(__file__).resolve().parents[1] / "src/naksha_ai3/engine/myria3d/myria3d/pctl/points_pre_transform/lidar_hd.py"
            spec = importlib.util.spec_from_file_location("ai3_test_lidar_hd", transform_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            import pdal
            import json
            pipeline = pdal.Pipeline(json.dumps([{"type": "readers.las", "filename": str(output)}]))
            pipeline.execute()
            points = pipeline.arrays[0]
            # Myria's iterator hands floating-point features to the transform.
            points = points.astype([(name, np.float64) for name in points.dtype.names])
            data = module.lidar_hd_pre_transform(points)
            self.assertTrue(np.isfinite(data.x).all())
            self.assertEqual(module.COLORS_NORMALIZATION_MAX_VALUE, 65280)

    def test_model_preparation_keeps_hag_from_legacy_extra_bytes(self):
        for version in ("1.2", "1.4"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as folder:
                source, output = Path(folder) / "normalized.las", Path(folder) / "model.las"
                header = laspy.LasHeader(point_format=3, version=version)
                header.add_extra_dims([
                    laspy.ExtraBytesParams(name="Amplitude", type=np.float32),
                    laspy.ExtraBytesParams(name="HeightAboveGround", type=np.float32),
                ])
                cloud = laspy.LasData(header)
                cloud.x = np.arange(100, dtype=np.float64)
                cloud.y = np.arange(100, dtype=np.float64) * 2
                cloud.z = np.arange(100, dtype=np.float64) * 0.5
                cloud.Amplitude = np.zeros(100, dtype=np.float32)
                cloud.HeightAboveGround = np.linspace(-1, 15, 100, dtype=np.float32)
                cloud.write(source)
                with contextlib.redirect_stdout(io.StringIO()):
                    prepare_model_input(str(source), str(output))
                result = laspy.read(output)
                np.testing.assert_array_equal(result.HeightAboveGround, cloud.HeightAboveGround)
                for axis in ("X", "Y", "Z"):
                    np.testing.assert_array_equal(result[axis], cloud[axis])

    def make_cloud(self, path, version="1.4", padding_size=21):
        header = laspy.LasHeader(point_format=3, version=version)
        header.scales = [0.001] * 3
        header.offsets = [500_000, 5_000_000, 100]
        header.add_extra_dims([
            laspy.ExtraBytesParams(name="Amplitude", type=np.float32),
            laspy.ExtraBytesParams(name="Deviation", type=np.uint16),
            laspy.ExtraBytesParams(name="Pulse width", type=np.float32),
            laspy.ExtraBytesParams(name="Reflectance", type=np.float32,
                                  scales=[0.25], offsets=[-10]),
            laspy.ExtraBytesParams(name="ExtraBytes", type=f"{padding_size}u1"),
        ])
        if padding_size == 21:
            # Match the failing file: 21 unregistered bytes at each record's end.
            for vlr in header.vlrs.get("ExtraBytesVlr"):
                vlr.extra_bytes_structs[:] = [entry for entry in vlr.extra_bytes_structs
                                             if entry.format_name() != "ExtraBytes"]
        cloud = laspy.LasData(header)
        x, y = np.meshgrid(np.arange(80) * 0.5, np.arange(80) * 0.5)
        cloud.x = x.ravel() + header.offsets[0]
        cloud.y = y.ravel() + header.offsets[1]
        cloud.z = 100 + 0.03 * x.ravel()
        n = len(cloud.points)
        cloud.classification = np.ones(n, dtype=np.uint8)
        cloud.Amplitude = np.arange(n, dtype=np.float32) / 3
        cloud.Deviation = np.arange(n, dtype=np.uint16)
        cloud["Pulse width"] = np.arange(n, dtype=np.float32) / 7
        cloud.Reflectance = np.arange(n) * 0.25 - 10
        cloud.ExtraBytes = np.arange(n * padding_size, dtype=np.uint8).reshape(n, padding_size)
        cloud.red = np.arange(n, dtype=np.uint16)
        cloud.write(path)
        return cloud

    def test_real_pdal_preserves_custom_fields(self):
        for version, padding_size in (("1.2", 21), ("1.4", 21), ("1.4", 5)):
            with self.subTest(version=version, padding_size=padding_size), tempfile.TemporaryDirectory() as folder:
                source, output = Path(folder) / "source.laz", Path(folder) / "normalized.las"
                original = self.make_cloud(source, version, padding_size)
                with contextlib.redirect_stdout(io.StringIO()):
                    run_step1_preprocessing(str(source), str(output))
                result = laspy.read(output)
                self.assertEqual(len(result.points), len(original.points))
                self.assertEqual(result.HeightAboveGround.dtype, np.dtype("float32"))
                self.assertTrue(np.isfinite(result.HeightAboveGround).all())
                self.assertTrue(np.any(result.classification == 2))
                for name in original.point_format.dimension_names:
                    if name != "classification":
                        np.testing.assert_array_equal(result[name], original[name], err_msg=name)
                for name in original.points.array.dtype.names:
                    if name in {"Amplitude", "Deviation", "Pulse width", "Reflectance", "ExtraBytes"}:
                        self.assertEqual(result.points.array[name].tobytes(),
                                         original.points.array[name].tobytes())

    def test_legacy_fence_promotion_passes_real_pdal(self):
        for version, padding_size in (("1.2", 21), ("1.4", 21), ("1.4", 5)):
            with self.subTest(version=version, padding_size=padding_size), tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / "fence_local.laz"
                output = Path(folder) / "normalized.las"
                original = self.make_cloud(source, version, padding_size)
                before = source.read_bytes()
                promoted = ensure_extended_classification(
                    str(source), {"classes": {"ground": 1, "low_point": 51}}, folder)
                with contextlib.redirect_stdout(io.StringIO()):
                    run_step1_preprocessing(promoted, str(output))
                result = laspy.read(output)
                self.assertEqual(result.point_format.id, 7)
                self.assertEqual(len(result.points), len(original.points))
                self.assertTrue(np.isfinite(result.HeightAboveGround).all())
                self.assertTrue(np.any(result.classification == 2))
                for name in original.point_format.extra_dimension_names:
                    self.assertEqual(result.points.array[name].tobytes(),
                                     original.points.array[name].tobytes(), name)
                for axis in ("X", "Y", "Z"):
                    np.testing.assert_array_equal(result[axis], original[axis])
                self.assertEqual(source.read_bytes(), before)

    def test_chunked_restore_and_alignment_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "source.las", Path(folder) / "normalized.las"
            original = self.make_cloud(source)
            normalized = laspy.LasData(laspy.LasHeader(point_format=3, version="1.4"))
            normalized.header.scales = original.header.scales.copy()
            normalized.header.offsets = original.header.offsets.copy()
            normalized.X, normalized.Y, normalized.Z = original.X, original.Y, original.Z
            normalized.add_extra_dim(laspy.ExtraBytesParams(name="HeightAboveGround", type=np.float32))
            normalized.HeightAboveGround = np.arange(len(original.points), dtype=np.float32) / 10
            normalized.classification = np.full(len(original.points), 2, dtype=np.uint8)
            normalized.write(output)
            restore_source_attributes(str(source), str(output), chunk_size=513)
            result = laspy.read(output)
            np.testing.assert_array_equal(result.ExtraBytes, original.ExtraBytes)
            np.testing.assert_array_equal(result.HeightAboveGround, normalized.HeightAboveGround)
            np.testing.assert_array_equal(result.classification, normalized.classification)
            normalized.X[600] += 1
            normalized.write(output)
            before = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "point-aligned"):
                restore_source_attributes(str(source), str(output), chunk_size=513)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(list(Path(folder).glob("ai3_attributes_*")), [])


if __name__ == "__main__":
    unittest.main()
