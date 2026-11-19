"""Keep source LAS records intact across tools that rewrite extra dimensions."""
import os
import tempfile

import laspy
import numpy as np
from las_compat import install_opaque_extra_bytes_fix

install_opaque_extra_bytes_fix()


def restore_source_records(source_path, processed_path, chunk_size=500_000):
    """Overlay classification and HAG, checking point order before replacement."""
    folder = os.path.dirname(os.path.abspath(processed_path))
    suffix = os.path.splitext(processed_path)[1]
    fd, temporary = tempfile.mkstemp(dir=folder, suffix=suffix)
    os.close(fd)
    try:
        with laspy.open(source_path) as source, laspy.open(processed_path) as processed:
            if source.header.point_count != processed.header.point_count:
                raise ValueError("[X] Point count mismatch restoring source dimensions")
            header = source.header.copy()
            if header.point_format.id < 6 and processed.header.point_format.id >= 6:
                # Inference promotes to LAS 1.4 for client classes above 31.
                extra = list(header.point_format.extra_dimensions)
                header.set_version_and_point_format(
                    processed.header.version,
                    laspy.PointFormat(processed.header.point_format.id))
                for dim in extra:
                    header.add_extra_dim(laspy.ExtraBytesParams(
                        name=dim.name, type=dim.dtype, description=dim.description,
                        scales=dim.scales, offsets=dim.offsets, no_data=dim.no_data))
            if "HeightAboveGround" not in header.point_format.dimension_names:
                header.add_extra_dim(laspy.ExtraBytesParams(
                    name="HeightAboveGround", type=np.float32))
            with laspy.open(temporary, mode="w", header=header) as writer:
                for original in source.chunk_iterator(chunk_size):
                    result = processed.read_points(len(original))
                    # Compare physical coordinates: writers may change LAS scales.
                    for axis in ("x", "y", "z"):
                        tolerance = max(source.header.scales["xyz".index(axis)],
                                        processed.header.scales["xyz".index(axis)]) * 0.51
                        if not np.allclose(original[axis], result[axis],
                                           rtol=0, atol=tolerance):
                            raise ValueError("[X] Processed cloud is not point-aligned with the input")
                    points = laspy.ScaleAwarePointRecord.zeros(len(original), header=header)
                    # Copy raw fields, including opaque arrays and scaled extra bytes.
                    if header.point_format.id == source.header.point_format.id:
                        for name in original.array.dtype.names:
                            points.array[name] = original.array[name]
                    else:
                        points.copy_fields_from(original)
                    points.classification = result.classification
                    points["HeightAboveGround"] = result["HeightAboveGround"]
                    writer.write_points(points)
        os.replace(temporary, processed_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
