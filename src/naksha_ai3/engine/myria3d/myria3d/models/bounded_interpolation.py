"""CPU inverse-distance interpolation with bounded neighbour-query storage."""
import numpy as np
import torch
from scipy.spatial import cKDTree


def bounded_knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k,
                            chunk_points=16_384):
    """Match PyG's squared-distance weights without a whole-batch KNN allocation.

    Build one spatial index per cloud, then query a bounded number of original
    points at a time. Never mix neighbours from different tiles in a batch.
    This function is for CPU inference only (no gradients).
    """
    if chunk_points < 1 or k < 1:
        raise ValueError("Interpolation chunk size and neighbour count must be positive")
    output = x.new_empty((len(pos_y), x.shape[1]))
    for cloud in torch.unique(batch_y):
        source = torch.nonzero(batch_x == cloud, as_tuple=False).flatten()
        target = torch.nonzero(batch_y == cloud, as_tuple=False).flatten()
        if not len(source):
            raise ValueError("Interpolation tile has no sampled points")
        positions = pos_x[source]
        features = x[source]
        tree = cKDTree(positions.numpy())
        neighbours = min(k, len(source))
        for start in range(0, len(target), chunk_points):
            rows = target[start:start + chunk_points]
            query = pos_y[rows]
            _, indices = tree.query(query.numpy(), k=neighbours, workers=1)
            indices = torch.from_numpy(np.asarray(indices).reshape(len(rows), neighbours)).long()
            distance_squared = ((positions[indices] - query[:, None, :]) ** 2).sum(dim=-1)
            weights = distance_squared.clamp(min=1e-16).reciprocal()
            output[rows] = ((features[indices] * weights[:, :, None]).sum(dim=1)
                            / weights.sum(dim=1, keepdim=True))
    return output
