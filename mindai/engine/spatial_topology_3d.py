"""BrainGeometry — Fibonacci-sphere coordinates for axonal-delay calculation.

Coordinates only — never the full N×N distance matrix (would be 3.6 TB for
the 400k-neuron default config).  Pairwise distances are computed on-demand
in vectorised form by ``build_delay_tensor`` (engine/axonal_delays.py),
which only needs distances for *existing* edges, i.e. O(synapses).
"""

import numpy as np


class BrainGeometry:

    def __init__(self, num_nodes: int, radius: float = 10.0):
        self.num_nodes   = num_nodes
        self.radius      = radius
        self.coordinates = self._generate_spherical_coordinates()

    def _generate_spherical_coordinates(self) -> np.ndarray:
        # Vectorised Fibonacci sphere — O(N), no Python loop
        n   = self.num_nodes
        idx = np.arange(n, dtype=np.float64)
        phi   = np.arccos(1.0 - 2.0 * (idx + 0.5) / n)
        theta = np.pi * (1.0 + 5.0 ** 0.5) * idx
        sin_phi = np.sin(phi)
        coords = np.empty((n, 3), dtype=np.float32)
        coords[:, 0] = self.radius * np.cos(theta) * sin_phi
        coords[:, 1] = self.radius * np.sin(theta) * sin_phi
        coords[:, 2] = self.radius * np.cos(phi)
        return coords

    def optimize_spatial_locality(self, layout) -> None:
        """Sort coordinates within each non-overlapping layout channel using Morton curves.

        This groups spatially adjacent neurons together in memory, maximizing CPU/GPU
        cache locality for synaptic operations and sparse tensor computations.
        """
        # Get all channels from layout
        channels = []
        for name, (start, end) in layout._ch.items():
            channels.append((start, end, name))

        # Sort by size descending to identify parent/sub-slices
        channels.sort(key=lambda x: (x[1] - x[0]), reverse=True)

        independent_slices = []
        for start, end, name in channels:
            # Check if this slice is a sub-slice of any existing independent slice
            is_sub = False
            for p_start, p_end, _ in independent_slices:
                if start >= p_start and end <= p_end:
                    is_sub = True
                    break
            if not is_sub:
                independent_slices.append((start, end, name))

        # Now sort the independent slices by start index to process them in order
        independent_slices.sort(key=lambda x: x[0])

        new_coords = self.coordinates.copy()

        # Within each independent slice, sort coordinates by Morton code
        for start, end, name in independent_slices:
            if end - start <= 1:
                continue
            slice_coords = self.coordinates[start:end]
            morton_codes = get_morton_codes_np(slice_coords)
            sort_idx = np.argsort(morton_codes)
            new_coords[start:end] = slice_coords[sort_idx]

        self.coordinates = new_coords

    def axonal_delay(self, node_a: int, node_b: int,
                     speed_of_conduction: float = 2.0) -> int:
        """Compute one pairwise delay on demand (O(1), no preallocated matrix)."""
        d = float(np.linalg.norm(self.coordinates[node_a] - self.coordinates[node_b]))
        return max(1, int(d / speed_of_conduction))


def get_morton_codes_np(coords: np.ndarray) -> np.ndarray:
    """Compute 3D Morton codes (Z-order curve) for a set of 3D coordinates.

    Normalizes coordinates to [0, 1023] integers and interleaves their bits.
    """
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    span = coords_max - coords_min
    span[span == 0] = 1.0
    norm = (coords - coords_min) / span
    xyz = (norm * 1023.0).astype(np.int64)

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    def split_by_3(a):
        a &= 0x3ff
        a = (a | (a << 16)) & 0x30000ff
        a = (a | (a << 8))  & 0x300f00f
        a = (a | (a << 4))  & 0x30c30c3
        a = (a | (a << 2))  & 0x9249249
        return a

    return (split_by_3(x) << 2) | (split_by_3(y) << 1) | split_by_3(z)