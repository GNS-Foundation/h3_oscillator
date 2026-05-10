"""
H3 region setup for the H3-Oscillator experiments.

Provides:
- H3Region class: encapsulates a region of H3 cells with cached neighbor adjacency
- Index mapping: H3 cell IDs <-> dense integer indices for efficient computation
- Boundary coordinates for visualization

This module is consumed by the Gray-Scott data generator and (later) by the
H3-Oscillator architecture's encoder/decoder. By centralizing region setup
here, we ensure consistency across the dataset and the model.
"""
from __future__ import annotations

import numpy as np
import h3


class H3Region:
    """A region of H3 cells with precomputed adjacency and coordinate mapping.

    Attributes
    ----------
    center : str
        H3 cell ID of the region center.
    resolution : int
        H3 resolution (0-15). Determines cell size.
    k_ring : int
        Maximum graph distance from center to include.
    cells : np.ndarray of object (H3 cell IDs as strings)
        All cells in the region, in stable order.
    n_cells : int
        Number of cells (== len(cells)).
    cell_to_idx : dict[str, int]
        Mapping from H3 cell ID to dense index.
    neighbor_indices : np.ndarray of shape (n_cells, 6)
        For each cell, the dense indices of its 6 hex neighbors.
        Pentagonal cells (5 neighbors) have a -1 sentinel in the 6th slot,
        but we exclude pentagonals in v0.1.
    has_pentagons : bool
        Whether any pentagon cells appear in the region (always False in v0.1).
    """

    def __init__(self, center_lat: float, center_lon: float, resolution: int, k_ring: int):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.resolution = resolution
        self.k_ring = k_ring

        # Set up the cell region
        self.center = h3.latlng_to_cell(center_lat, center_lon, resolution)
        ring_set = h3.grid_disk(self.center, k_ring)

        # Sort cells for stable, reproducible ordering
        self.cells = np.array(sorted(ring_set), dtype=object)
        self.n_cells = len(self.cells)
        self.cell_to_idx = {cell: i for i, cell in enumerate(self.cells)}

        # Detect and (in v0.1) reject pentagonals
        pent_mask = np.array([h3.is_pentagon(c) for c in self.cells])
        self.has_pentagons = pent_mask.any()
        if self.has_pentagons:
            n_pent = int(pent_mask.sum())
            raise ValueError(
                f"Region contains {n_pent} pentagonal cell(s); v0.1 requires "
                f"a region without pentagons. Choose a different center or "
                f"reduce k_ring."
            )

        # Build neighbor index lookup: shape (n_cells, 6).
        # For each cell, find its 6 hex neighbors by computing grid_disk(c, 1)
        # excluding c itself, then mapping to dense indices.
        self.neighbor_indices = np.full((self.n_cells, 6), -1, dtype=np.int64)
        for i, cell in enumerate(self.cells):
            ring = h3.grid_disk(cell, 1)
            neighbors = [n for n in ring if n != cell]
            # Some neighbors may be outside our region (boundary cells) - we
            # mark those with -1 sentinel.
            valid_neighbors = []
            for n in neighbors:
                if n in self.cell_to_idx:
                    valid_neighbors.append(self.cell_to_idx[n])
                # else: boundary cell, neighbor is outside region
            for j, idx in enumerate(valid_neighbors[:6]):
                self.neighbor_indices[i, j] = idx

        # Cache cell center lat/lons (used for visualization and positional encoding)
        self._cell_centers = None

    @property
    def cell_centers(self) -> np.ndarray:
        """Return (n_cells, 2) array of (lat, lon) centers for each cell."""
        if self._cell_centers is None:
            centers = np.array([h3.cell_to_latlng(c) for c in self.cells])
            self._cell_centers = centers
        return self._cell_centers

    def cell_boundary(self, idx: int) -> np.ndarray:
        """Return the polygon boundary of cell at index `idx` as (N, 2) lat/lon array.

        Used for visualization with matplotlib polygon patches.
        """
        cell = self.cells[idx]
        boundary = h3.cell_to_boundary(cell)  # list of (lat, lon) tuples
        return np.array(boundary)

    def n_valid_neighbors(self) -> np.ndarray:
        """Return count of valid (in-region) neighbors per cell, shape (n_cells,).

        Used for boundary-aware Laplacian computation: cells at the region boundary
        have fewer valid neighbors and the Laplacian should be normalized accordingly.
        """
        return (self.neighbor_indices >= 0).sum(axis=1)

    def __repr__(self) -> str:
        return (
            f"H3Region(center=({self.center_lat:.2f}, {self.center_lon:.2f}), "
            f"resolution={self.resolution}, k_ring={self.k_ring}, "
            f"n_cells={self.n_cells})"
        )


def laplacian(field: np.ndarray, region: H3Region) -> np.ndarray:
    """Compute the discrete hex Laplacian of a scalar field on H3 cells.

    The Laplacian is:
        L f(c) = mean(f(c') for c' in neighbors of c) - f(c)

    Boundary cells (with fewer than 6 valid neighbors in-region) use only
    their valid neighbors. This is the standard "no-flux" boundary condition.

    Parameters
    ----------
    field : np.ndarray of shape (n_cells,) or (n_cells, F)
        Scalar field defined on each H3 cell. The last axis (if multi-channel)
        is treated independently.
    region : H3Region
        The region the field lives on.

    Returns
    -------
    lap : np.ndarray of same shape as field
        The Laplacian.
    """
    nbr_idx = region.neighbor_indices  # (n_cells, 6), with -1 for missing
    n_valid = region.n_valid_neighbors()  # (n_cells,)

    # Use 0-padding for missing neighbors; this is safe because we'll divide
    # by n_valid (the actual count of in-region neighbors).
    valid_mask = nbr_idx >= 0  # (n_cells, 6)
    safe_idx = np.where(valid_mask, nbr_idx, 0)  # (n_cells, 6)

    if field.ndim == 1:
        gathered = field[safe_idx]  # (n_cells, 6)
        gathered = gathered * valid_mask  # zero out invalid neighbor contributions
        nbr_sum = gathered.sum(axis=1)  # (n_cells,)
        # Mean of valid neighbors:
        nbr_mean = nbr_sum / np.maximum(n_valid, 1)
        return nbr_mean - field
    else:
        # field is (n_cells, F)
        gathered = field[safe_idx]  # (n_cells, 6, F)
        gathered = gathered * valid_mask[:, :, None]
        nbr_sum = gathered.sum(axis=1)  # (n_cells, F)
        nbr_mean = nbr_sum / np.maximum(n_valid, 1)[:, None]
        return nbr_mean - field


if __name__ == "__main__":
    # Smoke test
    region = H3Region(center_lat=45.0, center_lon=0.0, resolution=5, k_ring=16)
    print(region)
    print(f"Pentagons in region: {region.has_pentagons}")
    print(f"Neighbor counts: min={region.n_valid_neighbors().min()}, "
          f"max={region.n_valid_neighbors().max()}, "
          f"interior cells (with all 6 neighbors): "
          f"{(region.n_valid_neighbors() == 6).sum()}")

    # Test Laplacian on a simple field
    f = np.ones(region.n_cells)  # constant field
    lap = laplacian(f, region)
    print(f"\nLaplacian of constant 1.0 field: max abs = {np.abs(lap).max():.6e} "
          f"(should be ~0)")

    # Random field test
    rng = np.random.default_rng(42)
    f = rng.standard_normal(region.n_cells)
    lap = laplacian(f, region)
    print(f"Laplacian of random field: shape {lap.shape}, "
          f"mean={lap.mean():.4f}, std={lap.std():.4f}")
