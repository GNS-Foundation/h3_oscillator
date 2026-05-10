"""
Visualization for H3 cell fields.

Renders each H3 cell as a colored polygon, with color determined by a scalar
field value. Used for:
  - Sanity-checking Gray-Scott pattern formation
  - Comparing predicted vs ground-truth fields in the experiment
  - Visualizing per-frequency energy from the DFT diagnostic head (later)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from .h3_region import H3Region


def plot_field(
    field: np.ndarray,
    region: H3Region,
    ax=None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    show_colorbar: bool = True,
) -> tuple:
    """Plot a scalar field on H3 cells as colored polygons.

    Parameters
    ----------
    field : np.ndarray of shape (n_cells,)
        Scalar values per cell.
    region : H3Region
    ax : matplotlib axes, optional
        If None, a new figure is created.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Color scale range. If None, use field's data range.
    title : str, optional
    show_colorbar : bool

    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    if vmin is None:
        vmin = float(field.min())
    if vmax is None:
        vmax = float(field.max())

    # Build polygon patches for each cell.
    # h3 returns boundaries as (lat, lon); we plot lon on x-axis, lat on y-axis.
    patches = []
    for i in range(region.n_cells):
        boundary = region.cell_boundary(i)  # (N, 2) of (lat, lon)
        # Swap to (lon, lat) for plotting
        xy = boundary[:, [1, 0]]
        patches.append(Polygon(xy, closed=True))

    # Color the patches
    cmap_obj = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap_obj(norm(field))

    pc = PatchCollection(patches, facecolors=colors, edgecolors="none")
    ax.add_collection(pc)

    # Set axis limits to the full extent of cells
    centers = region.cell_centers
    lat_min, lat_max = centers[:, 0].min(), centers[:, 0].max()
    lon_min, lon_max = centers[:, 1].min(), centers[:, 1].max()
    pad_lat = 0.05 * (lat_max - lat_min)
    pad_lon = 0.05 * (lon_max - lon_min)
    ax.set_xlim(lon_min - pad_lon, lon_max + pad_lon)
    ax.set_ylim(lat_min - pad_lat, lat_max + pad_lat)
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    if title:
        ax.set_title(title)

    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    return fig, ax


def plot_trajectory_panels(
    trajectory: np.ndarray,
    region: H3Region,
    field_idx: int = 1,  # 0 for u, 1 for v
    n_panels: int = 6,
    cmap: str = "viridis",
    title_prefix: str = "frame",
    figsize: tuple | None = None,
) -> "plt.Figure":
    """Plot multiple frames of a trajectory in a grid.

    Parameters
    ----------
    trajectory : np.ndarray of shape (n_frames, n_cells, 2)
    region : H3Region
    field_idx : int, default 1
        Which field to plot (0 for u, 1 for v). v often shows clearer patterns.
    n_panels : int
        Number of frames to show. Evenly spaced through the trajectory.
    cmap : str
    title_prefix : str
    figsize : tuple, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    n_frames = trajectory.shape[0]
    panel_indices = np.linspace(0, n_frames - 1, n_panels).astype(int)

    n_cols = min(n_panels, 3)
    n_rows = (n_panels + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (5 * n_cols, 5 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    # Use a global color scale across all panels
    field_name = "u" if field_idx == 0 else "v"
    all_values = trajectory[panel_indices, :, field_idx]
    vmin, vmax = float(all_values.min()), float(all_values.max())

    for ax_idx, frame_idx in enumerate(panel_indices):
        row, col = divmod(ax_idx, n_cols)
        ax = axes[row, col]
        field = trajectory[frame_idx, :, field_idx]
        plot_field(
            field, region, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
            title=f"{title_prefix} {frame_idx}", show_colorbar=False,
        )

    # Single colorbar for the figure
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label=field_name)

    # Hide unused subplots
    for ax_idx in range(n_panels, n_rows * n_cols):
        row, col = divmod(ax_idx, n_cols)
        axes[row, col].set_visible(False)

    return fig
