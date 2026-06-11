"""
Open3D visualization helpers for point cloud inspection and overlay rendering.

Functions cover the most common inspection tasks in the Path Matters pipeline:
single-cloud viewing, before/after alignment overlays, and side-by-side
comparison of reconstructed vs ground-truth geometry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:
    _HAS_O3D = False
    logger.warning("Open3D not available — visualization functions will be stubs")


PathLike = Union[str, Path]

# Colour palette used in figures
COLOUR_RECONSTRUCTION = [1.0, 0.706, 0.0]   # amber  — reconstructed cloud
COLOUR_GROUND_TRUTH   = [0.0, 0.651, 0.929] # cyan   — GT mesh / cloud
COLOUR_ALIGNED        = [0.0, 0.8,   0.2]   # green  — after alignment


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pcd(
    path: PathLike,
    sample_points: int = 100_000,
) -> "o3d.geometry.PointCloud":
    """Load a PLY/PCD file as an Open3D PointCloud.

    If the file is a mesh, it is sampled uniformly to *sample_points* points.

    Args:
        path:          Path to a .ply, .pcd, .obj, .stl, or similar file.
        sample_points: Points to sample when converting from mesh.

    Returns:
        Open3D PointCloud.
    """
    if not _HAS_O3D:
        raise ImportError("Open3D is required for visualization")
    path = Path(path)
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.has_points():
        return pcd

    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if not mesh.has_vertices():
        raise RuntimeError(f"Could not load any geometry from {path}")
    if mesh.has_triangles():
        return mesh.sample_points_uniformly(number_of_points=max(sample_points, 1_000))

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = mesh.vertices
    if mesh.has_vertex_colors():
        pcd2.colors = mesh.vertex_colors
    return pcd2


# ---------------------------------------------------------------------------
# Single-cloud viewer
# ---------------------------------------------------------------------------

def view_pcd(
    pcd: "o3d.geometry.PointCloud",
    title: str = "Point Cloud",
    width: int = 1280,
    height: int = 720,
) -> None:
    """Display a single point cloud in an interactive window.

    Args:
        pcd:    Open3D PointCloud to display.
        title:  Window title.
        width:  Window width in pixels.
        height: Window height in pixels.
    """
    if not _HAS_O3D:
        logger.warning("Open3D unavailable — skipping visualization")
        return
    logger.info("Visualizing: %s (%d points)", title, len(pcd.points))
    o3d.visualization.draw_geometries([pcd], window_name=title, width=width, height=height)


# ---------------------------------------------------------------------------
# Overlay (before/after alignment)
# ---------------------------------------------------------------------------

def overlay_pcds(
    source: "o3d.geometry.PointCloud",
    target: "o3d.geometry.PointCloud",
    transform: Optional[np.ndarray] = None,
    title: str = "Overlay",
    source_colour: Sequence[float] = COLOUR_RECONSTRUCTION,
    target_colour: Sequence[float] = COLOUR_GROUND_TRUTH,
    width: int = 1600,
    height: int = 900,
) -> None:
    """Display two point clouds overlaid with distinct colours.

    Args:
        source:        Source (reconstructed) point cloud.
        target:        Target (GT) point cloud.
        transform:     Optional 4x4 transformation to apply to *source*.
        title:         Window title.
        source_colour: RGB colour for the source cloud.
        target_colour: RGB colour for the target cloud.
        width / height: Window dimensions.
    """
    if not _HAS_O3D:
        return
    import copy
    src = copy.deepcopy(source)
    tgt = copy.deepcopy(target)
    src.paint_uniform_color(list(source_colour))
    tgt.paint_uniform_color(list(target_colour))
    if transform is not None:
        src.transform(transform)
    o3d.visualization.draw_geometries([src, tgt], window_name=title, width=width, height=height)


# ---------------------------------------------------------------------------
# Screenshot (headless)
# ---------------------------------------------------------------------------

def save_overlay_screenshot(
    source: "o3d.geometry.PointCloud",
    target: "o3d.geometry.PointCloud",
    out_png: PathLike,
    transform: Optional[np.ndarray] = None,
    width: int = 1600,
    height: int = 900,
) -> None:
    """Save a headless rendering of source+target overlay to a PNG file.

    Args:
        source:    Source point cloud.
        target:    Target point cloud.
        out_png:   Output PNG path.
        transform: Optional 4x4 transform applied to *source*.
        width / height: Render resolution.
    """
    if not _HAS_O3D:
        return
    import copy
    out_png = Path(out_png)
    src = copy.deepcopy(source)
    tgt = copy.deepcopy(target)
    src.paint_uniform_color(COLOUR_RECONSTRUCTION)
    tgt.paint_uniform_color(COLOUR_GROUND_TRUTH)
    if transform is not None:
        src.transform(transform)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(src)
    vis.add_geometry(tgt)
    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    vis.capture_screen_image(str(out_png), do_render=True)
    vis.destroy_window()
    logger.info("Saved overlay screenshot: %s", out_png)


# ---------------------------------------------------------------------------
# Multi-cloud comparison
# ---------------------------------------------------------------------------

def compare_clouds(
    clouds: List[Tuple[str, "o3d.geometry.PointCloud"]],
    colours: Optional[List[Sequence[float]]] = None,
    title: str = "Comparison",
    width: int = 1600,
    height: int = 900,
) -> None:
    """Display multiple point clouds simultaneously for visual comparison.

    Args:
        clouds:  List of (label, pcd) pairs.
        colours: Optional per-cloud RGB colours.  Auto-assigned if None.
        title:   Window title.
    """
    if not _HAS_O3D:
        return
    import copy

    auto_colours = [
        [1.0, 0.0, 0.0],
        [0.0, 0.65, 0.93],
        [0.0, 0.8, 0.2],
        [1.0, 0.7, 0.0],
    ]
    geometries = []
    for i, (label, pcd) in enumerate(clouds):
        c = copy.deepcopy(pcd)
        colour = (colours[i] if colours and i < len(colours)
                  else auto_colours[i % len(auto_colours)])
        c.paint_uniform_color(colour)
        logger.info("  %s: %d points, colour=%s", label, len(c.points), colour)
        geometries.append(c)

    o3d.visualization.draw_geometries(geometries, window_name=title, width=width, height=height)


# ---------------------------------------------------------------------------
# Metrics text overlay (console only — Open3D has no text HUD)
# ---------------------------------------------------------------------------

def print_alignment_summary(
    fitness: float,
    rmse: float,
    n_correspondences: int,
    elapsed_sec: float,
) -> None:
    """Print a formatted alignment quality summary to stdout."""
    quality = (
        "EXCELLENT" if fitness >= 0.85
        else "GOOD" if fitness >= 0.60
        else "MODERATE"
    )
    print(f"\n{'='*50}")
    print(f"  Alignment Summary")
    print(f"{'='*50}")
    print(f"  Fitness:          {fitness:.4f}  ({quality})")
    print(f"  Inlier RMSE:      {rmse:.6f} m")
    print(f"  Correspondences:  {n_correspondences}")
    print(f"  Elapsed:          {elapsed_sec:.1f} s")
    print(f"{'='*50}\n")
