"""
Unit tests for the Path Matters pipeline.

Tests cover:
  - test_center_mesh()          — bring_to_center.py
  - test_convert_mesh()         — convert_mesh.py
  - test_config_loading()       — pipeline_config.yaml
  - test_visualization_import() — visualization.py
  - test_coverage_tracker()     — rl/viewpoint_env.py
  - test_proximity_reward()     — rl/viewpoint_env.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add repo root to path so src/ is importable without installation
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_temp_stl(path: Path) -> None:
    """Write a minimal ASCII STL cube to *path*."""
    stl_content = """solid cube
  facet normal 0 0 -1
    outer loop
      vertex 1 0 0
      vertex 0 0 0
      vertex 0 1 0
    endloop
  endfacet
  facet normal 0 0 -1
    outer loop
      vertex 1 0 0
      vertex 0 1 0
      vertex 1 1 0
    endloop
  endfacet
endsolid cube
"""
    path.write_text(stl_content)


def make_temp_ply(path: Path, n_points: int = 10) -> None:
    """Write a minimal PLY point cloud to *path*."""
    rng    = np.random.default_rng(0)
    points = rng.uniform(-1, 1, (n_points, 3))
    header = (
        f"ply\nformat ascii 1.0\nelement vertex {n_points}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    lines  = "\n".join(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in points)
    path.write_text(header + lines + "\n")


# ---------------------------------------------------------------------------
# test_center_mesh
# ---------------------------------------------------------------------------

def test_center_mesh():
    """bring_to_center.py should translate the mesh centroid to the origin."""
    try:
        import trimesh
    except ImportError:
        pytest.skip("trimesh not installed")

    from src.preprocessing.bring_to_center import center_mesh

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        src = tmpdir / "mesh.stl"
        dst = tmpdir / "mesh_centered.stl"
        make_temp_stl(src)

        center_mesh(src, dst)
        assert dst.exists(), "Output file was not created"

        mesh = trimesh.load(str(dst))
        centroid = mesh.centroid
        np.testing.assert_allclose(
            centroid, [0.0, 0.0, 0.0], atol=1e-5,
            err_msg=f"Centroid after centering should be (0,0,0), got {centroid}",
        )


# ---------------------------------------------------------------------------
# test_convert_mesh
# ---------------------------------------------------------------------------

def test_convert_mesh():
    """convert_mesh.py should convert a PLY point cloud to another PLY."""
    try:
        import open3d as o3d
    except ImportError:
        pytest.skip("open3d not installed")

    from src.preprocessing.convert_mesh import FormatConverter

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir   = Path(tmpdir)
        src      = tmpdir / "cloud.ply"
        dst      = tmpdir / "cloud_copy.ply"
        make_temp_ply(src, n_points=20)

        converter = FormatConverter()
        ok = converter.convert(src, dst)
        assert ok,          "Conversion returned False (failure)"
        assert dst.exists(), "Output PLY was not created"

        pcd_out = o3d.io.read_point_cloud(str(dst))
        assert len(pcd_out.points) == 20, (
            f"Expected 20 points, got {len(pcd_out.points)}"
        )


def test_convert_mesh_normalize():
    """FormatConverter.convert with normalize=True should scale points to unit sphere."""
    try:
        import open3d as o3d
    except ImportError:
        pytest.skip("open3d not installed")

    from src.preprocessing.convert_mesh import FormatConverter

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        src    = tmpdir / "big.ply"
        dst    = tmpdir / "norm.ply"
        # Build a large cloud intentionally far from origin
        rng    = np.random.default_rng(1)
        pts    = rng.uniform(100, 200, (50, 3))
        header = (
            f"ply\nformat ascii 1.0\nelement vertex 50\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
        )
        lines  = "\n".join(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in pts)
        src.write_text(header + lines + "\n")

        converter = FormatConverter()
        ok = converter.convert(src, dst, normalize=True)
        assert ok

        pcd_out = o3d.io.read_point_cloud(str(dst))
        arr     = np.asarray(pcd_out.points)
        max_r   = np.max(np.linalg.norm(arr, axis=1))
        assert max_r <= 1.0 + 1e-4, (
            f"Normalised points should lie within unit sphere, max radius={max_r:.4f}"
        )


# ---------------------------------------------------------------------------
# test_config_loading
# ---------------------------------------------------------------------------

def test_config_loading():
    """pipeline_config.yaml should be valid YAML with required sections."""
    import yaml

    config_path = REPO_ROOT / "config" / "pipeline_config.yaml"
    assert config_path.exists(), f"Config not found: {config_path}"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    assert cfg is not None, "Config parsed as None — empty YAML?"

    required_sections = ["capture", "reconstruction", "vggt", "preprocessing",
                         "ransac", "icp", "adaptive", "registration", "rl"]
    for section in required_sections:
        assert section in cfg, f"Missing required config section: '{section}'"

    # Spot-check key values against published best parameters
    assert cfg["vggt"]["confidence_threshold"] == 4.0, (
        "conf_threshold should be 4.0 (best-found value)"
    )
    assert cfg["reconstruction"]["max_images"] == 54, (
        "max_images should match 54-view capture protocol"
    )
    assert cfg["preprocessing"]["no_plane_removal"] is True, (
        "no_plane_removal should be True (best-found setting)"
    )


# ---------------------------------------------------------------------------
# test_visualization_import
# ---------------------------------------------------------------------------

def test_visualization_import():
    """visualization.py should import cleanly and expose expected functions."""
    from src.utils import visualization

    for fn_name in (
        "load_pcd",
        "view_pcd",
        "overlay_pcds",
        "save_overlay_screenshot",
        "compare_clouds",
        "print_alignment_summary",
    ):
        assert hasattr(visualization, fn_name), (
            f"visualization module missing expected function: {fn_name}"
        )


def test_visualization_print_summary(capsys):
    """print_alignment_summary should produce output with the metric values."""
    from src.utils.visualization import print_alignment_summary

    print_alignment_summary(fitness=0.93, rmse=0.002, n_correspondences=4200, elapsed_sec=7.0)
    captured = capsys.readouterr()
    assert "0.93" in captured.out
    assert "0.002" in captured.out
    assert "EXCELLENT" in captured.out


# ---------------------------------------------------------------------------
# test_coverage_tracker
# ---------------------------------------------------------------------------

def test_coverage_tracker():
    """VoxelCoverageTracker should correctly track covered voxels."""
    import torch
    from src.rl.viewpoint_env import VoxelCoverageTracker

    bounds = {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 1.0)}
    tracker = VoxelCoverageTracker(grid_size=(8, 8, 8), workspace_bounds=bounds, device="cpu")

    assert tracker.coverage_ratio == 0.0, "Initial coverage should be 0"

    # Observe some points
    pts = torch.tensor([[0.0, 0.0, 0.5], [0.5, 0.5, 0.5], [-0.5, -0.5, 0.3]])
    delta = tracker.update(pts)
    assert delta > 0.0, "Delta coverage should increase after observation"

    cov = tracker.coverage_ratio
    assert 0.0 < cov < 1.0

    # Reset should clear everything
    tracker.reset()
    assert tracker.coverage_ratio == 0.0, "Coverage should be 0 after reset"


# ---------------------------------------------------------------------------
# test_proximity_reward
# ---------------------------------------------------------------------------

def test_proximity_reward():
    """proximity_shaping_reward should decay with distance."""
    import torch
    from src.rl.viewpoint_env import proximity_shaping_reward

    eef = torch.tensor([0.0, 0.0, 0.0])

    close_pts = torch.tensor([[0.01, 0.0, 0.0]])
    far_pts   = torch.tensor([[5.00, 0.0, 0.0]])

    r_close = proximity_shaping_reward(eef, close_pts, sigma=0.1)
    r_far   = proximity_shaping_reward(eef, far_pts,   sigma=0.1)

    assert r_close > r_far, (
        f"Proximity reward should be higher for closer targets: close={r_close:.4f}  far={r_far:.4f}"
    )
    assert 0.0 <= r_close <= 1.0, f"Reward out of [0,1]: {r_close}"
    assert 0.0 <= r_far   <= 1.0, f"Reward out of [0,1]: {r_far}"
