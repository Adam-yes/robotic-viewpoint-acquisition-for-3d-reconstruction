"""
3D format converter: mesh ↔ point cloud ↔ mesh.

Supports OBJ, STL, PLY, OFF, GLTF/GLB, PCD, XYZ, PTS.
Requires Open3D; Trimesh is used as a fallback for mesh loading.

Usage:
    python src/preprocessing/convert_mesh.py /path/to/model.obj 50000
    python src/preprocessing/convert_mesh.py /path/to/model.stl .pcd
    python src/preprocessing/convert_mesh.py /path/to/model.obj .ply 50000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:
    _HAS_O3D = False
    logging.warning("Open3D not available. Install: pip install open3d")

try:
    import trimesh
    _HAS_TRIMESH = True
except ImportError:
    _HAS_TRIMESH = False
    logging.warning("Trimesh not available. Install: pip install trimesh")

logger = logging.getLogger(__name__)


class FormatConverter:
    """Universal 3D format converter (mesh ↔ point cloud).

    Supported mesh formats:   .obj .stl .ply .off .gltf .glb
    Supported PC formats:     .ply .pcd .xyz .pts
    """

    MESH_FORMATS = {".obj", ".stl", ".ply", ".off", ".gltf", ".glb"}
    PC_FORMATS   = {".ply", ".pcd", ".xyz", ".pts"}

    def __init__(self) -> None:
        if not _HAS_O3D and not _HAS_TRIMESH:
            raise RuntimeError("At least one of Open3D or Trimesh must be installed")

    @staticmethod
    def _ext(path: Path) -> str:
        return path.suffix.lower()

    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        sampling_density: Optional[int] = None,
        normalize: bool = False,
    ) -> bool:
        """Convert between 3D formats.

        Args:
            input_path:       Source file.
            output_path:      Destination file.
            sampling_density: Points to sample when converting mesh→point cloud.
            normalize:        Normalize geometry to unit sphere centered at origin.

        Returns:
            True on success.
        """
        input_path  = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            logger.error("Input file not found: %s", input_path)
            return False

        in_ext  = self._ext(input_path)
        out_ext = self._ext(output_path)
        logger.info("Converting %s → %s", in_ext, out_ext)

        try:
            if out_ext == ".ply" and sampling_density is not None and in_ext in self.MESH_FORMATS:
                return self._mesh_to_pc(input_path, output_path, sampling_density, normalize)
            elif in_ext in self.MESH_FORMATS and out_ext in self.MESH_FORMATS:
                return self._mesh_to_mesh(input_path, output_path, normalize)
            elif in_ext in self.MESH_FORMATS and out_ext in self.PC_FORMATS:
                return self._mesh_to_pc(input_path, output_path, sampling_density, normalize)
            elif in_ext in self.PC_FORMATS and out_ext in self.PC_FORMATS:
                return self._pc_to_pc(input_path, output_path, normalize)
            elif in_ext in self.PC_FORMATS and out_ext in self.MESH_FORMATS:
                return self._pc_to_mesh(input_path, output_path, normalize)
            else:
                logger.error("Unsupported conversion: %s → %s", in_ext, out_ext)
                return False
        except Exception as exc:
            logger.exception("Conversion failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Mesh → mesh
    # ------------------------------------------------------------------

    def _mesh_to_mesh(self, src: Path, dst: Path, normalize: bool) -> bool:
        if _HAS_O3D:
            mesh = o3d.io.read_triangle_mesh(str(src))
            if not mesh.has_vertices():
                logger.error("Failed to load mesh from %s", src)
                return False
            if normalize:
                mesh = self._norm_mesh(mesh)
            dst.parent.mkdir(parents=True, exist_ok=True)
            ok = o3d.io.write_triangle_mesh(str(dst), mesh)
            if ok:
                logger.info("Saved mesh: %d vertices, %d faces", len(mesh.vertices), len(mesh.triangles))
            return ok
        elif _HAS_TRIMESH:
            mesh = trimesh.load(str(src))
            if normalize:
                mesh.vertices -= mesh.vertices.mean(axis=0)
                mesh.vertices /= np.max(np.linalg.norm(mesh.vertices, axis=1))
            dst.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(str(dst))
            logger.info("Saved mesh: %d vertices", len(mesh.vertices))
            return True
        return False

    # ------------------------------------------------------------------
    # Mesh → point cloud
    # ------------------------------------------------------------------

    def _mesh_to_pc(
        self,
        src: Path,
        dst: Path,
        sampling_density: Optional[int],
        normalize: bool,
    ) -> bool:
        if _HAS_O3D:
            mesh = o3d.io.read_triangle_mesh(str(src))
            if not mesh.has_vertices():
                logger.error("Failed to load mesh from %s", src)
                return False
            n_pts = sampling_density or max(10_000, len(mesh.vertices) * 2)
            pcd = mesh.sample_points_uniformly(number_of_points=n_pts)
            if normalize:
                pcd = self._norm_pc(pcd)
            dst.parent.mkdir(parents=True, exist_ok=True)
            ok = o3d.io.write_point_cloud(str(dst), pcd)
            if ok:
                logger.info("Saved point cloud: %d points", len(pcd.points))
            return ok
        elif _HAS_TRIMESH:
            mesh  = trimesh.load(str(src))
            n_pts = sampling_density or max(10_000, len(mesh.vertices) * 2)
            points, _ = trimesh.sample.sample_surface(mesh, n_pts)
            if normalize:
                points -= points.mean(axis=0)
                points /= np.max(np.linalg.norm(points, axis=1))
            dst.parent.mkdir(parents=True, exist_ok=True)
            trimesh.points.PointCloud(points).export(str(dst))
            logger.info("Saved point cloud: %d points", len(points))
            return True
        return False

    # ------------------------------------------------------------------
    # Point cloud → point cloud
    # ------------------------------------------------------------------

    def _pc_to_pc(self, src: Path, dst: Path, normalize: bool) -> bool:
        if not _HAS_O3D:
            logger.error("Open3D required for PC→PC conversion")
            return False
        pcd = o3d.io.read_point_cloud(str(src))
        if not pcd.has_points():
            logger.error("Failed to load point cloud from %s", src)
            return False
        if normalize:
            pcd = self._norm_pc(pcd)
        dst.parent.mkdir(parents=True, exist_ok=True)
        ok = o3d.io.write_point_cloud(str(dst), pcd)
        if ok:
            logger.info("Saved point cloud: %d points", len(pcd.points))
        return ok

    # ------------------------------------------------------------------
    # Point cloud → mesh (Poisson)
    # ------------------------------------------------------------------

    def _pc_to_mesh(self, src: Path, dst: Path, normalize: bool) -> bool:
        if not _HAS_O3D:
            logger.error("Open3D required for PC→mesh conversion")
            return False
        pcd = o3d.io.read_point_cloud(str(src))
        if not pcd.has_points():
            return False
        if not pcd.has_normals():
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.01))
        if normalize:
            mesh = self._norm_mesh(mesh)
        dst.parent.mkdir(parents=True, exist_ok=True)
        ok = o3d.io.write_triangle_mesh(str(dst), mesh)
        if ok:
            logger.info("Saved mesh: %d vertices, %d faces", len(mesh.vertices), len(mesh.triangles))
        return ok

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_mesh(mesh: "o3d.geometry.TriangleMesh") -> "o3d.geometry.TriangleMesh":
        verts = np.asarray(mesh.vertices)
        verts -= verts.mean(axis=0)
        verts /= np.max(np.linalg.norm(verts, axis=1))
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        return mesh

    @staticmethod
    def _norm_pc(pcd: "o3d.geometry.PointCloud") -> "o3d.geometry.PointCloud":
        pts = np.asarray(pcd.points)
        pts -= pts.mean(axis=0)
        pts /= np.max(np.linalg.norm(pts, axis=1))
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def batch_convert(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        output_format: str,
        input_formats: Optional[List[str]] = None,
        **kwargs,
    ) -> int:
        """Convert all matching files in a directory. Returns success count."""
        input_dir  = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fmts = set(input_formats) if input_formats else self.MESH_FORMATS | self.PC_FORMATS
        files = [f for ext in fmts for f in input_dir.glob(f"*{ext}")]
        logger.info("Batch: %d files to convert → %s", len(files), output_format)

        success = 0
        for f in files:
            out = output_dir / f"{f.stem}{output_format}"
            if self.convert(f, out, **kwargs):
                success += 1
        logger.info("Batch done: %d/%d converted", success, len(files))
        return success


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D format converter")
    p.add_argument("input", help="Input 3D file")
    p.add_argument("format_or_density", nargs="?", default=".ply",
                   help="Output format (.ply, .pcd, …) or sampling density (integer)")
    p.add_argument("density", nargs="?", type=int, default=None,
                   help="Sampling density (mesh→point cloud)")
    p.add_argument("--normalize", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("File not found: %s", input_path)
        sys.exit(1)

    second = args.format_or_density
    if isinstance(second, str) and second.lstrip("-").isdigit():
        output_format   = ".ply"
        sampling_density = int(second)
    elif second is not None and second.startswith("."):
        output_format    = second
        sampling_density = args.density
    else:
        output_format    = ".ply"
        sampling_density = int(second) if second else None

    output_path = input_path.parent / f"{input_path.stem}{output_format}"
    converter   = FormatConverter()
    ok = converter.convert(
        input_path, output_path,
        sampling_density=sampling_density,
        normalize=args.normalize,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
