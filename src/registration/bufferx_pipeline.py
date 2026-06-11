"""
CloudCompare / Open3D + BUFFER-X + ICP evaluation pipeline orchestrator.

Stages per scene:
  1. (Optional) Manual cleanup / alignment via Open3D or CloudCompare
  2. BUFFER-X global registration (initial transformation estimate)
  3. Open3D ICP refinement + evaluation
  4. Metrics aggregation to batch_summary.csv

Usage:
    conda run -n bufferx_o3d python src/registration/bufferx_pipeline.py \\
        --recon-root  /path/to/reconstructions \\
        --gt-root     /path/to/groundtruth \\
        --output-base /path/to/runs \\
        --bufferx-root ~/BUFFER-X \\
        --bufferx-env  bufferx_o3d \\
        --manual-mode  off

See config/pipeline_config.yaml for tunable parameters.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

DEFAULT_RECON_CANDIDATES = ["textured.ply", "textured.obj", "sparse/points.ply"]
DEFAULT_GT_CANDIDATES    = ["object.ply", "object.obj", "textured.ply", "points.ply"]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str]) -> dict:
    """Load registration section of pipeline_config.yaml."""
    defaults = {
        "registration": {
            "sample_points": 120_000,
            "max_points": 30_000,
            "icp_downsample_voxel": 0.0,
            "icp_max_correspondence": 0.0,
        }
    }
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        if "registration" in user_cfg:
            defaults["registration"].update(user_cfg["registration"])
    return defaults


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BUFFER-X + ICP batch evaluation pipeline")
    p.add_argument("--recon-root",   required=True)
    p.add_argument("--gt-root",      required=True)
    p.add_argument("--output-base",  default=".")
    p.add_argument("--run-name",     default=None)
    p.add_argument("--scene-names",  nargs="*", default=None)
    p.add_argument("--config",       default="config/pipeline_config.yaml")

    p.add_argument("--bufferx-root", required=True)
    p.add_argument("--bufferx-env",  default="bufferx_o3d")
    p.add_argument("--experiment-id", default="threedmatch")

    p.add_argument("--recon-candidates", nargs="*", default=DEFAULT_RECON_CANDIDATES)
    p.add_argument("--gt-candidates",    nargs="*", default=DEFAULT_GT_CANDIDATES)

    p.add_argument("--manual-mode",
                   choices=["off", "prefer", "require", "prepare"], default="prefer")
    p.add_argument("--manual-backend",
                   choices=["cloudcompare", "open3d"], default="cloudcompare")

    p.add_argument("--open3d-crop",              action="store_true")
    p.add_argument("--open3d-with-scaling",      action="store_true")
    p.add_argument("--open3d-pre-scale",
                   choices=["off", "aabb_diag", "aabb_max_extent"], default="off")
    p.add_argument("--open3d-sample-points",     type=int, default=120_000)
    p.add_argument("--open3d-reuse-cropped-recon", action="store_true")
    p.add_argument("--open3d-preview-after-manual", action="store_true")

    p.add_argument("--save-viz",        action="store_true")
    p.add_argument("--show-final-viz",  action="store_true")
    p.add_argument("--show-failed-viz", action="store_true")
    p.add_argument("--launch-cloudcompare", action="store_true")
    p.add_argument("--cloudcompare-cmd", default=None)

    p.add_argument("--sample-points",          type=int,   default=120_000)
    p.add_argument("--max-points",             type=int,   default=30_000)
    p.add_argument("--pose-refine",            action="store_true")
    p.add_argument("--icp-downsample-voxel",   type=float, default=0.0)
    p.add_argument("--icp-max-correspondence", type=float, default=0.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------

def list_scene_names(
    recon_root: Path,
    gt_root: Path,
    requested: Optional[List[str]],
) -> List[str]:
    recon_names = {p.name for p in recon_root.iterdir() if p.is_dir()}
    gt_names    = {p.name for p in gt_root.iterdir()   if p.is_dir()}
    names = sorted(recon_names & gt_names)
    if requested:
        names = [n for n in names if n in set(requested)]
    return names


def first_match(scene_dir: Path, candidates: Iterable[str]) -> Optional[Path]:
    """Find first existing candidate file; fallback to any .ply/.obj in the tree."""
    for rel in candidates:
        p = scene_dir / rel
        if p.exists():
            return p
    meshes = sorted(p for p in scene_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in {".ply", ".obj"})
    if not meshes:
        return None
    for kw in ("object", "textured", "points"):
        for p in meshes:
            if kw in p.name.lower():
                return p
    return meshes[0]


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run_subprocess(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Run a subprocess, stripping environment variables that pollute sub-envs."""
    env = os.environ.copy()
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "QT_PLUGIN_PATH",
                "QML2_IMPORT_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
        env.pop(key, None)
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        text=True, capture_output=True, env=env,
    )


# ---------------------------------------------------------------------------
# Open3D helpers
# ---------------------------------------------------------------------------

def load_geometry(path: Path, sample_points: int):
    """Load any 3D file as an Open3D point cloud."""
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) > 0:
        return pcd, "point_cloud"
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if len(mesh.vertices) == 0:
        raise RuntimeError(f"Could not load geometry from: {path}")
    if len(mesh.triangles) > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=max(sample_points, 5_000))
        return pcd, "mesh_sampled"
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = mesh.vertices
    if mesh.has_vertex_colors():
        pcd2.colors = mesh.vertex_colors
    return pcd2, "mesh_vertices"


def _aabb_scale_ratio(source, target, mode: str) -> float:
    """Compute scale ratio between source and target bounding boxes."""
    if mode == "off":
        return 1.0
    src_ext = np.asarray(source.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    tgt_ext = np.asarray(target.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    if mode == "aabb_diag":
        src_m, tgt_m = float(np.linalg.norm(src_ext)), float(np.linalg.norm(tgt_ext))
    else:
        src_m, tgt_m = float(np.max(src_ext)), float(np.max(tgt_ext))
    if src_m <= 1e-12 or tgt_m <= 1e-12:
        return 1.0
    return tgt_m / src_m


def save_overlay_screenshot(
    src_path: Path,
    tgt_path: Path,
    out_png: Path,
    sample_points: int,
) -> None:
    import open3d as o3d
    src_pcd, _ = load_geometry(src_path, sample_points)
    tgt_pcd, _ = load_geometry(tgt_path, sample_points)
    src = copy.deepcopy(src_pcd)
    tgt = copy.deepcopy(tgt_pcd)
    src.paint_uniform_color([1.0, 0.706, 0.0])
    tgt.paint_uniform_color([0.0, 0.651, 0.929])
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1600, height=900)
    vis.add_geometry(src)
    vis.add_geometry(tgt)
    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    vis.capture_screen_image(str(out_png), do_render=True)
    vis.destroy_window()


# ---------------------------------------------------------------------------
# Manual alignment (Open3D backend)
# ---------------------------------------------------------------------------

def run_open3d_manual_alignment(
    raw_recon: Path,
    raw_gt: Path,
    cropped_recon_out: Path,
    manual_recon_out: Path,
    manual_transform_out: Path,
    manual_meta_out: Path,
    sample_points: int,
    do_crop: bool,
    reuse_cropped: bool,
    allow_scaling: bool,
    preview: bool,
    pre_scale_mode: str,
) -> dict:
    import open3d as o3d

    source, src_mode = load_geometry(raw_recon, sample_points)
    target, tgt_mode = load_geometry(raw_gt,    sample_points)

    if do_crop:
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window("Crop recon", width=1600, height=900)
        vis.add_geometry(source)
        vis.run()
        cropped = vis.get_cropped_geometry()
        vis.destroy_window()
        if cropped and len(cropped.points) > 0:
            source = cropped
        cropped_recon_out.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(cropped_recon_out), source)

    elif reuse_cropped and cropped_recon_out.exists():
        source, src_mode = load_geometry(cropped_recon_out, sample_points)

    pre_scale = _aabb_scale_ratio(source, target, pre_scale_mode)
    if pre_scale_mode != "off":
        source_scaled = o3d.geometry.PointCloud(source)
        source_scaled.scale(pre_scale, center=source_scaled.get_center())
        source = source_scaled

    def _pick(pcd, title):
        v = o3d.visualization.VisualizerWithEditing()
        v.create_window(title, width=1600, height=900)
        v.add_geometry(pcd)
        v.run()
        pts = v.get_picked_points()
        v.destroy_window()
        return pts

    src_ids = _pick(source, "Pick SOURCE correspondences")
    tgt_ids = _pick(target, "Pick TARGET correspondences")

    if len(src_ids) < 3 or len(tgt_ids) < 3 or len(src_ids) != len(tgt_ids):
        raise RuntimeError("Need at least 3 matching correspondences")

    corr = np.zeros((len(src_ids), 2), dtype=np.int32)
    corr[:, 0] = src_ids
    corr[:, 1] = tgt_ids

    est   = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=allow_scaling)
    trans = est.compute_transformation(source, target, o3d.utility.Vector2iVector(corr))

    aligned = o3d.geometry.PointCloud(source)
    aligned.transform(trans)
    manual_recon_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(manual_recon_out), aligned)
    np.savetxt(manual_transform_out, trans, fmt="%.10f")

    meta = {
        "backend": "open3d",
        "pre_scale_mode": pre_scale_mode,
        "pre_scale_factor": float(pre_scale),
        "with_scaling": bool(allow_scaling),
        "num_correspondences": len(src_ids),
    }
    manual_meta_out.write_text(json.dumps(meta, indent=2))
    if preview:
        src_v = copy.deepcopy(source)
        src_v.transform(trans)
        src_v.paint_uniform_color([1, 0.7, 0])
        tgt_v = copy.deepcopy(target)
        tgt_v.paint_uniform_color([0, 0.65, 0.93])
        o3d.visualization.draw_geometries([src_v, tgt_v], window_name="Manual preview",
                                          width=1600, height=900)
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    cfg  = load_config(args.config)

    recon_root  = Path(args.recon_root).expanduser().resolve()
    gt_root     = Path(args.gt_root).expanduser().resolve()
    output_base = Path(args.output_base).expanduser().resolve()
    run_name    = args.run_name or f"bufferx_icp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir     = output_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    here             = Path(__file__).resolve().parent
    bufferx_wrapper  = Path(args.bufferx_root).expanduser().resolve() / "path_matters/haroun/Pipeline/cc_bufferx_pipeline_package/bufferx_pair_wrapper.py"
    icp_helper       = Path(args.bufferx_root).expanduser().resolve() / "path_matters/haroun/Pipeline/cc_bufferx_pipeline_package/icp_eval_helper.py"

    scenes = list_scene_names(recon_root, gt_root, args.scene_names)
    logger.info("Found %d scene(s): %s", len(scenes), scenes)

    summary_rows: List[Dict] = []

    for idx, scene_name in enumerate(scenes, 1):
        logger.info("=== [%d/%d] %s ===", idx, len(scenes), scene_name)
        scene_out = run_dir / scene_name
        raw_dir   = scene_out / "raw"
        manual_dir = scene_out / "manual"
        bx_dir    = scene_out / "bufferx"
        icp_dir   = scene_out / "icp"
        viz_dir   = scene_out / "viz"
        for d in (raw_dir, manual_dir, bx_dir, icp_dir, viz_dir):
            d.mkdir(parents=True, exist_ok=True)

        recon_src = first_match(recon_root / scene_name, args.recon_candidates)
        gt_src    = first_match(gt_root    / scene_name, args.gt_candidates)
        if recon_src is None or gt_src is None:
            logger.warning("Missing input for %s — skipping", scene_name)
            summary_rows.append({"scene": scene_name, "status": "missing_input"})
            continue

        raw_recon = raw_dir / f"recon_input{recon_src.suffix.lower()}"
        raw_gt    = raw_dir / f"gt_input{gt_src.suffix.lower()}"
        for src, dst in [(recon_src, raw_recon), (gt_src, raw_gt)]:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)

        manual_cropped = manual_dir / "recon_cropped.ply"
        manual_recon   = manual_dir / "recon_manual.ply"
        manual_init_tf = manual_dir / "manual_init_transform.txt"
        manual_meta    = manual_dir / "recon_manual.meta.json"
        manual_present = manual_recon.exists()

        if args.manual_mode in ("prepare", "require") and not manual_present:
            if args.manual_backend == "open3d":
                try:
                    run_open3d_manual_alignment(
                        raw_recon, raw_gt, manual_cropped, manual_recon,
                        manual_init_tf, manual_meta,
                        sample_points=args.open3d_sample_points,
                        do_crop=args.open3d_crop,
                        reuse_cropped=args.open3d_reuse_cropped_recon,
                        allow_scaling=args.open3d_with_scaling,
                        preview=args.open3d_preview_after_manual,
                        pre_scale_mode=args.open3d_pre_scale,
                    )
                    manual_present = manual_recon.exists()
                except Exception as exc:
                    logger.error("Open3D manual failed for %s: %s", scene_name, exc)
                    summary_rows.append({"scene": scene_name, "status": "manual_failed"})
                    continue
            else:
                logger.info("Manual CloudCompare alignment needed for %s", scene_name)
                summary_rows.append({"scene": scene_name, "status": "manual_needed"})
                continue

        if args.manual_mode == "off":
            work_recon = raw_recon
        elif args.manual_mode == "prefer":
            work_recon = manual_recon if manual_present else raw_recon
        else:
            work_recon = manual_recon
        work_gt = raw_gt

        # BUFFER-X
        init_tf      = bx_dir / "init_transform.txt"
        init_json    = bx_dir / "init_summary.json"
        init_aligned = bx_dir / "aligned_source_init.ply"
        bx_cmd = [
            "conda", "run", "-n", args.bufferx_env,
            "python3", str(bufferx_wrapper),
            "--bufferx-root", str(Path(args.bufferx_root).expanduser().resolve()),
            "--src", str(work_recon), "--tgt", str(work_gt),
            "--out-transform", str(init_tf),
            "--out-json", str(init_json),
            "--out-aligned-src", str(init_aligned),
            "--experiment-id", args.experiment_id,
            "--sample-points", str(args.sample_points),
            "--max-points", str(args.max_points),
        ]
        if args.pose_refine:
            bx_cmd.append("--pose-refine")

        bx_res = run_subprocess(bx_cmd)
        (bx_dir / "stdout.log").write_text(bx_res.stdout)
        (bx_dir / "stderr.log").write_text(bx_res.stderr)
        if bx_res.returncode != 0:
            logger.error("BUFFER-X failed for %s", scene_name)
            summary_rows.append({"scene": scene_name, "status": "bufferx_failed",
                                  "stderr": bx_res.stderr[-1000:]})
            continue

        # ICP / eval
        icp_tf      = icp_dir / "icp_transform.txt"
        icp_json    = icp_dir / "icp_summary.json"
        icp_aligned = icp_dir / "aligned_source_icp.ply"
        icp_cmd = [
            "conda", "run", "-n", args.bufferx_env,
            "python3", str(icp_helper),
            "--src", str(work_recon), "--tgt", str(work_gt),
            "--init-transform", str(init_tf),
            "--out-transform", str(icp_tf),
            "--out-json", str(icp_json),
            "--out-aligned-src", str(icp_aligned),
            "--sample-points", str(args.sample_points),
            "--downsample-voxel", str(args.icp_downsample_voxel),
            "--max-correspondence", str(args.icp_max_correspondence),
        ]
        icp_res = run_subprocess(icp_cmd)
        (icp_dir / "stdout.log").write_text(icp_res.stdout)
        (icp_dir / "stderr.log").write_text(icp_res.stderr)
        if icp_res.returncode != 0:
            logger.error("ICP failed for %s", scene_name)
            summary_rows.append({"scene": scene_name, "status": "icp_failed"})
            continue

        icp_metrics  = json.loads(icp_json.read_text())
        init_metrics = json.loads(init_json.read_text())
        row = {
            "scene": scene_name,
            "status": "ok",
            "icp_fitness": str(icp_metrics.get("icp_fitness", "")),
            "icp_inlier_rmse": str(icp_metrics.get("icp_inlier_rmse", "")),
            "eval_rmse": str(icp_metrics.get("rmse", "")),
            "bufferx_inliers": str(init_metrics.get("num_inliers", "")),
            "work_recon": str(work_recon),
            "work_gt": str(work_gt),
        }
        summary_rows.append(row)
        logger.info("OK: %s  fitness=%s  rmse=%s",
                    scene_name, row["icp_fitness"], row["eval_rmse"])

    summary_csv = run_dir / "batch_summary.csv"
    fieldnames  = sorted({k for r in summary_rows for k in r})
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    logger.info("Summary: %s", summary_csv)
    had_failures = any(r.get("status") != "ok" for r in summary_rows)
    return 1 if had_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
