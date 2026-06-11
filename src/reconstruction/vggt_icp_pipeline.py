"""
VGGT Reconstruction + ICP Alignment Pipeline.

Complete end-to-end pipeline that:
  1. Reconstructs a 3D scene from images using VGGT
  2. Preprocesses both the scene and a reference object point cloud
  3. Estimates initial scale via bounding-box diagonal ratio
  4. Runs RANSAC feature matching for coarse alignment
  5. Refines alignment with ICP
  6. Applies adaptive perturbation to escape local minima
  7. Searches for the optimal scale via binary search

Usage:
    conda run -n vggt python src/reconstruction/vggt_icp_pipeline.py \\
        --scene_dir  /path/to/scene \\
        --object_ply /path/to/object.ply \\
        --config     config/pipeline_config.yaml \\
        --no_visualize_final

Pipeline parameters are loaded from config/pipeline_config.yaml and can
be overridden via command-line flags.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
import yaml

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str | None) -> dict:
    """Load pipeline_config.yaml with sensible defaults."""
    defaults: dict = {
        "vggt": {
            "confidence_threshold": 2.0,
            "seed": 42,
            "max_images": 54,
        },
        "preprocessing": {
            "scene_downsample_voxel": 0.001,
            "scene_outlier_neighbors": 50,
            "scene_outlier_std_ratio": 5.0,
            "plane_distance_threshold": 0.015,
            "plane_offset": -0.015,
            "no_plane_removal": True,
            "object_downsample_voxel": 0.01,
        },
        "ransac": {
            "tries": 20,
            "downsample_voxel": 0.01,
            "max_correspondence_distance": 0.1,
            "normal_radius": 5.0,
            "normal_max_nn": 100,
            "fpfh_radius": 5.0,
            "fpfh_max_nn": 5,
        },
        "icp": {
            "max_correspondence_distance": 0.025,
            "max_iterations": 500,
        },
        "adaptive": {
            "max_iterations": 50,
            "fitness_threshold": 0.95,
            "rmse_threshold": 0.005,
            "rotation_noise_range": 0.01,
            "translation_noise_start": 0.1,
        },
        "scale_refinement": {
            "min_factor": 0.5,
            "max_factor": 1.5,
            "max_iterations": 6,
            "convergence_threshold": 0.05,
            "icp_max_iterations": 200,
        },
    }
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        for section, values in user_cfg.items():
            if section in defaults and isinstance(values, dict):
                defaults[section].update(values)
    return defaults


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VGGT Reconstruction + ICP Alignment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--scene_dir", required=True,
                   help="Scene directory (must contain an images/ sub-folder)")
    p.add_argument("--object_ply", required=True,
                   help="Reference object PLY for alignment")
    p.add_argument("--config", default="config/pipeline_config.yaml",
                   help="Path to pipeline_config.yaml")
    p.add_argument("--conf_thres_value", type=float, default=None)
    p.add_argument("--skip_reconstruction", action="store_true")
    p.add_argument("--no_plane_removal", action="store_true")
    p.add_argument("--visualize_final", action="store_true", default=False)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# VGGT reconstruction
# ---------------------------------------------------------------------------

def run_vggt_reconstruction(scene_dir: Path, cfg: dict) -> Path:
    """Reconstruct scene from images using VGGT-1B and save sparse/points.ply."""
    vggt_cfg = cfg["vggt"]
    seed = vggt_cfg["seed"]
    conf_thr = vggt_cfg["confidence_threshold"]
    max_images = vggt_cfg["max_images"]

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    major_cc = torch.cuda.get_device_capability()[0] if device == "cuda" else 0
    dtype = torch.bfloat16 if major_cc >= 8 else torch.float16 if device == "cuda" else torch.float32
    logger.info("VGGT: device=%s  dtype=%s", device, dtype)

    logger.info("Loading VGGT-1B...")
    model = VGGT()
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            map_location=device,
        )
    )
    model.eval().to(device)

    image_dir = scene_dir / "images"
    image_paths = sorted(glob.glob(str(image_dir / "*")))[:max_images]
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    logger.info("Loading %d images...", len(image_paths))

    images, original_coords = load_and_preprocess_images_square(image_paths, 1024)
    images = images.to(device)
    original_coords = original_coords.to(device)

    logger.info("VGGT inference...")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        images_batch = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)[None]
        aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)

    extrinsic  = extrinsic.squeeze(0).cpu().numpy()
    intrinsic  = intrinsic.squeeze(0).cpu().numpy()
    depth_map  = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()

    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    n_frames, height, width, _ = points_3d.shape

    points_rgb = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

    points_xyf = create_pixel_coordinate_grid(n_frames, height, width)
    conf_mask  = depth_conf >= conf_thr
    conf_mask  = randomly_limit_trues(conf_mask, 100_000)

    points_3d  = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]
    logger.info("Filtered to %d points (conf >= %.1f)", len(points_3d), conf_thr)

    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d, points_xyf, points_rgb, extrinsic, intrinsic,
        np.array([518, 518]), shared_camera=False, camera_type="PINHOLE",
    )
    base_names = [os.path.basename(p) for p in image_paths]
    for pyimageid in reconstruction.images:
        pyimage   = reconstruction.images[pyimageid]
        pycamera  = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = base_names[pyimageid - 1]
        real_size = original_coords.cpu().numpy()[pyimageid - 1, -2:]
        ratio     = max(real_size) / 518
        params    = copy.deepcopy(pycamera.params) * ratio
        params[-2:] = real_size / 2
        pycamera.params  = params
        pycamera.width   = int(real_size[0])
        pycamera.height  = int(real_size[1])

    sparse_dir = scene_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    reconstruction.write(str(sparse_dir))

    scene_ply = sparse_dir / "points.ply"
    trimesh.PointCloud(points_3d, colors=points_rgb).export(str(scene_ply))
    logger.info("Saved reconstruction: %s", scene_ply)
    return scene_ply


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_scene(pcd_path: Path, cfg: dict) -> o3d.geometry.PointCloud:
    """Remove outliers, optionally strip the table plane, then voxel-downsample."""
    pre = cfg["preprocessing"]
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    n0  = len(pcd.points)

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=pre["scene_outlier_neighbors"],
        std_ratio=pre["scene_outlier_std_ratio"],
    )

    if not pre["no_plane_removal"]:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
        )
        _, inliers = pcd.segment_plane(
            distance_threshold=pre["plane_distance_threshold"],
            ransac_n=3, num_iterations=1000,
        )
        pcd = pcd.select_by_index(inliers, invert=True)

    pcd = pcd.voxel_down_sample(pre["scene_downsample_voxel"])
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    logger.info("Scene: %d → %d points", n0, len(pcd.points))
    return pcd


def preprocess_object(pcd_path: Path, cfg: dict) -> o3d.geometry.PointCloud:
    """Downsample and compute normals for the reference object."""
    pre = cfg["preprocessing"]
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    n0  = len(pcd.points)
    pcd = pcd.voxel_down_sample(pre["object_downsample_voxel"])
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    logger.info("Object: %d → %d points", n0, len(pcd.points))
    return pcd


# ---------------------------------------------------------------------------
# Scale estimation
# ---------------------------------------------------------------------------

def estimate_scale(source: o3d.geometry.PointCloud,
                   target: o3d.geometry.PointCloud) -> tuple[float, np.ndarray]:
    """Estimate scale by bounding-box diagonal ratio and return centering transform."""
    src_diag = float(np.linalg.norm(source.get_axis_aligned_bounding_box().get_extent()))
    tgt_diag = float(np.linalg.norm(target.get_axis_aligned_bounding_box().get_extent()))
    scale = tgt_diag / src_diag

    src_scaled = copy.deepcopy(source)
    src_scaled.scale(scale, center=src_scaled.get_center())
    T = np.eye(4)
    T[:3, 3] = target.get_center() - src_scaled.get_center()
    logger.info("Scale estimate: %.6f  (src_diag=%.4f  tgt_diag=%.4f)", scale, src_diag, tgt_diag)
    return scale, T


# ---------------------------------------------------------------------------
# RANSAC + ICP
# ---------------------------------------------------------------------------

def ransac_alignment(source: o3d.geometry.PointCloud,
                     target: o3d.geometry.PointCloud,
                     cfg: dict) -> o3d.pipelines.registration.RegistrationResult:
    """Coarse RANSAC feature-matching alignment using FPFH descriptors."""
    rc = cfg["ransac"]
    tgt_down = target.voxel_down_sample(rc["downsample_voxel"])

    def compute_fpfh(pcd):
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=rc["normal_radius"], max_nn=rc["normal_max_nn"])
        )
        return o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=rc["fpfh_radius"], max_nn=rc["fpfh_max_nn"])
        )

    src_fpfh = compute_fpfh(copy.deepcopy(source))
    tgt_fpfh = compute_fpfh(copy.deepcopy(tgt_down))

    best, best_fitness = None, -1.0
    for i in range(rc["tries"]):
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, tgt_down, src_fpfh, tgt_fpfh,
            mutual_filter=True,
            max_correspondence_distance=rc["max_correspondence_distance"] * 0.1,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(rc["max_correspondence_distance"]),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100_000, 0.999),
        )
        logger.debug("RANSAC %d/%d: fitness=%.4f", i + 1, rc["tries"], result.fitness)
        if result.fitness > best_fitness:
            best_fitness, best = result.fitness, result

    logger.info("Best RANSAC: fitness=%.4f  RMSE=%.6f", best.fitness, best.inlier_rmse)
    return best


def icp_refine(source: o3d.geometry.PointCloud,
               target: o3d.geometry.PointCloud,
               init_transform: np.ndarray,
               cfg: dict) -> o3d.pipelines.registration.RegistrationResult:
    """Point-to-point ICP refinement."""
    ic = cfg["icp"]
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        ic["max_correspondence_distance"],
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=ic["max_iterations"]
        ),
    )
    logger.info("ICP: fitness=%.4f  RMSE=%.6f", result.fitness, result.inlier_rmse)
    return result


# ---------------------------------------------------------------------------
# Adaptive refinement
# ---------------------------------------------------------------------------

def adaptive_refinement(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial_result: o3d.pipelines.registration.RegistrationResult,
    cfg: dict,
) -> o3d.pipelines.registration.RegistrationResult:
    """Escape local minima via small random perturbations followed by ICP."""
    ac = cfg["adaptive"]
    ic = cfg["icp"]

    best_fitness = initial_result.fitness
    best_rmse    = initial_result.inlier_rmse
    best_T       = initial_result.transformation
    noise_t      = ac["translation_noise_start"]

    for i in range(ac["max_iterations"]):
        noise_R = o3d.geometry.get_rotation_matrix_from_xyz(
            np.random.uniform(-ac["rotation_noise_range"], ac["rotation_noise_range"], 3)
        )
        noise_tf = np.eye(4)
        noise_tf[:3, :3] = noise_R
        noise_tf[:3, 3]  = np.random.uniform(-noise_t, noise_t, 3)

        try:
            res = o3d.pipelines.registration.registration_icp(
                source, target,
                ic["max_correspondence_distance"] * 0.5,
                noise_tf @ best_T,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )
            if res.fitness > best_fitness or (res.fitness == best_fitness and res.inlier_rmse < best_rmse):
                best_fitness, best_rmse, best_T = res.fitness, res.inlier_rmse, res.transformation
                logger.debug("Adaptive iter %d: fitness=%.4f  RMSE=%.6f", i + 1, best_fitness, best_rmse)
                if best_fitness >= ac["fitness_threshold"] and best_rmse <= ac["rmse_threshold"]:
                    break
        except Exception:
            noise_t += 0.1

        if best_fitness < ac["fitness_threshold"]:
            noise_t += 0.05

    final = o3d.pipelines.registration.RegistrationResult()
    final.fitness      = best_fitness
    final.inlier_rmse  = best_rmse
    final.transformation = best_T
    logger.info("Adaptive: fitness=%.4f  RMSE=%.6f", best_fitness, best_rmse)
    return final


# ---------------------------------------------------------------------------
# Scale refinement
# ---------------------------------------------------------------------------

def refine_scale(
    object_pcd: o3d.geometry.PointCloud,
    scene_pcd:  o3d.geometry.PointCloud,
    initial_scale: float,
    initial_result: o3d.pipelines.registration.RegistrationResult,
    cfg: dict,
) -> tuple[float, o3d.pipelines.registration.RegistrationResult]:
    """Binary search over scale to minimise ICP inlier RMSE."""
    sc = cfg["scale_refinement"]
    ic = cfg["icp"]

    scale_min = initial_scale * sc["min_factor"]
    scale_max = initial_scale * sc["max_factor"]
    best_scale, best_rmse, best_result = initial_scale, initial_result.inlier_rmse, initial_result

    for _ in range(sc["max_iterations"]):
        scale_range = scale_max - scale_min
        if scale_range < initial_scale * sc["convergence_threshold"]:
            break
        scales = [scale_min, (scale_min + scale_max) / 2, scale_max]
        rmses  = []
        for s in scales:
            obj = copy.deepcopy(object_pcd)
            obj.scale(s, center=obj.get_center())
            T = np.eye(4)
            T[:3, 3] = scene_pcd.get_center() - obj.get_center()
            obj.transform(T)
            try:
                res = o3d.pipelines.registration.registration_icp(
                    obj, scene_pcd, ic["max_correspondence_distance"],
                    initial_result.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=sc["icp_max_iterations"]
                    ),
                )
                rmses.append(res.inlier_rmse)
                if res.inlier_rmse < best_rmse:
                    best_rmse, best_scale, best_result = res.inlier_rmse, s, res
            except Exception:
                rmses.append(float("inf"))

        if rmses[1] <= rmses[0] and rmses[1] <= rmses[2]:
            scale_min = (scale_min + scales[1]) / 2
            scale_max = (scale_max + scales[1]) / 2
        elif rmses[0] <= rmses[2]:
            scale_max = scales[1]
        else:
            scale_min = scales[1]

    logger.info("Scale refined: %.6f → %.6f  RMSE: %.6f → %.6f",
                initial_scale, best_scale, initial_result.inlier_rmse, best_rmse)
    return best_scale, best_result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> dict:
    """Execute the full reconstruction + alignment pipeline."""
    t0       = time.perf_counter()
    cfg      = load_config(args.config)
    scene_dir = Path(args.scene_dir)
    output_dir = scene_dir / "icp_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.conf_thres_value is not None:
        cfg["vggt"]["confidence_threshold"] = args.conf_thres_value
    if args.no_plane_removal:
        cfg["preprocessing"]["no_plane_removal"] = True

    # Stage 1: Reconstruct
    scene_ply = scene_dir / "sparse" / "points.ply"
    if args.skip_reconstruction and scene_ply.exists():
        logger.info("Using existing reconstruction: %s", scene_ply)
    else:
        scene_ply = run_vggt_reconstruction(scene_dir, cfg)

    # Stage 2: Preprocess
    scene_pcd  = preprocess_scene(scene_ply, cfg)
    object_pcd = preprocess_object(Path(args.object_ply), cfg)

    # Stage 3: Scale + centering
    scale, scale_T = estimate_scale(object_pcd, scene_pcd)
    obj_scaled = copy.deepcopy(object_pcd)
    obj_scaled.scale(scale, center=obj_scaled.get_center())
    obj_scaled.transform(scale_T)

    # Stage 4: RANSAC
    ransac_result = ransac_alignment(obj_scaled, scene_pcd, cfg)

    # Stage 5: ICP refinement
    icp_result = icp_refine(obj_scaled, scene_pcd, ransac_result.transformation, cfg)

    # Stage 6: Adaptive refinement
    adaptive_result = adaptive_refinement(obj_scaled, scene_pcd, icp_result, cfg)

    # Stage 7: Scale refinement
    refined_scale, refined_result = refine_scale(object_pcd, scene_pcd, scale, adaptive_result, cfg)
    if refined_result.inlier_rmse < adaptive_result.inlier_rmse:
        scale, final_result = refined_scale, refined_result
    else:
        final_result = adaptive_result

    # Finalize
    final_T = final_result.transformation @ scale_T
    obj_aligned = copy.deepcopy(object_pcd)
    obj_aligned.scale(scale, center=obj_aligned.get_center())
    obj_aligned.transform(final_T)

    if args.visualize_final:
        tgt_vis = copy.deepcopy(scene_pcd).paint_uniform_color([1, 0, 0])
        aln_vis = copy.deepcopy(obj_aligned).paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([tgt_vis, aln_vis],
                                          window_name="Final Alignment", width=1280, height=720)

    np.save(output_dir / "transformation.npy", final_T)
    np.save(output_dir / "scale.npy", np.array([scale]))
    o3d.io.write_point_cloud(str(output_dir / "object_aligned.ply"), obj_aligned)

    metrics = {
        "scale": float(scale),
        "final_fitness": float(final_result.fitness),
        "final_rmse":    float(final_result.inlier_rmse),
        "elapsed_sec":   time.perf_counter() - t0,
        "transformation": final_T.tolist(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Pipeline done in %.1fs  fitness=%.4f  RMSE=%.6f",
                metrics["elapsed_sec"], metrics["final_fitness"], metrics["final_rmse"])
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    with torch.no_grad():
        run_pipeline(args)


if __name__ == "__main__":
    main()
