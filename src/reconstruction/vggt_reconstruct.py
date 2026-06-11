"""
VGGT point cloud export.

Loads images from a scene directory, runs the VGGT-1B model, unprojects
depth maps to world-space, filters by confidence, and writes a colored
PLY point cloud.

Usage:
    conda run -n vggt python src/reconstruction/vggt_reconstruct.py \
        --image_dir /path/to/scene/images \
        --out_ply   /path/to/scene/points.ply \
        --max_images 54 \
        --conf_percentile 70.0
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import yaml

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

logger = logging.getLogger(__name__)

VGGT_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
VGGT_FIXED_RESOLUTION = 518
IMAGE_LOAD_RESOLUTION = 1024


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str | None) -> dict:
    """Load pipeline_config.yaml and return reconstruction section."""
    defaults = {
        "reconstruction": {
            "max_images": 54,
            "stride": 1,
            "conf_percentile": 70.0,
            "max_points": 800_000,
        }
    }
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        defaults.update(user_cfg)
    return defaults


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export VGGT reconstruction to PLY")
    p.add_argument("--image_dir", required=True, help="Directory of input images")
    p.add_argument("--out_ply", required=True, help="Output PLY path")
    p.add_argument("--config", default=None, help="Path to pipeline_config.yaml")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--conf_percentile", type=float, default=None,
                   help="Keep points >= this percentile confidence (0-100)")
    p.add_argument("--max_points", type=int, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def choose_dtype_and_device() -> tuple[str, torch.dtype]:
    """Select CUDA/CPU and appropriate AMP dtype."""
    if torch.cuda.is_available():
        major_cc = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
        return "cuda", dtype
    return "cpu", torch.float32


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_paths(image_dir: str, stride: int, max_images: int) -> list[str]:
    """Glob and filter image paths from a directory."""
    paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    paths = [p for p in paths if Path(p).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    if stride > 1:
        paths = paths[::stride]
    if max_images > 0:
        paths = paths[:max_images]
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def reconstruct(
    image_dir: str,
    out_ply: str,
    max_images: int = 54,
    stride: int = 1,
    conf_percentile: float = 70.0,
    max_points: int = 800_000,
) -> Path:
    """
    Run VGGT on images and export a colored PLY point cloud.

    Args:
        image_dir:      Directory containing input images.
        out_ply:        Destination PLY file path.
        max_images:     Maximum number of images to use.
        stride:         Take every Nth image (default: every image).
        conf_percentile: Confidence percentile threshold for filtering.
        max_points:     Downsample to this many points if exceeded.

    Returns:
        Path to the written PLY file.
    """
    image_paths = load_image_paths(image_dir, stride, max_images)
    if len(image_paths) < 2:
        raise RuntimeError(f"Need at least 2 images in {image_dir}, found {len(image_paths)}")

    device, dtype = choose_dtype_and_device()
    logger.info("device=%s  dtype=%s  images=%d", device, dtype, len(image_paths))

    logger.info("Loading VGGT-1B model...")
    model = VGGT()
    state_dict = torch.hub.load_state_dict_from_url(VGGT_MODEL_URL, map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    logger.info("Preprocessing images...")
    images, _original_coords = load_and_preprocess_images_square(image_paths, IMAGE_LOAD_RESOLUTION)
    images = images.to(device)

    images_vggt = F.interpolate(
        images,
        size=(VGGT_FIXED_RESOLUTION, VGGT_FIXED_RESOLUTION),
        mode="bilinear",
        align_corners=False,
    )
    images_batch = images_vggt[None]

    amp_ctx = torch.cuda.amp.autocast(dtype=dtype) if device == "cuda" else nullcontext()

    logger.info("Running VGGT inference...")
    with torch.no_grad(), amp_ctx:
        aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, images_batch.shape[-2:]
        )
        depth_map, depth_conf = model.depth_head(
            aggregated_tokens_list, images_batch, ps_idx
        )

    extrinsic  = extrinsic.squeeze(0).detach().cpu().numpy()
    intrinsic  = intrinsic.squeeze(0).detach().cpu().numpy()
    depth_map  = depth_map.squeeze(0).detach().cpu().numpy()
    depth_conf = depth_conf.squeeze(0).detach().cpu().numpy()

    logger.info(
        "extrinsic=%s  intrinsic=%s  depth_map=%s  depth_conf=%s",
        extrinsic.shape, intrinsic.shape, depth_map.shape, depth_conf.shape,
    )

    world = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    rgb = (images_vggt.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    rgb = rgb.transpose(0, 2, 3, 1)

    pts  = world.reshape(-1, 3)
    col  = rgb.reshape(-1, 3)
    conf = depth_conf
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    conf = conf.reshape(-1)

    finite_mask = np.isfinite(pts).all(axis=1)
    if not np.any(finite_mask):
        raise RuntimeError("No finite 3D points after unprojection")

    conf_thr = np.percentile(conf[finite_mask], conf_percentile)
    keep = finite_mask & (conf >= conf_thr)
    pts, col = pts[keep], col[keep]

    if len(pts) == 0:
        raise RuntimeError("No points remaining after confidence filtering")

    if max_points > 0 and len(pts) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts, col = pts[idx], col[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64) / 255.0)

    out_path = Path(out_ply)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(out_path), pcd):
        raise RuntimeError(f"Failed to write point cloud to {out_path}")

    logger.info("Wrote %s  (points=%d)", out_path, len(pts))
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    cfg = load_config(args.config).get("reconstruction", {})

    reconstruct(
        image_dir=args.image_dir,
        out_ply=args.out_ply,
        max_images=args.max_images   or cfg.get("max_images",    54),
        stride=args.stride           or cfg.get("stride",         1),
        conf_percentile=args.conf_percentile if args.conf_percentile is not None
                        else cfg.get("conf_percentile", 70.0),
        max_points=args.max_points   or cfg.get("max_points", 800_000),
    )


if __name__ == "__main__":
    main()
