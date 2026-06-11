"""
Isaac Sim image capture for robotic 3D reconstruction.

Moves a single camera through N positions around a target object and
captures one image per position.  Designed to run inside Isaac Sim's
script editor (standalone Python mode).

Usage:
    # Inside Isaac Sim Script Editor:
    exec(open("capture_images.py").read())

    # With a YAML config override:
    python capture_images.py --config /path/to/pipeline_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str | None = None) -> dict:
    """Load pipeline_config.yaml, fall back to built-in defaults."""
    defaults = {
        "capture": {
            "num_positions": 54,
            "circle_radius": 0.45,
            "camera_height_offset": 0.35,
            "focal_length": 24.0,
            "stabilize_frames": 20,
            "stabilize_sleep": 0.3,
        }
    }
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        defaults.update(user_cfg)
    return defaults


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isaac Sim image capture")
    p.add_argument("--config", default=None, help="Path to pipeline_config.yaml")
    p.add_argument("--target-path", default="/World/Object",
                   help="USD prim path of the target object")
    p.add_argument("--camera-path", default="/World/Camera",
                   help="USD prim path of the camera")
    p.add_argument("--output-dir", required=False, default=None,
                   help="Root capture output directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Isaac Sim runtime helpers
# ---------------------------------------------------------------------------

def get_object_center(target_prim_path: str) -> np.ndarray:
    """Return world-space center of the target prim."""
    from omni.isaac.core.utils.prims import get_prim_at_path
    from pxr import UsdGeom

    prim = get_prim_at_path(target_prim_path)
    if not prim:
        raise RuntimeError(f"Object not found at USD path: {target_prim_path}")

    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    t = transform.ExtractTranslation()
    return np.array([t[0], t[1], t[2]])


def setup_camera(stage, camera_path: str, focal_length: float):
    """Create or retrieve camera prim and set focal length."""
    from pxr import UsdGeom

    cam_prim = stage.GetPrimAtPath(camera_path)
    if not cam_prim:
        cam_prim = stage.DefinePrim(camera_path, "Camera")

    UsdGeom.Camera(cam_prim).GetFocalLengthAttr().Set(focal_length)
    logger.info("Camera ready at %s (focal_length=%.1f)", camera_path, focal_length)
    return cam_prim


def compute_positions(
    object_center: np.ndarray,
    num_positions: int,
    radius: float,
    height_offset: float,
) -> list[dict]:
    """Compute equally-spaced camera positions on a circle around the object."""
    positions = []
    cam_z = object_center[2] + height_offset
    for i in range(num_positions):
        angle = 2 * np.pi * i / num_positions
        positions.append({
            "index": i,
            "angle_deg": float(np.degrees(angle)),
            "position": np.array([
                object_center[0] + radius * np.cos(angle),
                object_center[1] + radius * np.sin(angle),
                cam_z,
            ]),
        })
    return positions


def look_at_rotation(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute XYZ Euler angles so the camera points from cam_pos toward target."""
    from scipy.spatial.transform import Rotation as R_scipy

    direction = target - cam_pos
    direction /= np.linalg.norm(direction)

    up = np.array([0.0, 0.0, 1.0])
    z_axis = -direction
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-3:
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
    return R_scipy.from_matrix(rot_matrix).as_euler("xyz", degrees=True)


def move_camera_and_capture(
    app,
    stage,
    camera_prim,
    object_center: np.ndarray,
    positions: list[dict],
    output_path: Path,
    stabilize_frames: int,
    stabilize_sleep: float,
) -> int:
    """Iterate positions, move camera, and capture each frame."""
    from pxr import Gf, UsdGeom
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    try:
        vp_api = get_active_viewport()
        vp_api.set_active_camera(camera_prim.GetPath().pathString)
    except Exception as e:
        logger.warning("Viewport switch failed: %s", e)

    captured = 0
    for pos_data in positions:
        i = pos_data["index"]
        cam_pos = pos_data["position"]

        euler = look_at_rotation(cam_pos, object_center)

        xform = UsdGeom.Xformable(camera_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*cam_pos.tolist()))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*euler.tolist()))

        for _ in range(stabilize_frames):
            app.update()
        time.sleep(stabilize_sleep)

        filepath = output_path / f"view_{i:03d}.png"
        try:
            capture_viewport_to_file(get_active_viewport(), str(filepath))
            time.sleep(0.2)
            if filepath.exists():
                logger.info("Captured view_%03d.png (%.1f KB)", i, filepath.stat().st_size / 1024)
                captured += 1
            else:
                logger.warning("Frame %d: file not created", i)
        except Exception as e:
            logger.error("Frame %d capture error: %s", i, e)

    return captured


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    cap_cfg = cfg.get("capture", {})

    import omni.kit.app
    from omni.isaac.core import World

    app = omni.kit.app.get_app()
    world = World.instance() or World()
    if not world.is_playing():
        world.reset()
        world.play()
    for _ in range(30):
        app.update()

    stage = omni.usd.get_context().get_stage()

    object_center = get_object_center(args.target_path)
    logger.info("Object center: %s", object_center)

    camera_prim = setup_camera(stage, args.camera_path, cap_cfg.get("focal_length", 24.0))

    positions = compute_positions(
        object_center,
        num_positions=cap_cfg.get("num_positions", 54),
        radius=cap_cfg.get("circle_radius", 0.45),
        height_offset=cap_cfg.get("camera_height_offset", 0.35),
    )
    logger.info("Planned %d capture positions", len(positions))

    out_root = Path(args.output_dir) if args.output_dir else Path.home() / "captures"
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = out_root / f"session_{session_ts}"
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_path)

    captured = move_camera_and_capture(
        app=app,
        stage=stage,
        camera_prim=camera_prim,
        object_center=object_center,
        positions=positions,
        output_path=output_path,
        stabilize_frames=cap_cfg.get("stabilize_frames", 20),
        stabilize_sleep=cap_cfg.get("stabilize_sleep", 0.3),
    )

    logger.info("Capture complete: %d/%d images saved to %s", captured, len(positions), output_path)


if __name__ == "__main__":
    main()
