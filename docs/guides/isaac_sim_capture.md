# Isaac Sim Capture — Setup Guide

> [Docs home](../README.md) · [Pipeline Guide](../pipeline.md) · **Isaac Sim Capture Guide**

This guide explains how to set up the Isaac Sim 5.1 capture environment
and run [`src/capture/capture_images.py`](../../src/capture/capture_images.py)
for the Path Matters dataset.

---

## Prerequisites

- NVIDIA Isaac Sim 5.1 installed at the default location
- Isaac Lab 0.48 extensions installed
- UR5e robot USD asset (`ur5e_wr_description` package built)
- Python 3.10 (bundled with Isaac Sim)

---

## Environment Setup

The UR5e workspace uses a ROS2 description package.  Build it once:

```bash
cd ~/path_matters/Isaacsim/envs/ur5e_wr_description
colcon build --symlink-install
source install/setup.bash
```

---

## Scene Creation

### 1. Create the capture USD scene

Run `src/capture/` → open Isaac Sim Script Editor and execute:

```python
# Creates a USD stage with:
#   - Ground plane
#   - Point light
#   - UR5e robot at origin
#   - Camera prim at /World/Camera
exec(open("path_matters/Isaacsim/scripts/env_setup/create_capture_env_usd.py").read())
```

### 2. Load your target object

In the Isaac Sim GUI:
1. File → Import → select `data/raw/<object_name>/object.usd` (or .obj/.ply)
2. Place object on table surface at approximately (0, 0, 0.05)
3. Note the exact USD prim path (e.g. `/World/alarm_clock_2`)

### 3. Update config

Edit `config/pipeline_config.yaml`:
```yaml
capture:
  num_positions: 54
  circle_radius: 0.45        # Adjust for object size
  camera_height_offset: 0.35 # Adjust so object stays in frame
```

---

## Running the Capture

In the Isaac Sim Script Editor:

```python
import sys
sys.argv = [
    "capture_images.py",
    "--target-path", "/World/alarm_clock_2",
    "--camera-path", "/World/Camera",
    "--output-dir",  "/home/AP_PathMatters/path_matters/datasets/raw/alarm_clock_2",
    "--config",      "/home/AP_PathMatters/path-matters-robotic-3d-reconstruction/config/pipeline_config.yaml",
]
exec(open("src/capture/capture_images.py").read())
```

---

## Quality Checks

After capture, verify:

```bash
# Check image count
ls data/raw/<object_name>/images/ | wc -l   # expect 54

# Check no white/blank images
python - <<'EOF'
from pathlib import Path
import numpy as np
from PIL import Image
imgs = sorted(Path("data/raw/<object_name>/images").glob("*.png"))
for img_path in imgs:
    arr = np.array(Image.open(img_path))
    mean = arr.mean()
    if mean > 250 or mean < 5:
        print(f"SUSPECT: {img_path.name}  mean={mean:.1f}")
print("Check complete")
EOF
```

---

## Common Issues

### All images are white
The RTX renderer needs more frames to load.  Increase
`stabilize_frames` in `config/pipeline_config.yaml` to 40–60.

### Camera misses the object
- Reduce `circle_radius` if the object is small.
- Increase `camera_height_offset` if the table occludes the object from below.

### Script crashes with "Object not found"
The USD prim path is case-sensitive.  Use Isaac Sim's Stage panel to copy
the exact path.

### Rotation is 90° off
The look-at rotation uses Isaac Sim's Y-up convention.  If your scene uses
Z-up, the `up` vector in `look_at_rotation()` may need to be `[0, 1, 0]`.

---

## Coordinate Conventions

Isaac Sim uses:
- **Y-up** by default
- **Metres** as the physical unit
- Camera -Z axis pointing into the scene

The UR5e URDF uses standard ROS convention (Z-up, REP-103).  The
`ur5e_wr_description` package includes the coordinate conversion.
