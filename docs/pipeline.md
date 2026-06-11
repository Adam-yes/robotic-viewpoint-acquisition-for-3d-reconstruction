# Pipeline Reproduction Guide

> [Docs home](README.md) · [Methodology](methodology.md) · **Pipeline Guide** · [Data Management](data_management.md)

This document describes how to reproduce the results from the Path Matters paper
step by step.

**Contents:**
[Prerequisites](#prerequisites) ·
[1 Environment Setup](#step-1-environment-setup) ·
[2 Ground-Truth Meshes](#step-2-prepare-ground-truth-meshes) ·
[3 Image Capture](#step-3-capture-images-in-isaac-sim) ·
[4 VGGT Reconstruction](#step-4-vggt-reconstruction) ·
[5 VGGT + ICP](#step-5-vggt--icp-full-pipeline) ·
[6 BUFFER-X Evaluation](#step-6-batch-evaluation-with-buffer-x) ·
[7 Automated Pipeline](#step-7-full-automated-pipeline) ·
[8 RL Training](#step-8-rl-training) ·
[Table Results](#reproducing-table-results) ·
[Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware

- GPU: NVIDIA RTX A6000 (47 GB) or equivalent ≥ 24 GB VRAM
- CPU: 8+ cores recommended
- RAM: 32 GB+
- OS: Ubuntu 22.04 LTS

### Software

- Isaac Sim 5.1 (for image capture)
- Isaac Lab 0.48 (for RL training)
- Conda ≥ 23.0
- Git

---

## Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/Adam-yes/path-matters-robotic-3d-reconstruction.git
cd path-matters-robotic-3d-reconstruction

# Create VGGT environment
conda env create -f environment_vggt.yml
conda activate vggt

# Create BUFFER-X / Open3D environment
conda env create -f environment_bufferx.yml
conda activate bufferx_o3d
```

Set the CUDA memory allocator (required for VGGT on large scenes):

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Step 2: Prepare Ground-Truth Meshes

Place per-object ground-truth meshes under `data/raw/`:

```
data/raw/
  <object_name>/
    object.ply    ← or .obj
```

Center each mesh at the origin (required for ICP alignment):

```bash
python src/preprocessing/bring_to_center.py data/raw/<object_name>/object.ply
```

If the mesh is in a non-PLY format, convert it first:

```bash
python src/preprocessing/convert_mesh.py data/raw/<object_name>/object.stl .ply 50000
```

---

## Step 3: Capture Images in Isaac Sim

Launch Isaac Sim 5.1 and open a scene containing the UR5e robot and the
target object (full environment setup: [Isaac Sim Capture Guide](guides/isaac_sim_capture.md)).
In the Script Editor, run:

```python
# Update paths in config/pipeline_config.yaml first
exec(open("src/capture/capture_images.py").read())
```

Images are saved to `data/raw/<object_name>/images/`.

Expected output: 54 PNG files named `view_000.png` … `view_053.png`.

**Troubleshooting — white images:**
- Increase `stabilize_frames` in `config/pipeline_config.yaml` from 20 to 40.
- Ensure the RTX renderer has finished loading textures (`app.update()` loop).

**Troubleshooting — rotation offset (camera points wrong direction):**
- Verify the `TARGET_OBJECT_PATH` in the config matches the exact USD prim path.
- Run `get_object_center()` interactively and compare with the visual scene.

---

## Step 4: VGGT Reconstruction

```bash
conda activate vggt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/reconstruction/vggt_reconstruct.py \
    --image_dir data/raw/<object_name>/images \
    --out_ply   data/processed/<object_name>/points.ply \
    --max_images 54 \
    --conf_percentile 70.0
```

The first run downloads `model.pt` (~4 GB) from Hugging Face and caches it
in `~/.cache/torch/hub/`.

**Troubleshooting — CUDA OOM:**
```bash
# Reduce max_images or add:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# If still OOM, reduce IMAGE_LOAD_RESOLUTION in vggt_reconstruct.py from 1024 to 512
```

---

## Step 5: VGGT + ICP Full Pipeline

For each object, run the combined reconstruction + alignment pipeline:

```bash
conda activate vggt

python src/reconstruction/vggt_icp_pipeline.py \
    --scene_dir  data/raw/<object_name> \
    --object_ply data/raw/<object_name>/object.ply \
    --config     config/pipeline_config.yaml \
    --no_plane_removal
```

Results are written to `data/raw/<object_name>/icp_results/`:
- `object_aligned.ply`   — aligned object cloud
- `transformation.npy`   — final 4×4 transformation matrix
- `scale.npy`            — estimated scale factor
- `metrics.json`         — fitness, RMSE, correspondences, elapsed time

---

## Step 6: Batch Evaluation with BUFFER-X

```bash
conda activate bufferx_o3d

python src/registration/bufferx_pipeline.py \
    --recon-root  data/processed \
    --gt-root     data/raw \
    --output-base data/runs \
    --bufferx-root ~/BUFFER-X \
    --bufferx-env  bufferx_o3d \
    --manual-mode  off \
    --save-viz
```

A `batch_summary.csv` is written under the run directory with per-scene
fitness, RMSE, and BUFFER-X inlier counts.

---

## Step 7: Full Automated Pipeline

The shell script in `scripts/run_full_pipeline.sh` chains Steps 4–6 for a
single scene:

```bash
bash scripts/run_full_pipeline.sh \
    --scene <object_name> \
    --data-root data/raw \
    --output data/processed
```

---

## Step 8: RL Training

```bash
conda activate vggt   # or your Isaac Lab environment

python src/rl/viewpoint_env.py  # runs stub mode for unit testing

# For full Isaac Lab training:
# python -m isaaclab.train --task ViewpointSelection-v0 \
#     --num_envs 1 --max_iterations 1000
```

---

## Reproducing Table Results

### Table 7 — Reconstruction quality (28 objects)

Run Steps 4–6 for all 28 objects in the dataset.  Aggregate `batch_summary.csv`
using:

```python
import pandas as pd
df = pd.read_csv("data/runs/latest/batch_summary.csv")
print(df[["scene", "icp_fitness", "icp_inlier_rmse"]].describe())
```

Expected aggregates (mean over 28 objects):

| Metric        | VGGT | Fast3R | SAM3D |
|---------------|------|--------|-------|
| Fitness       | 0.93 | 0.89   | 0.91  |
| RMSE (m)      | 0.002| 0.010  | 0.008 |

### Table 11 — Camera orientation comparison (5 patterns × 2 orientations)

Run Step 4–5 with images captured under each pattern and orientation.
Compare `metrics.json` files across runs.

---

## Troubleshooting

### VGGT produces very few points
- Lower `conf_percentile` from 70 to 50 in `config/pipeline_config.yaml`.
- Verify images are not overexposed (white images issue).

### ICP fitness < 0.5
- Enable `--visualize_final` to inspect the initial alignment visually.
- Try a different `scale_method` or run with `--open3d-with-scaling`.
- Check that `bring_to_center.py` was run on the GT mesh.

### BUFFER-X fails with CUDA error
- Ensure `bufferx_o3d` conda environment is used, not the VGGT env.
- Check BUFFER-X dependencies: `conda activate bufferx_o3d && python -c "import open3d"`.

### Isaac Sim camera does not move
- The script must run from inside the Isaac Sim Script Editor, not as a
  standalone Python process.
- Ensure the World is in play mode before capture starts.

---

**See also:** [Methodology](methodology.md) for the scientific background,
[Data Management](data_management.md) for dataset layout and storage.
