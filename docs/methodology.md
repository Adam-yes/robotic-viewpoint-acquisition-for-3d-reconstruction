# Methodology

> [Docs home](README.md) · **Methodology** · [Pipeline Guide](pipeline.md) · [Data Management](data_management.md)

**Contents:**
[Overview](#overview) ·
[2D→3D Reconstruction](#2d--3d-reconstruction-with-vggt) ·
[Point Cloud Alignment](#point-cloud-alignment) ·
[Camera Orientation](#camera-orientation-the-dominant-factor) ·
[Scanning Trajectories](#scanning-trajectories) ·
[Reinforcement Learning](#reinforcement-learning-for-viewpoint-selection)

---

## Overview

Robotic viewpoint acquisition for 3d-reconstruction investigates a deceptively simple question: *given a fixed number
of camera viewpoints, does the trajectory through which those views are collected
affect the quality of a learned 2D→3D reconstruction?*

Our answer — yes, dramatically — motivates an experimental programme that
evaluates five scanning patterns, two camera orientation strategies, three
reconstruction models, and a reinforcement-learning agent that learns to
select viewpoints autonomously.

---

## 2D → 3D Reconstruction with VGGT

### Model selection

We evaluate three state-of-the-art multi-view reconstruction models:

| Model  | Fitness | RMSE (m) | Density (pts) | Time (s) |
|--------|---------|----------|---------------|---------|
| **VGGT** | **0.93** | **0.002** | 8 200 | 7 |
| SAM3D  | 0.91    | 0.008    | 45 000        | 9 |
| Fast3R | 0.89    | 0.010    | 5 600         | 7 |

VGGT (Visual Geometry Grounded Transformer) achieves the highest alignment
fitness and lowest inlier RMSE.  Its per-pixel confidence map is a key asset:
by retaining only points above a calibrated threshold (best value: 4.0) we
eliminate floating artefacts without sacrificing coverage.

### Inference pipeline

1. **Image loading** — up to 54 images are loaded at 1 024 × 1 024 and
   bilinearly down-sampled to the model's fixed 518 × 518 input resolution.
2. **Feature aggregation** — the VGGT aggregator produces multi-scale tokens
   from all input views simultaneously via cross-view attention.
3. **Camera head** — predicts extrinsic (Rt) and intrinsic (K) parameters
   for every view from the aggregated tokens.
4. **Depth head** — outputs a dense depth map and per-pixel confidence score
   for each view.
5. **Unprojection** — depth maps are lifted to world-space XYZ using
   `unproject_depth_map_to_point_map`, then filtered by confidence percentile.
6. **Export** — a coloured PLY point cloud is written for downstream alignment.

---

## Point Cloud Alignment

### Why alignment is non-trivial

VGGT produces metric-*free* reconstructions: the absolute scale of the output
cloud is undefined.  Before any quality metric can be computed against a
ground-truth mesh, three problems must be solved:

1. **Scale** — the reconstruction may be 10× smaller or larger than the GT.
2. **Pose** — the cloud may be arbitrarily rotated and translated.
3. **Density mismatch** — GT meshes contain ≫1 M triangles; reconstructed clouds
   contain 5 000–50 000 points.

### Scale estimation via 3-estimator consensus

We estimate an initial scale factor from three independent geometrical cues and
take their median:

- **Bounding-box diagonal** (`aabb_diag`): ratio of the L2 norms of the AABB
  extents of the reconstructed cloud and the GT mesh.
- **PCA axis lengths**: ratio of the first principal component length.
- **Convex hull volume**: cube-root ratio of convex hull volumes.

Three estimators guard against failure modes that any single estimator
encounters when the object has thin protrusions (hull volume distorted) or
flat geometry (PCA dominated by noise).

### RANSAC global registration

With the object scaled and translated to the scene centroid, FPFH (Fast Point
Feature Histograms) descriptors are computed and matched across clouds.  We run
20 RANSAC attempts and retain the hypothesis with the highest inlier ratio.

### Iterative Closest Point (ICP)

The RANSAC hypothesis is refined by point-to-point ICP:

- **max_correspondence_distance**: 0.025 m (empirically tuned)
- **max_iterations**: 500
- **Convergence**: relative RMSE change < 1 × 10⁻⁶

### Adaptive perturbation

ICP is a local optimiser and can stagnate in saddle points.  After convergence
we apply up to 50 random perturbations (rotation noise ≤ 0.01 rad,
translation noise starting at 0.1 m) followed by another ICP pass.
If any perturbation yields a higher fitness or lower RMSE, it replaces the
current best estimate.

### BUFFER-X registration baseline

As a deep-learning baseline we evaluate BUFFER-X, a descriptor-based
point-cloud registration network.  Key comparison:

| Method   | Fitness | Recall  |
|----------|---------|---------|
| ICP      | **0.87** | **78.6%** |
| BUFFER-X | 0.81    | 67.9%  |

ICP achieves better accuracy; BUFFER-X is faster and more robust to poor
initial alignment, making it a useful initialiser for ICP.

---

## Camera Orientation: a first-order acquisition factor

The single most surprising finding of the project is that **camera orientation
is the dominant controllable factor** in reconstruction quality — more
influential than trajectory geometry, number of viewpoints, or reconstruction
model choice.

| Approach | Strategy | Mean Fitness | Mean RMSE (m) |
|----------|----------|-------------|---------------|
| Approach 1 | Camera axis fixed downward | 0.68 | 0.035 |
| **Approach 2** | **Camera axis pointing at object** | **0.79** | **0.022** |

Switching from a fixed-down orientation (Approach 1) to an object-pointing
orientation (Approach 2) improves fitness by +16% and halves the RMSE, **with
no change to trajectory geometry or the number of captures**.

The mechanism is well-understood: object-pointing views maximise the
overlapping scene region across views, which is the prerequisite for
multi-view attention to resolve accurate correspondences.

---

## Scanning Trajectories

Five trajectory patterns were tested, each with both camera orientations:

| Pattern    | Viewpoints | Strategy |
|------------|-----------|----------|
| Lawnmower  | 12        | Horizontal raster |
| Zigzag     | 12        | Alternating elevation |
| **Hemisphere** | **12** | **1 top + 11 ring at 50° elev.** |
| Spiral     | 12        | Helix from top to equator |
| Random     | 12        | Uniform random on sphere |

**Best fixed strategy**: Hemisphere + Approach 2 → Fitness = 0.86, RMSE = 0.015 m.

---

## Reinforcement Learning for Viewpoint Selection

To move beyond fixed patterns, we train a UR5e robot with an eye-in-hand
camera using PPO in Isaac Lab 0.48.

### Reward design

The coverage reward alone is too sparse for the agent to bootstrap learning:

```
r_total = r_proximity × λ_prox + r_coverage × λ_cov + r_success
```

Where:
- `r_proximity = exp(−d / σ)` for d = distance to nearest uncovered voxel
- `r_coverage  = Δcoverage / total_voxels × scale`
- `r_success   = 10.0` when coverage ≥ 75%

**Critical finding**: without proximity shaping (exp_06), the agent achieves
only 0.4% success and never discovers the coverage reward.  With proximity
shaping (exp_07), success rate jumps to 45.2% and mean reward from 22.5 to 32.7.

### Coverage metric

The workspace is discretised into a 16 × 16 × 16 voxel grid.  A voxel is
marked covered when a 3D point from the current depth observation falls within
it.  Coverage ratio is the fraction of occupied voxels.

The RL agent achieves **75%+ workspace coverage** compared to 20.6% for a
random-action baseline — a 3.6× improvement.

---

**Next:** reproduce these results step by step with the
[Pipeline Reproduction Guide](pipeline.md).
