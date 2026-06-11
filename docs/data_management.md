# Data Management

> [Docs home](README.md) · [Methodology](methodology.md) · [Pipeline Guide](pipeline.md) · **Data Management**

## Directory Layout

```
data/
├── raw/                  ← Original captures and GT meshes (gitignored)
│   └── <object_name>/
│       ├── images/       ← PNG frames from Isaac Sim
│       ├── object.ply    ← Ground-truth mesh (centered)
│       └── sparse/       ← VGGT COLMAP output
├── processed/            ← Pipeline outputs (gitignored)
│   └── <object_name>/
│       ├── points.ply           ← VGGT point cloud
│       ├── icp_results/
│       │   ├── object_aligned.ply
│       │   ├── transformation.npy
│       │   ├── scale.npy
│       │   └── metrics.json
│       └── bufferx/
└── metadata/
    └── dataset_info.md   ← Object list, sources, licenses
```

## Datasets

### Synthetic dataset — Objaverse

28 household objects rendered in Isaac Sim 5.1 under controlled lighting.
Objects selected to span a range of geometries: convex (mug, sphere),
non-convex (alarm clock, wrench), thin (ruler), and symmetric (bottle).

Images are captured at 720 × 720 px with the RTX Path Tracer renderer.
Each object has 54 views (see `config/pipeline_config.yaml` for capture params).

### Ground-truth meshes

GT meshes are sourced from Objaverse, BOP benchmark datasets (YCB-V, T-LESS),
and manually scanned objects.  All meshes are:
- Converted to `.ply` format via `src/preprocessing/convert_mesh.py`
- Centered at the world origin via `src/preprocessing/bring_to_center.py`
- Sampled to 50 000 points where polygon count exceeds 500 K faces

### BOP benchmark splits

The `data/raw/datasets/bop/` directory mirrors the standard BOP layout:

```
bop/
├── lm/        ← LineMOD
├── ycbv/      ← YCB-Video
├── tless/     ← T-LESS
└── tyol/      ← TYOL
```

## Storage Estimates

| Item                         | Size per object | 28 objects |
|------------------------------|----------------|-----------|
| Raw images (54 × 720² PNG)   | ~25 MB         | ~700 MB   |
| VGGT PLY (800K pts)          | ~15 MB         | ~420 MB   |
| GT mesh (50K pts .ply)       | ~2 MB          | ~56 MB    |
| ICP results                  | ~10 MB         | ~280 MB   |

Total: ~1.5 GB per full run.

## Data Versioning

Raw data is **not committed to Git**.  The `.gitignore` excludes:

```
data/raw/
data/processed/
*.ply
*.pcd
*.npy
```

Use DVC or a shared network drive to version large assets.  A lightweight
`data/metadata/dataset_info.md` file tracks object names, sources, and
license information.

## Licenses

| Source      | License      | Objects used |
|-------------|-------------|-------------|
| Objaverse   | CC-BY / CC0 | 22          |
| BOP / YCB-V | CC-BY-NC    | 4           |
| Manual scan | Proprietary | 2           |

For commercial use, replace BOP objects with CC0-licensed alternatives.
