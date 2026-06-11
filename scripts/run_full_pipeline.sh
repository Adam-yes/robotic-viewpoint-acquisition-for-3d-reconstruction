#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh
# Full Path Matters pipeline: VGGT reconstruction → ICP alignment → evaluation
#
# Usage:
#   bash scripts/run_full_pipeline.sh \
#       --scene alarm_clock \
#       --data-root ~/path_matters/datasets/raw \
#       --output    ~/path_matters/experiments/run_01 \
#       [--skip-reconstruction]
#       [--no-plane-removal]
#       [--config   config/pipeline_config.yaml]
#
# Required conda environments:
#   vggt        (VGGT reconstruction)
#   bufferx_o3d (BUFFER-X + Open3D registration)
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
SCENE=""
DATA_ROOT="data/raw"
OUTPUT="data/processed"
CONFIG="config/pipeline_config.yaml"
SKIP_RECONSTRUCTION=false
NO_PLANE_REMOVAL=false
BUFFERX_ROOT="${HOME}/BUFFER-X"
VGGT_ENV="vggt"
BUFFERX_ENV="bufferx_o3d"

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)             SCENE="$2";              shift 2 ;;
        --data-root)         DATA_ROOT="$2";          shift 2 ;;
        --output)            OUTPUT="$2";             shift 2 ;;
        --config)            CONFIG="$2";             shift 2 ;;
        --skip-reconstruction) SKIP_RECONSTRUCTION=true; shift ;;
        --no-plane-removal)  NO_PLANE_REMOVAL=true;   shift ;;
        --bufferx-root)      BUFFERX_ROOT="$2";       shift 2 ;;
        *)                   echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$SCENE" ]]; then
    echo "ERROR: --scene <object_name> is required" >&2
    exit 1
fi

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
SCENE_DIR="${DATA_ROOT}/${SCENE}"
GT_PLY="${SCENE_DIR}/object.ply"
IMAGE_DIR="${SCENE_DIR}/images"
OUT_PLY="${OUTPUT}/${SCENE}/points.ply"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================================"
echo "  Path Matters Pipeline"
echo "============================================================"
echo "  Scene:        ${SCENE}"
echo "  Images:       ${IMAGE_DIR}"
echo "  GT mesh:      ${GT_PLY}"
echo "  Output:       ${OUTPUT}"
echo "  Config:       ${CONFIG}"
echo "  VGGT env:     ${VGGT_ENV}"
echo "  BUFFER-X env: ${BUFFERX_ENV}"
echo "============================================================"

# --------------------------------------------------------------------------
# Validate inputs
# --------------------------------------------------------------------------
if [[ ! -d "${IMAGE_DIR}" ]]; then
    echo "ERROR: Image directory not found: ${IMAGE_DIR}" >&2
    exit 1
fi
if [[ ! -f "${GT_PLY}" ]]; then
    echo "WARNING: GT mesh not found at ${GT_PLY} — ICP alignment will be skipped"
    GT_PLY=""
fi

mkdir -p "${OUTPUT}/${SCENE}"

# --------------------------------------------------------------------------
# Step 0: Center GT mesh
# --------------------------------------------------------------------------
if [[ -n "${GT_PLY}" ]]; then
    echo ""
    echo ">>> Step 0: Centering GT mesh..."
    conda run -n "${BUFFERX_ENV}" \
        python "${REPO_ROOT}/src/preprocessing/bring_to_center.py" \
        "${GT_PLY}"
    echo "    Done."
fi

# --------------------------------------------------------------------------
# Step 1: VGGT Reconstruction
# --------------------------------------------------------------------------
if [[ "${SKIP_RECONSTRUCTION}" == true && -f "${OUT_PLY}" ]]; then
    echo ""
    echo ">>> Step 1: Skipping reconstruction (${OUT_PLY} exists)"
else
    echo ""
    echo ">>> Step 1: VGGT reconstruction..."
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    conda run -n "${VGGT_ENV}" \
        python "${REPO_ROOT}/src/reconstruction/vggt_reconstruct.py" \
        --image_dir "${IMAGE_DIR}" \
        --out_ply   "${OUT_PLY}" \
        --config    "${CONFIG}"
    echo "    Done → ${OUT_PLY}"
fi

# --------------------------------------------------------------------------
# Step 2: VGGT + ICP alignment
# --------------------------------------------------------------------------
if [[ -n "${GT_PLY}" ]]; then
    echo ""
    echo ">>> Step 2: VGGT + ICP alignment pipeline..."
    PLANE_FLAG=""
    if [[ "${NO_PLANE_REMOVAL}" == true ]]; then
        PLANE_FLAG="--no_plane_removal"
    fi
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    conda run -n "${VGGT_ENV}" \
        python "${REPO_ROOT}/src/reconstruction/vggt_icp_pipeline.py" \
        --scene_dir  "${SCENE_DIR}" \
        --object_ply "${GT_PLY}" \
        --config     "${CONFIG}" \
        --skip_reconstruction \
        ${PLANE_FLAG}
    echo "    Done → ${SCENE_DIR}/icp_results/"
fi

# --------------------------------------------------------------------------
# Step 3: BUFFER-X batch evaluation (optional)
# --------------------------------------------------------------------------
echo ""
echo ">>> Step 3: BUFFER-X evaluation..."
conda run -n "${BUFFERX_ENV}" \
    python "${REPO_ROOT}/src/registration/bufferx_pipeline.py" \
    --recon-root   "${OUTPUT}" \
    --gt-root      "${DATA_ROOT}" \
    --output-base  "${OUTPUT}/runs" \
    --scene-names  "${SCENE}" \
    --bufferx-root "${BUFFERX_ROOT}" \
    --bufferx-env  "${BUFFERX_ENV}" \
    --manual-mode  off \
    --config       "${CONFIG}" \
    --save-viz
echo "    Done → ${OUTPUT}/runs/"

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
METRICS_JSON="${SCENE_DIR}/icp_results/metrics.json"
if [[ -f "${METRICS_JSON}" ]]; then
    echo ""
    echo "============================================================"
    echo "  Results"
    echo "============================================================"
    python3 -c "
import json
m = json.load(open('${METRICS_JSON}'))
print(f'  Fitness:    {m[\"final_fitness\"]:.4f}')
print(f'  RMSE:       {m[\"final_rmse\"]:.6f} m')
print(f'  Scale:      {m[\"scale\"]:.4f}')
print(f'  Time:       {m[\"elapsed_sec\"]:.1f} s')
"
    echo "============================================================"
fi

echo ""
echo "Pipeline complete for scene: ${SCENE}"
