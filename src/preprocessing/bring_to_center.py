"""
Center a 3D mesh or point cloud at the origin.

For ground-truth meshes whose centroid is offset from world origin,
this tool translates vertices so the centroid lies at (0, 0, 0).
This is required before ICP alignment because the pipeline assumes
the reference object is centered.

Usage:
    python src/preprocessing/bring_to_center.py /path/to/model.stl
    python src/preprocessing/bring_to_center.py /path/to/model.ply --no-inplace --output centered.ply
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def center_mesh(input_path: Path, output_path: Path) -> None:
    """Translate mesh/point-cloud centroid to the origin.

    Supports any format readable by Trimesh (.stl, .obj, .ply, .glb, …).

    Args:
        input_path:  Source file.
        output_path: Destination file (may be the same as input for in-place edit).
    """
    import trimesh

    geometry = trimesh.load(str(input_path), force="mesh")

    if hasattr(geometry, "vertices"):
        centroid = geometry.centroid
        geometry.vertices -= centroid
        logger.info("Shifted by centroid %s", centroid)
    else:
        # Fallback for point-cloud types
        pts = np.asarray(geometry.vertices if hasattr(geometry, "vertices") else geometry.points)
        centroid = pts.mean(axis=0)
        pts -= centroid
        logger.info("Shifted point cloud by centroid %s", centroid)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    geometry.export(str(output_path))
    logger.info("Saved centered geometry to %s", output_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Center a 3D mesh/point cloud at the origin")
    p.add_argument("input", help="Input 3D file")
    p.add_argument("--output", default=None, help="Output file (default: overwrite input)")
    p.add_argument("--no-inplace", action="store_true",
                   help="Write to <stem>_centered<ext> instead of overwriting")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error("File not found: %s", input_path)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    elif args.no_inplace:
        output_path = input_path.parent / f"{input_path.stem}_centered{input_path.suffix}"
    else:
        output_path = input_path

    center_mesh(input_path, output_path)


if __name__ == "__main__":
    main()
