# Dataset Information

> [Project README](../../README.md) · [Data Management](../../docs/data_management.md) · **Dataset Information**

## Overview

The Path Matters benchmark dataset consists of 28 household objects rendered
in Isaac Sim 5.1 with accompanying ground-truth meshes.

## Object List

| ID  | Name                  | Source     | License   | GT format | Notes |
|-----|-----------------------|------------|-----------|-----------|-------|
| 001 | alarm_clock           | Objaverse  | CC-BY     | .ply      | Asymmetric geometry |
| 002 | mug                   | Objaverse  | CC0       | .ply      | Symmetric (cylinder) |
| 003 | wrench                | Objaverse  | CC-BY     | .ply      | Thin, elongated |
| 004 | bottle                | Objaverse  | CC0       | .ply      | Rotationally symmetric |
| 005 | book                  | Objaverse  | CC-BY     | .ply      | Flat slab |
| 006 | dice                  | manual     | Proprietary | .stl   | 6-faced cube |
| 007 | baby_yoda             | Objaverse  | CC-BY     | .ply      | Complex organic shape |
| 008 | hammer                | Objaverse  | CC0       | .ply      | L-shaped |
| 009 | screwdriver           | Objaverse  | CC-BY     | .ply      | Thin cylindrical |
| 010 | scissors              | Objaverse  | CC0       | .ply      | Dual-arm, thin |
| 011 | teddy_bear            | Objaverse  | CC-BY     | .ply      | Soft organic |
| 012 | teapot                | Objaverse  | CC0       | .ply      | Handle + spout |
| 013 | bowl                  | Objaverse  | CC-BY     | .ply      | Concave interior |
| 014 | banana                | YCB-V      | CC-BY-NC  | .ply      | BOP benchmark |
| 015 | cracker_box           | YCB-V      | CC-BY-NC  | .ply      | BOP benchmark |
| 016 | tomato_soup_can       | YCB-V      | CC-BY-NC  | .ply      | BOP benchmark |
| 017 | mustard_bottle        | YCB-V      | CC-BY-NC  | .ply      | BOP benchmark |
| 018 | drill                 | Objaverse  | CC-BY     | .ply      | Complex tool |
| 019 | stapler               | Objaverse  | CC0       | .ply      | Hinged |
| 020 | calculator            | Objaverse  | CC-BY     | .ply      | Flat + buttons |
| 021 | headphones            | Objaverse  | CC-BY     | .ply      | Large, curved |
| 022 | mouse                 | Objaverse  | CC0       | .ply      | Ergonomic |
| 023 | keyboard              | Objaverse  | CC-BY     | .ply      | Flat, many features |
| 024 | plant_pot             | Objaverse  | CC0       | .ply      | Tapered cylinder |
| 025 | watch                 | Objaverse  | CC-BY     | .ply      | Small, detailed |
| 026 | glasses               | Objaverse  | CC0       | .ply      | Thin frame |
| 027 | remote_control        | Objaverse  | CC-BY     | .ply      | Flat, many buttons |
| 028 | water_bottle          | manual     | Proprietary | .stl   | Real-world scan |

## Capture Parameters

- Simulator:    Isaac Sim 5.1
- Renderer:     RTX Path Tracer
- Resolution:   720 × 720 px
- Views/object: 54
- Format:       PNG
- Lighting:     Single point light at (1, 0, 2) m

## Ground-Truth Mesh Processing

All GT meshes were:
1. Converted to `.ply` (50 000 points, uniform sampling)
2. Centered at the origin (`src/preprocessing/bring_to_center.py`)
3. Inspected visually to verify mesh integrity
