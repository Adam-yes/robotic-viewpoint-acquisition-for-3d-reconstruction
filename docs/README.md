# Documentation

Welcome to the Path Matters documentation. Start with the document that matches your goal:

| I want to… | Read |
|------------|------|
| Understand the scientific approach and findings | [Methodology](methodology.md) |
| Reproduce the paper's results end to end | [Pipeline Reproduction Guide](pipeline.md) |
| Understand the dataset layout, storage, and licenses | [Data Management](data_management.md) |
| Set up the Isaac Sim capture environment | [Isaac Sim Capture Guide](guides/isaac_sim_capture.md) |
| See the full 28-object dataset list | [Dataset Information](../data/metadata/dataset_info.md) |
| Contribute code | [Contributing Guide](../CONTRIBUTING.md) |

## Reading order for newcomers

1. [Project README](../README.md) — overview, key results, quick start
2. [Methodology](methodology.md) — what was done and why
3. [Pipeline Reproduction Guide](pipeline.md) — how to run it yourself
4. [Data Management](data_management.md) — where the data lives

## Configuration reference

Every pipeline stage reads its defaults from a single file:
[`config/pipeline_config.yaml`](../config/pipeline_config.yaml).
Command-line arguments override individual keys.
