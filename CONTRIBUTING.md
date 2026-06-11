# Contributing to Path Matters

Thank you for your interest in contributing!

## Branch Model

```
main       ← protected, requires 2 reviewer approvals
dev        ← integration branch for tested features
feature/*  ← development branches (one per feature/fix)
```

All work happens on `feature/*` branches.  Open a pull request to `dev`,
not directly to `main`.

## Workflow

1. Fork the repository (external contributors) or create a branch (team members):
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/my-feature
   ```

2. Make your changes.  Keep commits small and descriptive.

3. Run the test suite locally before pushing:
   ```bash
   conda activate vggt   # or bufferx_o3d
   python -m pytest tests/ -v --tb=short
   ```

4. Ensure flake8 passes:
   ```bash
   flake8 src/ tests/ --max-line-length=120 --ignore=E203,W503
   ```

5. Push and open a pull request to `dev`.

## Code Style

- Python 3.10, type hints encouraged
- `flake8` with `--max-line-length=120`
- No commented-out code blocks
- No print() in library code — use `logging`
- Docstrings on all public functions (one-line summary + Args/Returns)

## Commit Messages

Use imperative mood, reference issues where applicable:

```
Add proximity shaping reward to RL environment

Fixes #42 — exp_06 failed to discover coverage rewards without shaping.
```

## Pull Request Checklist

- [ ] All tests pass (`pytest tests/`)
- [ ] flake8 clean
- [ ] Docstrings added/updated for changed functions
- [ ] `config/pipeline_config.yaml` updated if new parameters were added
- [ ] `docs/` updated if behaviour changed
- [ ] `results/metrics/reconstruction_results.csv` updated if new experiments added

## Reporting Issues

Use GitHub Issues — the [issue templates](.github/ISSUE_TEMPLATE) will guide you.
For bugs, include:
- OS and GPU model
- Conda environment (`conda list`)
- Minimal reproduction script
- Full traceback

## Questions

Open a Discussion on GitHub or contact the team via TU Berlin channels.
