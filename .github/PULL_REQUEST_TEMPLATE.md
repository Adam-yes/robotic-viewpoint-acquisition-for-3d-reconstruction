# Pull Request

## Summary

<!-- What does this PR change, and why? Reference related issues (e.g. "Fixes #42"). -->

## Type of change

<!-- Keep the lines that apply. -->

- Bug fix
- New feature
- Documentation
- Refactoring / maintenance

## Checklist

<!-- From CONTRIBUTING.md — please confirm before requesting review. -->

- [ ] All tests pass (`python -m pytest tests/ -v`)
- [ ] `flake8 src/ tests/ --max-line-length=120 --ignore=E203,W503` is clean
- [ ] Docstrings added/updated for changed functions
- [ ] `config/pipeline_config.yaml` updated if new parameters were added
- [ ] `docs/` updated if behaviour changed
- [ ] `results/metrics/reconstruction_results.csv` updated if new experiments added
- [ ] PR targets `dev` (not `main`)
