# Legacy QuantVLA Environment Notes

This file is kept only to explain the migration status of the legacy QuantVLA environment notes.

The old standalone setup guide used machine-specific paths and a two-environment layout from the original reproduction workspace. quant-vla replaces that workflow with project-level configuration variables and a single public entrypoint.

Use the current documentation instead:

- Main installation: `docs/installation.md`
- Checkpoint preparation: `docs/checkpoints.md`
- Benchmark usage: `docs/usage.md`

The legacy scripts under `scripts/quantvla_legacy/` and auxiliary tools under `tools/quantvla/` are retained for reference and reproduction, but new users should start from the top-level README and the docs listed above.
