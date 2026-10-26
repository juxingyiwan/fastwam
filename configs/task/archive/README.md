# Archived task configurations

The parent `configs/task/` directory contains the eight current B601 real-robot configurations.

- `simulation/`: LIBERO and RoboTwin recipes, retained for simulation workflows.
- `legacy/`: historical B601 full-state continuation to ten epochs. It retains the saved learning-rate schedule and is not the fresh ten-epoch recipe.

Use the archive prefix when explicitly selecting these tasks, for example:

```bash
python scripts/train.py task=archive/simulation/libero_uncond_2cam224_1e-4
```

Existing simulation configurations, command examples and the legacy resume entrypoint use the updated selectors. Model checkpoints, dataset paths and current B601 training recipes are unchanged.
The archive manifest records the original paths and content hashes. Inheritance between archived RoboTwin recipes and the legacy B601 recipe's active base uses explicit absolute Hydra paths.
