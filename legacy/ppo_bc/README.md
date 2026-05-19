# Legacy PPO/BC Pipeline

This folder contains older Azul experiments that are no longer part of the active transformer self-play loop.

Current active loop:

- `run_iteration.py`
- `loop_train.py`
- `get_dataset.py`
- `train_mcts_nn.py`
- `battle.py`
- `server.py`

Legacy status means:

- kept for checkpoint compatibility and historical reference
- not used by the transformer champion loop
- no new feature work should land here unless needed for migration/debugging

Temporary root-level wrappers still exist for commands such as `train.py` and `bc_to_ppo.py`, but they are deprecated and may be removed after the migration period.
