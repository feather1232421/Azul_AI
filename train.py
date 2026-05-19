"""Deprecated compatibility wrapper for legacy PPO/BC code."""

from legacy.ppo_bc.train import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy
    import warnings

    warnings.warn(
        "train.py is deprecated. Use `python -m legacy.ppo_bc.train` instead.",
        FutureWarning,
        stacklevel=1,
    )
    runpy.run_module("legacy.ppo_bc.train", run_name="__main__")
