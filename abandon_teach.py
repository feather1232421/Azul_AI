"""Deprecated compatibility wrapper for legacy PPO/BC code."""

from legacy.ppo_bc.abandon_teach import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy
    import warnings

    warnings.warn(
        "abandon_teach.py is deprecated. Use `python -m legacy.ppo_bc.abandon_teach` instead.",
        FutureWarning,
        stacklevel=1,
    )
    runpy.run_module("legacy.ppo_bc.abandon_teach", run_name="__main__")
