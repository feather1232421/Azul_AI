"""Deprecated compatibility wrapper for legacy PPO/BC code."""

from legacy.ppo_bc.trian_nn import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy
    import warnings

    warnings.warn(
        "trian_nn.py is deprecated. Use `python -m legacy.ppo_bc.trian_nn` instead.",
        FutureWarning,
        stacklevel=1,
    )
    runpy.run_module("legacy.ppo_bc.trian_nn", run_name="__main__")
