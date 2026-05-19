"""Deprecated compatibility wrapper for legacy PPO/BC code."""

from legacy.ppo_bc.environment import *  # noqa: F401,F403


if __name__ == "__main__":
    import warnings

    warnings.warn(
        "environment.py is deprecated and kept only for legacy PPO/BC imports.",
        FutureWarning,
        stacklevel=1,
    )
