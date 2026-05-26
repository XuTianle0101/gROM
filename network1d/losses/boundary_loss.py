"""Boundary loss compatibility shim."""

import torch as th


def boundary_loss(*_, **__):
    """Return zero to preserve existing training objective."""
    return th.tensor(0.0)


__all__ = ["boundary_loss"]
