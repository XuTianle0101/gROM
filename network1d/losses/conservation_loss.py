"""Conservation loss compatibility shim."""

import torch as th


def conservation_loss(*_, **__):
    """Return zero to preserve existing training objective."""
    return th.tensor(0.0)


__all__ = ["conservation_loss"]
