"""Supervised loss utilities."""

import torch as th


def mse(input_tensor, target_tensor, mask=None):
    """Mean squared error, equivalent to legacy implementation."""
    if mask is None:
        return ((input_tensor - target_tensor) ** 2).mean()
    return (mask * (input_tensor - target_tensor) ** 2).mean()


def mae(input_tensor, target_tensor, mask=None):
    """Mean absolute error, equivalent to legacy implementation."""
    if mask is None:
        return th.abs(input_tensor - target_tensor).mean()
    return (mask * th.abs(input_tensor - target_tensor)).mean()


__all__ = ["mse", "mae"]
