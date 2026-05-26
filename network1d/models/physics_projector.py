"""Physics projector helpers.

This compatibility layer keeps the refactored package layout executable while
preserving original behavior (identity projection).
"""


def project_physics_constraints(features, *_, **__):
    """Return inputs unchanged to preserve original repository behavior."""
    return features


__all__ = ["project_physics_constraints"]
