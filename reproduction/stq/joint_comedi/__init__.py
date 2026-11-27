"""Centralized joint-output implementation of CoMeDi."""

from .model import (
    ACTION_DIM,
    NUM_ROLES,
    OBS_DIM,
    STATE_DIM,
    SPATIAL_SHAPE,
    ComediPolicy,
    JointActor,
    build_actor,
)

__all__ = [
    "ACTION_DIM", "NUM_ROLES", "OBS_DIM", "STATE_DIM", "SPATIAL_SHAPE",
    "ComediPolicy", "JointActor", "build_actor",
]
