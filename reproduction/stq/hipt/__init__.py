"""Hierarchical Population Training (HiPT) on the three-chef SharedKitchen task."""

from .model import HiPTPolicy, influence_kl

__all__ = ["HiPTPolicy", "influence_kl"]
