"""Portable STVG protocol audit toolkit (CPU-only)."""
from .score import score_condition_from_pairs, iou_xywh
from .aggregate import aggregate_group, aggregate_regen

__all__ = [
    "score_condition_from_pairs",
    "iou_xywh",
    "aggregate_group",
    "aggregate_regen",
]
