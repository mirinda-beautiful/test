"""Box IoU helpers (xywh absolute pixels). No project imports."""
from __future__ import annotations

from typing import Optional, Sequence


def xywh_to_xyxy(b: Sequence[float]) -> tuple:
    x, y, w, h = [float(v) for v in b]
    return x, y, x + w, y + h


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def iou_xywh(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    return iou_xyxy(xywh_to_xyxy(a), xywh_to_xyxy(b))
