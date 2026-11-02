"""Re-score overlap / fixed-GT from portable pair rows + temporal intervals.

Mechanism taxonomy (explicit_gt_interval_v1):
  A — GT interval ∩ pred interval is empty → s_ov = 0
  B — intervals intersect, but no GT-candidate sample falls in the intersection → s_ov = 0
  M — samples exist in the intersection but no valid box pair (gap; not a time mechanism)
  C / C_full — nonempty scored support (legal zero IoU allowed)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .iou import iou_xywh

__all__ = ["iou_xywh", "score_condition_from_pairs", "intervals_overlap"]


def _mean(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    return float(np.mean(np.asarray(list(vals), dtype=np.float64)))


def _as_interval(x: Optional[Sequence[Any]]) -> Optional[Tuple[int, int]]:
    if x is None or x[0] is None or x[1] is None:
        return None
    return int(x[0]), int(x[1])


def intervals_overlap(a: Optional[Sequence[Any]], b: Optional[Sequence[Any]]) -> bool:
    ia, ib = _as_interval(a), _as_interval(b)
    if ia is None or ib is None:
        return False
    return max(ia[0], ib[0]) <= min(ia[1], ib[1])


def score_condition_from_pairs(
    pair_rows: Sequence[Dict[str, Any]],
    pred_temporal_frames: Optional[Sequence[int]],
    *,
    gt_temporal: Optional[Sequence[int]] = None,
    empty_overlap_value: float = 0.0,
    recompute_iou: bool = True,
) -> Dict[str, Any]:
    frames: List[Dict[str, Any]] = []
    for r in pair_rows:
        f = int(r["abs_frame_id"])
        gt = r.get("gt_box_xywh")
        pr = r.get("pred_box_xywh")
        if recompute_iou:
            iou = iou_xywh(gt, pr)
        else:
            iou = r.get("iou")
            if iou is not None:
                iou = float(iou)
            elif gt is not None and pr is not None:
                iou = iou_xywh(gt, pr)
        missing_gt = gt is None
        missing_pred = pr is None
        usable = (not missing_gt) and (not missing_pred) and iou is not None
        frames.append(
            {
                "abs_frame_id": f,
                "sampled_idx": r.get("sampled_idx"),
                "gt_box_xywh": gt,
                "pred_box_xywh": pr,
                "iou": iou,
                "usable": usable,
                "missing_gt": missing_gt,
                "missing_pred": missing_pred,
            }
        )

    gt_ious = [float(x["iou"]) for x in frames if x["usable"]]
    n_cand = len(frames)
    n_missing_gt = sum(1 for x in frames if x["missing_gt"])
    n_missing_pred = sum(1 for x in frames if x["missing_pred"] and not x["missing_gt"])
    if not gt_ious:
        s_GT = None
        fixed_empty = "unscorable_no_valid_pairs" if n_cand else "no_gt_candidate_slots"
        fixed_scorable = False
    else:
        s_GT = _mean(gt_ious)
        fixed_empty = "ok"
        fixed_scorable = True

    pred_t = _as_interval(pred_temporal_frames)
    gt_t = _as_interval(gt_temporal)

    # If gt_temporal missing, fall back to span of candidate frames (Tube/DeViL evidence).
    if gt_t is None and frames:
        fs = [x["abs_frame_id"] for x in frames]
        gt_t = (min(fs), max(fs))

    if pred_t is None:
        s_ov = float(empty_overlap_value)
        ov_meta = {
            "empty_overlap": True,
            "n_in": 0,
            "n_scored": 0,
            "empty_rule": "no_pred_temporal_to_zero",
            "mech_class": "A",
            "time_intersection": None,
        }
    elif not intervals_overlap(gt_t, pred_t):
        s_ov = float(empty_overlap_value)
        ov_meta = {
            "empty_overlap": True,
            "n_in": 0,
            "n_scored": 0,
            "empty_rule": "empty_time_intersection_to_zero",
            "mech_class": "A",
            "time_intersection": None,
        }
    else:
        assert gt_t is not None and pred_t is not None
        i0, i1 = max(gt_t[0], pred_t[0]), min(gt_t[1], pred_t[1])
        in_frames = [x for x in frames if i0 <= int(x["abs_frame_id"]) <= i1]
        if not in_frames:
            s_ov = float(empty_overlap_value)
            ov_meta = {
                "empty_overlap": True,
                "n_in": 0,
                "n_scored": 0,
                "empty_rule": "time_intersect_but_no_sample_support_to_zero",
                "mech_class": "B",
                "time_intersection": [i0, i1],
            }
        else:
            scored = [float(x["iou"]) for x in in_frames if x["usable"]]
            if not scored:
                s_ov = float(empty_overlap_value)
                ov_meta = {
                    "empty_overlap": True,
                    "n_in": len(in_frames),
                    "n_scored": 0,
                    "empty_rule": "samples_in_intersection_but_no_valid_pairs",
                    "mech_class": "M",
                    "time_intersection": [i0, i1],
                }
            else:
                s_ov = _mean(scored)
                all_zero = all(v == 0.0 for v in scored)
                ov_meta = {
                    "empty_overlap": False,
                    "n_in": len(in_frames),
                    "n_scored": len(scored),
                    "empty_rule": "ok",
                    "mech_class": "C" if all_zero else "C_full",
                    "time_intersection": [i0, i1],
                }

    return {
        "s_ov": float(s_ov) if s_ov is not None else None,
        "s_GT": s_GT,
        "s_GT_scorable": fixed_scorable,
        "fixed_empty_reason": fixed_empty,
        "n_gt_candidate": n_cand,
        "n_gt_valid_pairs": len(gt_ious),
        "n_missing_gt": n_missing_gt,
        "n_missing_pred": n_missing_pred,
        "frame_ious": frames,
        "overlap": ov_meta,
        "mech_class": ov_meta["mech_class"],
        "g": (None if s_GT is None else float(s_GT) - float(s_ov if s_ov is not None else 0.0)),
        "gt_temporal_used": list(gt_t) if gt_t else None,
        "pred_temporal_used": list(pred_t) if pred_t else None,
    }
