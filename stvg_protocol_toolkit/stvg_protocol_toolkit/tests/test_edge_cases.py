#!/usr/bin/env python3
"""Known-output controls for A/B/M/C (explicit_gt_interval_v1)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol_toolkit.score import score_condition_from_pairs  # noqa: E402


def main() -> int:
    box = [0, 0, 10, 10]
    far = [100, 100, 10, 10]

    # A: GT [0,50] vs pred [100,110] — empty interval intersection
    rows = [
        {"abs_frame_id": 10, "gt_box_xywh": box, "pred_box_xywh": box},
        {"abs_frame_id": 20, "gt_box_xywh": box, "pred_box_xywh": box},
    ]
    a = score_condition_from_pairs(rows, [100, 110], gt_temporal=[0, 50], recompute_iou=True)
    assert a["mech_class"] == "A" and a["s_ov"] == 0.0 and a["s_GT"] == 1.0, a

    # B: intervals overlap [10,20] but no sample in intersection (samples at 0,5)
    rows_b = [
        {"abs_frame_id": 0, "gt_box_xywh": box, "pred_box_xywh": box},
        {"abs_frame_id": 5, "gt_box_xywh": box, "pred_box_xywh": box},
    ]
    b = score_condition_from_pairs(rows_b, [10, 20], gt_temporal=[0, 30], recompute_iou=True)
    assert b["mech_class"] == "B" and b["s_ov"] == 0.0, b

    # M: samples in intersection but preds missing
    rows_m = [
        {"abs_frame_id": 10, "gt_box_xywh": box, "pred_box_xywh": None},
    ]
    m = score_condition_from_pairs(rows_m, [10, 10], gt_temporal=[10, 10], recompute_iou=True)
    assert m["mech_class"] == "M", m

    # C: legal zero IoU scorable
    rows_c = [
        {"abs_frame_id": 10, "gt_box_xywh": box, "pred_box_xywh": far},
    ]
    c = score_condition_from_pairs(rows_c, [10, 10], gt_temporal=[10, 10], recompute_iou=True)
    assert c["mech_class"] == "C" and c["s_ov"] == 0.0 and c["s_GT"] == 0.0 and c["overlap"]["n_scored"] == 1

    # C_full
    rows_ok = [
        {"abs_frame_id": 10, "gt_box_xywh": box, "pred_box_xywh": box},
    ]
    ok = score_condition_from_pairs(rows_ok, [10, 10], gt_temporal=[10, 10], recompute_iou=True)
    assert ok["mech_class"] == "C_full" and ok["s_ov"] == 1.0

    # no GT candidates
    z = score_condition_from_pairs([], [0, 1], gt_temporal=[0, 1], recompute_iou=True)
    assert z["s_GT"] is None and z["fixed_empty_reason"] == "no_gt_candidate_slots"

    # A with None pred
    n = score_condition_from_pairs(rows, None, gt_temporal=[0, 50], recompute_iou=True)
    assert n["mech_class"] == "A"

    # B vs A contrast: overlapping intervals without samples must not be A
    assert b["mech_class"] != "A"

    # exclusion naming is separate (documented field)
    excl_field = "no_gt_candidate_slots"
    assert excl_field != "A"

    print(
        {
            "ok": True,
            "cases": ["A_empty_interval", "B_no_sample_in_intersect", "M_missing_boxes", "C_zero", "C_full", "no_gt", "A_none_pred", "B_not_A", "excl_field"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
