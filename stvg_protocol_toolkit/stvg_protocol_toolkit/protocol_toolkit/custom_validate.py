"""Input checks for custom STVG prediction JSONL (user exports)."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


REQUIRED_TOP = ("group", "dataset", "model", "sample_id", "video_id", "round", "arms")
REQUIRED_ARM_FIELDS = ("pred_temporal_frames", "gt_temporal", "pair_rows")


def _finite_number(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def _as_int_interval(x: Any) -> Optional[Tuple[int, int]]:
    if x is None or not isinstance(x, (list, tuple)) or len(x) != 2:
        return None
    try:
        a, b = int(x[0]), int(x[1])
    except (TypeError, ValueError):
        return None
    return a, b


def _box_ok(box: Any, *, allow_none: bool) -> Tuple[bool, str]:
    if box is None:
        return (True, "missing_ok") if allow_none else (False, "box_is_null")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False, "box_must_be_xywh_len4"
    if not all(_finite_number(v) for v in box):
        return False, "box_non_finite"
    w, h = float(box[2]), float(box[3])
    if w < 0 or h < 0:
        return False, "box_negative_wh"
    return True, "ok"


def validate_custom_records(
    records: Sequence[Dict[str, Any]],
    *,
    cluster_field: str = "video_id",
    require_clean: bool = False,
) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    coverage: List[Dict[str, Any]] = []

    def err(msg: str, **kw: Any) -> None:
        errors.append({"message": msg, **kw})

    def warn(msg: str, **kw: Any) -> None:
        warnings.append({"message": msg, **kw})

    if cluster_field not in ("video_id", "source_video_id"):
        err("cluster_field must be video_id or source_video_id", cluster_field=cluster_field)

    id_rounds = []
    for i, rec in enumerate(records):
        loc = {"line": i + 1, "sample_id": rec.get("sample_id"), "round": rec.get("round")}
        for k in REQUIRED_TOP:
            if k not in rec:
                err(f"missing top-level field '{k}'", **loc)
        if cluster_field == "source_video_id" and not rec.get("source_video_id"):
            err("cluster_field=source_video_id but source_video_id is missing/empty", **loc)
        if not rec.get("video_id"):
            err("video_id is required", **loc)

        arms = rec.get("arms") or {}
        if not isinstance(arms, dict):
            err("arms must be an object", **loc)
            continue

        for need in ("S", "T"):
            if need not in arms:
                err(f"arms.{need} is required for paired S/T diagnostics", **loc)
        if require_clean and "Clean" not in arms:
            err("Clean arm required by config.require_clean", **loc)
        elif "Clean" not in arms:
            warn("Clean arm absent; baseline Clean checks skipped (S/T still scored)", **loc)

        # exclusion path
        if rec.get("exclusion_reason") == "no_gt_candidate_slots":
            coverage.append(
                {
                    **loc,
                    "group": rec.get("group"),
                    "status": "excluded",
                    "reason": "no_gt_candidate_slots",
                    "n_gt_candidate_slots": int(rec.get("n_gt_candidate_slots") or 0),
                    "n_valid_pairs_S": 0,
                    "n_valid_pairs_T": 0,
                }
            )
            continue

        arm_stats = {}
        for name, arm in arms.items():
            if not isinstance(arm, dict):
                err(f"arms.{name} must be an object", **loc)
                continue
            if arm.get("missing_evidence"):
                warn(f"arms.{name}.missing_evidence set", **loc)
                arm_stats[name] = {"missing_evidence": True}
                continue
            for f in REQUIRED_ARM_FIELDS:
                if f not in arm:
                    err(f"arms.{name} missing '{f}'", **loc)

            gt = _as_int_interval(arm.get("gt_temporal"))
            pr = _as_int_interval(arm.get("pred_temporal_frames"))
            if gt is None:
                err(f"arms.{name}.gt_temporal must be [start,end] integer frames (inclusive)", **loc)
            elif gt[0] > gt[1]:
                err(f"arms.{name}.gt_temporal start>end", gt=gt, **loc)
            if pr is None:
                err(
                    f"arms.{name}.pred_temporal_frames must be [start,end] integer frames (inclusive)",
                    **loc,
                )
            elif pr[0] > pr[1]:
                err(f"arms.{name}.pred_temporal_frames start>end", pred=pr, **loc)

            rows = arm.get("pair_rows") or []
            if not isinstance(rows, list):
                err(f"arms.{name}.pair_rows must be a list", **loc)
                continue
            seen_frames: Dict[int, Dict[str, Any]] = {}
            n_valid = 0
            for j, row in enumerate(rows):
                if "abs_frame_id" not in row:
                    err(f"arms.{name}.pair_rows[{j}] missing abs_frame_id", **loc)
                    continue
                try:
                    fid = int(row["abs_frame_id"])
                except (TypeError, ValueError):
                    err(f"arms.{name}.pair_rows[{j}].abs_frame_id not int-like", **loc)
                    continue
                if fid in seen_frames:
                    prev = seen_frames[fid]
                    if prev.get("gt_box_xywh") != row.get("gt_box_xywh") or prev.get("pred_box_xywh") != row.get(
                        "pred_box_xywh"
                    ):
                        err(
                            f"arms.{name} duplicate abs_frame_id={fid} with conflicting boxes",
                            **loc,
                        )
                    else:
                        warn(f"arms.{name} duplicate abs_frame_id={fid}", **loc)
                seen_frames[fid] = row
                ok_gt, why_gt = _box_ok(row.get("gt_box_xywh"), allow_none=True)
                ok_pr, why_pr = _box_ok(row.get("pred_box_xywh"), allow_none=True)
                if not ok_gt:
                    err(f"arms.{name}.pair_rows[{j}] bad gt_box_xywh ({why_gt})", **loc)
                if not ok_pr:
                    err(f"arms.{name}.pair_rows[{j}] bad pred_box_xywh ({why_pr})", **loc)
                if row.get("gt_box_xywh") is not None and row.get("pred_box_xywh") is not None and ok_gt and ok_pr:
                    n_valid += 1

            sampling = arm.get("sampling") or {}
            map_ids = sampling.get("frame_ids") or sampling.get("gt_candidate_frame_ids")
            if map_ids is not None:
                try:
                    map_set = {int(x) for x in map_ids}
                except (TypeError, ValueError):
                    err(f"arms.{name}.sampling frame id list not int-like", **loc)
                    map_set = set()
                row_set = set(seen_frames)
                if row_set - map_set:
                    warn(
                        f"arms.{name} pair_rows frames not listed in sampling map",
                        extra=sorted(row_set - map_set)[:10],
                        **loc,
                    )

            arm_stats[name] = {
                "n_candidate_slots": len(rows),
                "n_unique_frames": len(seen_frames),
                "n_valid_pairs": n_valid,
                "gt_temporal": list(gt) if gt else None,
                "pred_temporal": list(pr) if pr else None,
            }

        # S/T shared GT / grid consistency
        if "S" in arms and "T" in arms and isinstance(arms["S"], dict) and isinstance(arms["T"], dict):
            gs, gt_ = arms["S"].get("gt_temporal"), arms["T"].get("gt_temporal")
            if gs is not None and gt_ is not None and list(gs) != list(gt_):
                err("S and T gt_temporal differ; shared GT required", gt_S=gs, gt_T=gt_, **loc)
            fs = [int(r["abs_frame_id"]) for r in (arms["S"].get("pair_rows") or []) if "abs_frame_id" in r]
            ft = [int(r["abs_frame_id"]) for r in (arms["T"].get("pair_rows") or []) if "abs_frame_id" in r]
            if fs and ft and sorted(set(fs)) != sorted(set(ft)):
                warn("S and T pair_rows frame sets differ; fixed-GT grids should match", **loc)

        coverage.append(
            {
                **loc,
                "group": rec.get("group"),
                "status": "candidate",
                "reason": rec.get("exclusion_reason") or "",
                "n_gt_candidate_slots": (arm_stats.get("S") or {}).get("n_candidate_slots"),
                "n_valid_pairs_S": (arm_stats.get("S") or {}).get("n_valid_pairs"),
                "n_valid_pairs_T": (arm_stats.get("T") or {}).get("n_valid_pairs"),
                "arms": arm_stats,
            }
        )
        id_rounds.append((rec.get("sample_id"), rec.get("round"), rec.get("group")))

    dup = [k for k, c in Counter(id_rounds).items() if c > 1]
    for sample_id, rnd, group in dup:
        err("duplicate sample_id/round/group", sample_id=sample_id, round=rnd, group=group)

    groups = sorted({r.get("group") for r in records if r.get("group")})
    return {
        "ok": len(errors) == 0,
        "n_records": len(records),
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "groups": groups,
        "errors": errors,
        "warnings": warnings,
        "coverage_rows": coverage,
    }
