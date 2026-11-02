"""Group aggregation: W, decomposition, ranking, regen paired deltas."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .score import score_condition_from_pairs
from .video_cluster_bootstrap import boot_from_pairs

NEAR = 1e-8


def sgn(x: float, eps: float = NEAR) -> int:
    if abs(x) <= eps:
        return 0
    return 1 if x > 0 else -1


def _as_score(
    arm: Dict[str, Any],
    *,
    tube_legacy_empty_zero: bool,
    recompute_iou: bool,
) -> Dict[str, Any]:
    out = score_condition_from_pairs(
        arm.get("pair_rows") or [],
        arm.get("pred_temporal_frames"),
        gt_temporal=arm.get("gt_temporal"),
        recompute_iou=recompute_iou,
    )
    if tube_legacy_empty_zero and out["s_GT"] is None:
        out["s_GT"] = 0.0
        out["s_GT_scorable"] = True
        out["fixed_empty_reason"] = "tube_legacy_empty_returns_zero"
        out["g"] = float(out["s_GT"]) - float(out["s_ov"])
    return out


def score_query(
    rec: Dict[str, Any],
    *,
    recompute_iou: bool = True,
    ranking_eps: float = NEAR,
) -> Dict[str, Any]:
    model = rec.get("model", "")
    tube = "Tube" in model or model.lower() == "tubedetr"
    arms_out = {}
    for name in ("Clean", "S", "T"):
        if name not in rec.get("arms", {}):
            continue
        arms_out[name] = _as_score(
            rec["arms"][name],
            tube_legacy_empty_zero=tube,
            recompute_iou=recompute_iou,
        )
    if "S" not in arms_out or "T" not in arms_out:
        return {"sample_id": rec["sample_id"], "scorable_pair": False, "reason": "missing_S_or_T"}

    # Mechanism M on either attack arm → gap (not time-mechanism A/B)
    for name in ("S", "T"):
        if arms_out[name].get("mech_class") == "M":
            return {
                "sample_id": rec["sample_id"],
                "scorable_pair": False,
                "reason": "mechanism_M_missing_box_pairs",
                "arms": arms_out,
            }

    sS, sT = arms_out["S"], arms_out["T"]
    if sS["s_GT"] is None or sT["s_GT"] is None or sS["s_ov"] is None or sT["s_ov"] is None:
        return {
            "sample_id": rec["sample_id"],
            "scorable_pair": False,
            "reason": "unscorable_fixed_or_overlap",
            "arms": arms_out,
        }
    d_ov = float(sT["s_ov"]) - float(sS["s_ov"])
    d_GT = float(sT["s_GT"]) - float(sS["s_GT"])
    w = d_GT - d_ov
    g_S, g_T = float(sS["g"]), float(sT["g"])
    so, sg = sgn(d_ov, ranking_eps), sgn(d_GT, ranking_eps)
    if so * sg < 0:
        rank = "strict_reverse"
    elif so == 0 and sg != 0:
        rank = "tie_to_nontie"
    elif so != 0 and sg == 0:
        rank = "nontie_to_tie"
    elif so == sg:
        rank = "preserved"
    else:
        rank = "other"

    def decomp_term(k: str) -> float:
        def hit(cls: str, key: str) -> bool:
            if key == "C":
                return cls in ("C", "C_full")
            return cls == key

        t = 0.0
        if hit(sT["mech_class"], k):
            t += g_T
        if hit(sS["mech_class"], k):
            t -= g_S
        return t

    return {
        "sample_id": rec["sample_id"],
        "video_id": rec.get("video_id"),
        "source_video_id": rec.get("source_video_id") or rec.get("source_group"),
        "cluster_id": rec.get("cluster_id"),
        "group": rec.get("group"),
        "dataset": rec.get("dataset"),
        "model": rec.get("model"),
        "round": rec.get("round", "baseline"),
        "scorable_pair": True,
        "s_ov_S": float(sS["s_ov"]),
        "s_ov_T": float(sT["s_ov"]),
        "s_GT_S": float(sS["s_GT"]),
        "s_GT_T": float(sT["s_GT"]),
        "d_ov": d_ov,
        "d_GT": d_GT,
        "w": w,
        "g_S": g_S,
        "g_T": g_T,
        "w_from_g": g_T - g_S,
        "mech_class_S": sS["mech_class"],
        "mech_class_T": sT["mech_class"],
        "decomp_A": decomp_term("A"),
        "decomp_B": decomp_term("B"),
        "decomp_C": decomp_term("C"),
        "rank_category": rank,
        "strict_reverse": rank == "strict_reverse",
        "tie_change": rank in ("tie_to_nontie", "nontie_to_tie"),
        "arms": arms_out,
        "orig_w": rec.get("orig_w"),
        "orig_d_ov": rec.get("orig_d_ov"),
        "orig_d_GT": rec.get("orig_d_GT"),
    }


def aggregate_group(
    query_scores: Sequence[Dict[str, Any]],
    *,
    B: int = 10000,
    seed: int = 20261008,
    cluster_key: str = "cluster_id",
) -> Dict[str, Any]:
    rows = [r for r in query_scores if r.get("scorable_pair")]
    if not rows:
        return {"n": 0}
    # Deterministic cluster order: sorted canonical ID strings (bootstrap module also sorts).
    for r in rows:
        r[cluster_key] = str(r.get(cluster_key) or r.get("video_id"))
    clusters = [str(r[cluster_key]) for r in rows]
    ws = [float(r["w"]) for r in rows]
    dov = [float(r["d_ov"]) for r in rows]
    dgt = [float(r["d_GT"]) for r in rows]
    boot_w = boot_from_pairs(ws, clusters, B=B, seed=seed)
    boot_ov = boot_from_pairs(dov, clusters, B=B, seed=seed)
    boot_gt = boot_from_pairs(dgt, clusters, B=B, seed=seed)
    boot_A = boot_from_pairs([float(r["decomp_A"]) for r in rows], clusters, B=B, seed=seed)
    boot_B = boot_from_pairs([float(r["decomp_B"]) for r in rows], clusters, B=B, seed=seed)
    boot_C = boot_from_pairs([float(r["decomp_C"]) for r in rows], clusters, B=B, seed=seed)
    boot_strict = boot_from_pairs([1.0 if r["strict_reverse"] else 0.0 for r in rows], clusters, B=B, seed=seed)
    boot_tie = boot_from_pairs([1.0 if r.get("tie_change") else 0.0 for r in rows], clusters, B=B, seed=seed)
    A, Bb, C = float(boot_A["mean"]), float(boot_B["mean"]), float(boot_C["mean"])
    residual = (A + Bb + C) - float(boot_w["mean"])
    n_strict = sum(1 for r in rows if r["strict_reverse"])
    n_tie = sum(1 for r in rows if r.get("tie_change"))
    return {
        "n": len(rows),
        "n_clusters": boot_w["n_video"],
        "W_mean": boot_w["mean"],
        "W_ci95": boot_w["ci95"],
        "Delta_ov_mean": boot_ov["mean"],
        "Delta_ov_ci95": boot_ov["ci95"],
        "Delta_GT_mean": boot_gt["mean"],
        "Delta_GT_ci95": boot_gt["ci95"],
        "decomp_A": A,
        "decomp_B": Bb,
        "decomp_C": C,
        "decomp_A_ci95": boot_A["ci95"],
        "decomp_B_ci95": boot_B["ci95"],
        "decomp_C_ci95": boot_C["ci95"],
        "decomp_residual": residual,
        "n_strict_reverse": n_strict,
        "frac_strict_reverse": n_strict / len(rows),
        "strict_reverse_ci95": boot_strict["ci95"],
        "n_tie_change": n_tie,
        "tie_change_ci95": boot_tie["ci95"],
        "per_query": rows,
    }


def aggregate_regen(
    baseline: Sequence[Dict[str, Any]],
    round_scores: Sequence[Dict[str, Any]],
    *,
    B: int = 10000,
    seed: int = 20261008,
) -> Dict[str, Any]:
    bmap = {r["sample_id"]: r for r in baseline if r.get("scorable_pair")}
    rmap = {r["sample_id"]: r for r in round_scores if r.get("scorable_pair")}
    ids = sorted(set(bmap) & set(rmap))
    ws, dws, clusters = [], [], []
    per = []
    for sid in ids:
        b, r = bmap[sid], rmap[sid]
        dw = float(r["w"]) - float(b["w"])
        ws.append(float(r["w"]))
        dws.append(dw)
        clusters.append(str(r.get("cluster_id") or r.get("video_id")))
        per.append(
            {
                "sample_id": sid,
                "w_round": float(r["w"]),
                "w_baseline": float(b["w"]),
                "delta_w": dw,
                "d_ov_round": float(r["d_ov"]),
                "d_GT_round": float(r["d_GT"]),
                "strict_reverse_round": r["strict_reverse"],
                "strict_reverse_baseline": b["strict_reverse"],
            }
        )
    bw = boot_from_pairs(ws, clusters, B=B, seed=seed)
    bdw = boot_from_pairs(dws, clusters, B=B, seed=seed)
    return {
        "n": len(ids),
        "W_mean": bw["mean"],
        "W_ci95": bw["ci95"],
        "delta_w_vs_baseline_mean": bdw["mean"],
        "delta_w_ci95": bdw["ci95"],
        "per_query": per,
    }
