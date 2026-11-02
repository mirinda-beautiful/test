#!/usr/bin/env python3
"""Independent CPU protocol audit entrypoint.

Example:
  python audit_protocol.py \\
    --input data/raw_predictions.jsonl \\
    --config config/stat_config.json \\
    --out outputs/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "stvg_protocol_toolkit"))

from protocol_toolkit.aggregate import aggregate_group, aggregate_regen, score_query  # noqa: E402


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Force LF line endings on all platforms so SHA256SUMS stay portable.
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser(description="STVG protocol audit (CPU, from boxes/times)")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-iou-recompute", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    B = int(cfg["bootstrap_B"])
    seed = int(cfg["bootstrap_seed"])
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    raw = load_jsonl(args.input)
    recompute = bool(cfg.get("recompute_iou_from_boxes", True)) and not args.skip_iou_recompute

    # Split exclusions vs scorable inputs
    exclusions = [r for r in raw if r.get("exclusion_reason") == "no_gt_candidate_slots"]
    usable = [r for r in raw if r.get("exclusion_reason") != "no_gt_candidate_slots"]

    scored: List[Dict[str, Any]] = []
    gaps: List[Dict[str, Any]] = []
    for rec in usable:
        if rec.get("missing_arms"):
            gaps.append({"sample_id": rec["sample_id"], "round": rec.get("round"), "missing_arms": rec["missing_arms"]})
        # skip if S/T evidence missing
        arms = rec.get("arms") or {}
        if arms.get("S", {}).get("missing_evidence") or arms.get("T", {}).get("missing_evidence"):
            gaps.append({"sample_id": rec["sample_id"], "round": rec.get("round"), "gap": "missing_evidence"})
            continue
        s = score_query(rec, recompute_iou=recompute, ranking_eps=float(cfg["ranking_eps"]))
        if not s.get("scorable_pair"):
            gaps.append({"sample_id": rec.get("sample_id"), "round": rec.get("round"), "reason": s.get("reason")})
            continue
        scored.append(s)

    # Main four groups
    summary_rows = []
    per_rows = []
    decomp_rows = []
    rank_rows = []
    for group in cfg["groups_main"]:
        sub = [r for r in scored if r.get("group") == group and r.get("round") == "baseline"]
        # cluster key
        for r in sub:
            if r.get("dataset") == "HC-STVG":
                r["cluster_id"] = r.get("source_video_id") or r.get("cluster_id")
            else:
                r["cluster_id"] = r.get("video_id") or r.get("cluster_id")
        agg = aggregate_group(sub, B=B, seed=seed)
        summary_rows.append(
            {
                "group": group,
                "n": agg.get("n"),
                "n_clusters": agg.get("n_clusters"),
                "Delta_ov_ci95_lo": agg["Delta_ov_ci95"][0],
                "Delta_ov_ci95_hi": agg["Delta_ov_ci95"][1],
                "Delta_GT_ci95_lo": agg["Delta_GT_ci95"][0],
                "Delta_GT_ci95_hi": agg["Delta_GT_ci95"][1],
                **{f"decomp_{k}_ci95_{side}": agg[f"decomp_{k}_ci95"][i]
                   for k in ("A", "B", "C") for i, side in enumerate(("lo", "hi"))},
                "strict_reverse_ci95_lo": agg["strict_reverse_ci95"][0],
                "strict_reverse_ci95_hi": agg["strict_reverse_ci95"][1],
                "tie_change_ci95_lo": agg["tie_change_ci95"][0],
                "tie_change_ci95_hi": agg["tie_change_ci95"][1],
                "W_mean": agg.get("W_mean"),
                "W_ci95_lo": (agg.get("W_ci95") or [None, None])[0],
                "W_ci95_hi": (agg.get("W_ci95") or [None, None])[1],
                "Delta_ov_mean": agg.get("Delta_ov_mean"),
                "Delta_GT_mean": agg.get("Delta_GT_mean"),
                "decomp_A": agg.get("decomp_A"),
                "decomp_B": agg.get("decomp_B"),
                "decomp_C": agg.get("decomp_C"),
                "decomp_residual": agg.get("decomp_residual"),
                "n_strict_reverse": agg.get("n_strict_reverse"),
                "frac_strict_reverse": agg.get("frac_strict_reverse"),
                "n_tie_change": agg.get("n_tie_change"),
            }
        )
        for r in agg.get("per_query") or []:
            per_rows.append(
                {
                    "group": group,
                    "sample_id": r["sample_id"],
                    "video_id": r.get("video_id"),
                    "cluster_id": r.get("cluster_id"),
                    "s_ov_S": r["s_ov_S"],
                    "s_ov_T": r["s_ov_T"],
                    "s_GT_S": r["s_GT_S"],
                    "s_GT_T": r["s_GT_T"],
                    "d_ov": r["d_ov"],
                    "d_GT": r["d_GT"],
                    "w": r["w"],
                    "g_S": r["g_S"],
                    "g_T": r["g_T"],
                    "mech_class_S": r["mech_class_S"],
                    "mech_class_T": r["mech_class_T"],
                    "decomp_A": r["decomp_A"],
                    "decomp_B": r["decomp_B"],
                    "decomp_C": r["decomp_C"],
                    "rank_category": r["rank_category"],
                    "strict_reverse": r["strict_reverse"],
                    "orig_w": r.get("orig_w"),
                }
            )
            decomp_rows.append(
                {
                    "group": group,
                    "sample_id": r["sample_id"],
                    "decomp_A": r["decomp_A"],
                    "decomp_B": r["decomp_B"],
                    "decomp_C": r["decomp_C"],
                    "sum_ABC": r["decomp_A"] + r["decomp_B"] + r["decomp_C"],
                    "w": r["w"],
                    "residual": (r["decomp_A"] + r["decomp_B"] + r["decomp_C"]) - r["w"],
                }
            )
            rank_rows.append(
                {
                    "group": group,
                    "sample_id": r["sample_id"],
                    "d_ov": r["d_ov"],
                    "d_GT": r["d_GT"],
                    "rank_category": r["rank_category"],
                    "strict_reverse": r["strict_reverse"],
                }
            )

    # Regen 32 DeViL frozen
    regen_group = cfg["regen_group"]
    base = [r for r in scored if r.get("group") == regen_group and r.get("round") == "baseline"]
    regen_summary = []
    regen_per = []
    for rnd in ("R1", "R2"):
        cur = [r for r in scored if r.get("group") == regen_group and r.get("round") == rnd]
        for r in base + cur:
            r["cluster_id"] = r.get("video_id") or r.get("cluster_id")
        agg = aggregate_regen(base, cur, B=B, seed=seed)
        regen_summary.append(
            {
                "model": "DeViL",
                "round": rnd,
                "n": agg["n"],
                "W_mean": agg["W_mean"],
                "W_ci95_lo": agg["W_ci95"][0],
                "W_ci95_hi": agg["W_ci95"][1],
                "delta_w_vs_baseline_mean": agg["delta_w_vs_baseline_mean"],
                "delta_w_ci95_lo": agg["delta_w_ci95"][0],
                "delta_w_ci95_hi": agg["delta_w_ci95"][1],
            }
        )
        for p in agg["per_query"]:
            p = dict(p)
            p["round"] = rnd
            regen_per.append(p)
    # baseline W for regen set
    base_agg = aggregate_group(base, B=B, seed=seed)
    regen_summary.insert(
        0,
        {
            "model": "DeViL",
            "round": "baseline",
            "n": base_agg.get("n"),
            "W_mean": base_agg.get("W_mean"),
            "W_ci95_lo": (base_agg.get("W_ci95") or [None, None])[0],
            "W_ci95_hi": (base_agg.get("W_ci95") or [None, None])[1],
            "delta_w_vs_baseline_mean": 0.0,
            "delta_w_ci95_lo": 0.0,
            "delta_w_ci95_hi": 0.0,
        },
    )

    # Write outputs
    write_csv(
        out / "frozen_protocol_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()) if summary_rows else ["group"],
    )
    write_csv(
        out / "frozen_per_query.csv",
        per_rows,
        list(per_rows[0].keys()) if per_rows else ["group"],
    )
    write_csv(
        out / "decomposition_per_query.csv",
        decomp_rows,
        list(decomp_rows[0].keys()) if decomp_rows else ["group"],
    )
    write_csv(
        out / "decomposition_summary.csv",
        [
            {
                "group": s["group"],
                "A": s["decomp_A"],
                "B": s["decomp_B"],
                "C": s["decomp_C"],
                "residual": s["decomp_residual"],
                **{f"{k}_ci95_{side}": s[f"decomp_{k}_ci95_{side}"]
                   for k in ("A", "B", "C") for side in ("lo", "hi")},
            }
            for s in summary_rows
        ],
        ["group", "A", "B", "C", "residual"] + [f"{k}_ci95_{side}" for k in ("A", "B", "C") for side in ("lo", "hi")],
    )
    write_csv(
        out / "ranking_per_query.csv",
        rank_rows,
        list(rank_rows[0].keys()) if rank_rows else ["group"],
    )
    write_csv(
        out / "ranking_summary.csv",
        [
            {
                "group": s["group"],
                "n_strict_reverse": s["n_strict_reverse"],
                "n": s["n"],
                "frac_strict_reverse": s["frac_strict_reverse"],
                "n_tie_change": s["n_tie_change"],
                "strict_ci95_lo": s["strict_reverse_ci95_lo"],
                "strict_ci95_hi": s["strict_reverse_ci95_hi"],
                "tie_ci95_lo": s["tie_change_ci95_lo"],
                "tie_ci95_hi": s["tie_change_ci95_hi"],
            }
            for s in summary_rows
        ],
        ["group", "n_strict_reverse", "n", "frac_strict_reverse", "n_tie_change", "strict_ci95_lo", "strict_ci95_hi", "tie_ci95_lo", "tie_ci95_hi"],
    )
    write_csv(
        out / "regeneration_summary.csv",
        regen_summary,
        list(regen_summary[0].keys()) if regen_summary else ["round"],
    )
    write_csv(
        out / "regeneration_per_query.csv",
        regen_per,
        list(regen_per[0].keys()) if regen_per else ["sample_id"],
    )
    excl_out = []
    for e in exclusions:
        excl_out.append(
            {
                "sample_id": e["sample_id"],
                "group": e.get("group"),
                "exclusion_reason": e.get("exclusion_reason"),
                "n_gt_candidate_slots": e.get("n_gt_candidate_slots", 0),
                "note": e.get("exclusion_note"),
            }
        )
    write_csv(
        out / "exclusion_no_gt_candidates.csv",
        excl_out,
        ["sample_id", "group", "exclusion_reason", "n_gt_candidate_slots", "note"],
    )
    (out / "gaps.json").write_text(json.dumps(gaps, indent=2) + "\n", encoding="utf-8")
    (out / "run_summary.json").write_text(
        json.dumps(
            {
                "n_raw": len(raw),
                "n_exclusions_no_gt": len(exclusions),
                "n_scored": len(scored),
                "n_gaps": len(gaps),
                "groups": summary_rows,
                "regen": regen_summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(out), "n_scored": len(scored), "n_excl": len(exclusions), "n_gaps": len(gaps)}, indent=2))
    for s in summary_rows:
        print(s["group"], "n", s["n"], "W", s["W_mean"], "strict", s["n_strict_reverse"])
    for s in regen_summary:
        print("regen", s["round"], "W", s["W_mean"], "dw", s["delta_w_vs_baseline_mean"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
