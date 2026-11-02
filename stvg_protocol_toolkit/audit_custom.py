#!/usr/bin/env python3
"""Custom STVG prediction audit (user-exported frozen boxes/times).

Example:
  python audit_custom.py \\
    --input examples/custom_predictions.jsonl \\
    --config config/custom_example.json \\
    --out outputs_custom_demo/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "stvg_protocol_toolkit"))

from protocol_toolkit.aggregate import aggregate_group, aggregate_regen, score_query  # noqa: E402
from protocol_toolkit.custom_validate import validate_custom_records  # noqa: E402


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _resolve_groups(cfg: Dict[str, Any], records: List[Dict[str, Any]]) -> List[str]:
    if cfg.get("groups"):
        return list(cfg["groups"])
    return sorted({r.get("group") for r in records if r.get("group")})


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Custom STVG protocol audit from user predictions")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-iou-recompute", action="store_true")
    args = ap.parse_args(argv)

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    B = int(cfg.get("bootstrap_B", 10000))
    seed = int(cfg.get("bootstrap_seed", 20261008))
    ranking_eps = float(cfg.get("ranking_eps", 1e-8))
    cluster_field = str(cfg.get("cluster_field", "video_id"))
    require_clean = bool(cfg.get("require_clean", False))
    enable_regen = bool(cfg.get("enable_regeneration", False))
    # Explicit policy only — never infer from model name in custom mode.
    empty_fixed_as_zero = bool(cfg.get("empty_fixed_as_zero", False))
    recompute = bool(cfg.get("recompute_iou_from_boxes", True)) and not args.skip_iou_recompute
    fail_on_input_error = bool(cfg.get("fail_on_input_error", True))

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    raw = load_jsonl(args.input)
    validation = validate_custom_records(
        raw,
        cluster_field=cluster_field,
        require_clean=require_clean,
    )
    (out / "input_validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    if fail_on_input_error and not validation["ok"]:
        print(json.dumps({"ok": False, "stage": "input_validation", "n_errors": validation["n_errors"]}, indent=2))
        for e in validation["errors"][:20]:
            print("ERROR", e)
        return 2

    groups = _resolve_groups(cfg, raw)
    exclusions = [r for r in raw if r.get("exclusion_reason") == "no_gt_candidate_slots"]
    usable = [r for r in raw if r.get("exclusion_reason") != "no_gt_candidate_slots"]

    scored: List[Dict[str, Any]] = []
    gaps: List[Dict[str, Any]] = []
    for rec in usable:
        arms = rec.get("arms") or {}
        if arms.get("S", {}).get("missing_evidence") or arms.get("T", {}).get("missing_evidence"):
            gaps.append(
                {
                    "sample_id": rec.get("sample_id"),
                    "round": rec.get("round"),
                    "group": rec.get("group"),
                    "gap": "missing_evidence",
                    "note": "Missing temporal/spatial evidence is recorded as a gap, not empty-time class A.",
                }
            )
            continue
        s = score_query(
            rec,
            recompute_iou=recompute,
            ranking_eps=ranking_eps,
            empty_fixed_as_zero=empty_fixed_as_zero,
        )
        if not s.get("scorable_pair"):
            gaps.append(
                {
                    "sample_id": rec.get("sample_id"),
                    "round": rec.get("round"),
                    "group": rec.get("group"),
                    "reason": s.get("reason"),
                }
            )
            continue
        # Attach cluster id from configured field.
        s["cluster_id"] = rec.get(cluster_field) or rec.get("cluster_id") or rec.get("video_id")
        scored.append(s)

    summary_rows: List[Dict[str, Any]] = []
    per_rows: List[Dict[str, Any]] = []
    decomp_rows: List[Dict[str, Any]] = []
    rank_rows: List[Dict[str, Any]] = []

    for group in groups:
        sub = [r for r in scored if r.get("group") == group and r.get("round", "baseline") == "baseline"]
        agg = aggregate_group(sub, B=B, seed=seed)
        if not agg.get("n"):
            summary_rows.append({"group": group, "n": 0, "n_clusters": 0})
            continue
        summary_rows.append(
            {
                "group": group,
                "n": agg.get("n"),
                "n_clusters": agg.get("n_clusters"),
                "Delta_ov_mean": agg.get("Delta_ov_mean"),
                "Delta_ov_ci95_lo": agg["Delta_ov_ci95"][0],
                "Delta_ov_ci95_hi": agg["Delta_ov_ci95"][1],
                "Delta_GT_mean": agg.get("Delta_GT_mean"),
                "Delta_GT_ci95_lo": agg["Delta_GT_ci95"][0],
                "Delta_GT_ci95_hi": agg["Delta_GT_ci95"][1],
                "W_mean": agg.get("W_mean"),
                "W_ci95_lo": agg["W_ci95"][0],
                "W_ci95_hi": agg["W_ci95"][1],
                "decomp_A": agg.get("decomp_A"),
                "decomp_B": agg.get("decomp_B"),
                "decomp_C": agg.get("decomp_C"),
                "decomp_residual": agg.get("decomp_residual"),
                "decomp_A_ci95_lo": agg["decomp_A_ci95"][0],
                "decomp_A_ci95_hi": agg["decomp_A_ci95"][1],
                "decomp_B_ci95_lo": agg["decomp_B_ci95"][0],
                "decomp_B_ci95_hi": agg["decomp_B_ci95"][1],
                "decomp_C_ci95_lo": agg["decomp_C_ci95"][0],
                "decomp_C_ci95_hi": agg["decomp_C_ci95"][1],
                "n_strict_reverse": agg.get("n_strict_reverse"),
                "frac_strict_reverse": agg.get("frac_strict_reverse"),
                "strict_reverse_ci95_lo": agg["strict_reverse_ci95"][0],
                "strict_reverse_ci95_hi": agg["strict_reverse_ci95"][1],
                "n_tie_change": agg.get("n_tie_change"),
                "tie_change_ci95_lo": agg["tie_change_ci95"][0],
                "tie_change_ci95_hi": agg["tie_change_ci95"][1],
            }
        )
        for r in agg.get("per_query") or []:
            per_rows.append(
                {
                    "group": group,
                    "sample_id": r["sample_id"],
                    "video_id": r.get("video_id"),
                    "source_video_id": r.get("source_video_id"),
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

    write_csv(
        out / "protocol_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()) if summary_rows else ["group"],
    )
    write_csv(out / "per_query.csv", per_rows, list(per_rows[0].keys()) if per_rows else ["group"])
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
                "A": s.get("decomp_A"),
                "B": s.get("decomp_B"),
                "C": s.get("decomp_C"),
                "residual": s.get("decomp_residual"),
                "A_ci95_lo": s.get("decomp_A_ci95_lo"),
                "A_ci95_hi": s.get("decomp_A_ci95_hi"),
                "B_ci95_lo": s.get("decomp_B_ci95_lo"),
                "B_ci95_hi": s.get("decomp_B_ci95_hi"),
                "C_ci95_lo": s.get("decomp_C_ci95_lo"),
                "C_ci95_hi": s.get("decomp_C_ci95_hi"),
            }
            for s in summary_rows
            if s.get("n")
        ],
        [
            "group",
            "A",
            "B",
            "C",
            "residual",
            "A_ci95_lo",
            "A_ci95_hi",
            "B_ci95_lo",
            "B_ci95_hi",
            "C_ci95_lo",
            "C_ci95_hi",
        ],
    )
    write_csv(out / "ranking_per_query.csv", rank_rows, list(rank_rows[0].keys()) if rank_rows else ["group"])
    write_csv(
        out / "ranking_summary.csv",
        [
            {
                "group": s["group"],
                "n_strict_reverse": s.get("n_strict_reverse"),
                "n": s.get("n"),
                "frac_strict_reverse": s.get("frac_strict_reverse"),
                "n_tie_change": s.get("n_tie_change"),
                "strict_ci95_lo": s.get("strict_reverse_ci95_lo"),
                "strict_ci95_hi": s.get("strict_reverse_ci95_hi"),
                "tie_ci95_lo": s.get("tie_change_ci95_lo"),
                "tie_ci95_hi": s.get("tie_change_ci95_hi"),
            }
            for s in summary_rows
            if s.get("n")
        ],
        [
            "group",
            "n_strict_reverse",
            "n",
            "frac_strict_reverse",
            "n_tie_change",
            "strict_ci95_lo",
            "strict_ci95_hi",
            "tie_ci95_lo",
            "tie_ci95_hi",
        ],
    )

    # Coverage / exclusions report
    cov_rows = []
    for row in validation.get("coverage_rows") or []:
        cov_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "round": row.get("round"),
                "group": row.get("group"),
                "status": row.get("status"),
                "reason": row.get("reason"),
                "n_gt_candidate_slots": row.get("n_gt_candidate_slots"),
                "n_valid_pairs_S": row.get("n_valid_pairs_S"),
                "n_valid_pairs_T": row.get("n_valid_pairs_T"),
            }
        )
    for g in gaps:
        cov_rows.append(
            {
                "sample_id": g.get("sample_id"),
                "round": g.get("round"),
                "group": g.get("group"),
                "status": "gap",
                "reason": g.get("reason") or g.get("gap"),
                "n_gt_candidate_slots": "",
                "n_valid_pairs_S": "",
                "n_valid_pairs_T": "",
            }
        )
    for e in exclusions:
        # already in coverage as excluded; keep explicit row if not present
        pass
    write_csv(
        out / "coverage_and_exclusions.csv",
        cov_rows,
        [
            "sample_id",
            "round",
            "group",
            "status",
            "reason",
            "n_gt_candidate_slots",
            "n_valid_pairs_S",
            "n_valid_pairs_T",
        ],
    )

    regen_summary: List[Dict[str, Any]] = []
    if enable_regen:
        regen_group = cfg.get("regen_group")
        if not regen_group:
            print(json.dumps({"ok": False, "error": "enable_regeneration true but regen_group missing"}, indent=2))
            return 2
        base = [r for r in scored if r.get("group") == regen_group and r.get("round") == "baseline"]
        for rnd in cfg.get("regen_rounds") or ["R1", "R2"]:
            cur = [r for r in scored if r.get("group") == regen_group and r.get("round") == rnd]
            agg = aggregate_regen(base, cur, B=B, seed=seed)
            regen_summary.append(
                {
                    "group": regen_group,
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
        write_csv(
            out / "regeneration_summary.csv",
            regen_summary,
            list(regen_summary[0].keys()) if regen_summary else ["round"],
        )

    (out / "gaps.json").write_text(json.dumps(gaps, indent=2) + "\n", encoding="utf-8")
    (out / "run_summary.json").write_text(
        json.dumps(
            {
                "mode": "custom",
                "n_raw": len(raw),
                "n_exclusions_no_gt": len(exclusions),
                "n_scored": len(scored),
                "n_gaps": len(gaps),
                "groups": groups,
                "cluster_field": cluster_field,
                "empty_fixed_as_zero": empty_fixed_as_zero,
                "enable_regeneration": enable_regen,
                "bootstrap_B": B,
                "bootstrap_seed": seed,
                "input_validation_ok": validation["ok"],
                "summary": summary_rows,
                "regen": regen_summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "out": str(out),
                "n_scored": len(scored),
                "n_excl": len(exclusions),
                "n_gaps": len(gaps),
                "groups": groups,
                "input_validation_ok": validation["ok"],
            },
            indent=2,
        )
    )
    for s in summary_rows:
        print(s.get("group"), "n", s.get("n"), "W", s.get("W_mean"), "strict", s.get("n_strict_reverse"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
