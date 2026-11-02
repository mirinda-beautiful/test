#!/usr/bin/env python3
"""Known-answer and plumbing tests for custom prediction audits.

Reported separately from the paper acceptance suite (149/149 + 9/9).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stvg_protocol_toolkit"))

from protocol_toolkit.aggregate import score_query  # noqa: E402
from protocol_toolkit.custom_validate import validate_custom_records  # noqa: E402


def _load_demo():
    path = ROOT / "examples" / "custom_predictions.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return {r["sample_id"]: r for r in rows}


def test_known_answers():
    demo = _load_demo()
    n1 = score_query(demo["demo_N1"], empty_fixed_as_zero=False)
    assert n1["scorable_pair"]
    assert n1["mech_class_S"] == "C"
    assert n1["mech_class_T"] == "C_full"
    assert abs(n1["d_ov"] - 1.0) < 1e-12
    assert abs(n1["d_GT"] - 0.4) < 1e-12
    assert abs(n1["w"] - (-0.6)) < 1e-12

    a1 = score_query(demo["demo_A1"], empty_fixed_as_zero=False)
    assert a1["scorable_pair"] and a1["mech_class_S"] == "A" and a1["mech_class_T"] == "A"

    b1 = score_query(demo["demo_B1"], empty_fixed_as_zero=False)
    assert b1["scorable_pair"] and b1["mech_class_S"] == "B" and b1["mech_class_T"] == "B"

    z1 = score_query(demo["demo_Z1"], empty_fixed_as_zero=False)
    assert z1["scorable_pair"] and z1["mech_class_S"] == "C" and z1["s_ov_S"] == 0.0

    m1 = score_query(demo["demo_M1"], empty_fixed_as_zero=False)
    assert not m1["scorable_pair"] and m1["reason"] == "mechanism_M_missing_box_pairs"

    assert demo["demo_EX1"].get("exclusion_reason") == "no_gt_candidate_slots"


def test_validate_and_cluster_config():
    demo = list(_load_demo().values())
    ok = validate_custom_records(demo, cluster_field="source_video_id")
    assert ok["ok"], ok["errors"]

    bad = json.loads(json.dumps(demo[0]))
    del bad["video_id"]
    bad_report = validate_custom_records([bad], cluster_field="source_video_id")
    assert not bad_report["ok"]
    assert any("video_id" in e["message"] for e in bad_report["errors"])

def test_cli_demo_and_flags():
    with tempfile.TemporaryDirectory(prefix="stvg_custom_") as td:
        out = Path(td) / "out"
        cfg = json.loads((ROOT / "config" / "custom_example.json").read_text(encoding="utf-8"))
        cfg_path = Path(td) / "cfg.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "audit_custom.py"),
                "--input",
                str(ROOT / "examples" / "custom_predictions.jsonl"),
                "--config",
                str(cfg_path),
                "--out",
                str(out),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (out / "protocol_summary.csv").exists()
        assert (out / "per_query.csv").exists()
        assert (out / "decomposition_summary.csv").exists()
        assert (out / "ranking_summary.csv").exists()
        assert (out / "ranking_per_query.csv").exists()
        assert (out / "coverage_and_exclusions.csv").exists()
        assert (out / "input_validation.json").exists()
        summary = (out / "run_summary.json").read_text(encoding="utf-8")
        assert '"enable_regeneration": false' in summary
        assert "Demo_CustomModel" in summary
        # no regeneration file when disabled
        assert not (out / "regeneration_summary.csv").exists()

        # missing field should fail closed
        bad_in = Path(td) / "bad.jsonl"
        rows = []
        with (ROOT / "examples" / "custom_predictions.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        del rows[0]["video_id"]
        bad_in.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
        proc2 = subprocess.run(
            [
                sys.executable,
                str(ROOT / "audit_custom.py"),
                "--input",
                str(bad_in),
                "--config",
                str(cfg_path),
                "--out",
                str(Path(td) / "out_bad"),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        assert proc2.returncode == 2


def main() -> int:
    test_known_answers()
    test_validate_and_cluster_config()
    test_cli_demo_and_flags()
    print(
        {
            "ok": True,
            "suite": "custom_audit",
            "cases": [
                "known_N1_W",
                "mech_A",
                "mech_B",
                "legal_zero_iou",
                "missing_boxes_M",
                "exclusion_no_gt",
                "cli_demo_outputs",
                "regen_off",
                "custom_group_name",
                "missing_field_errors",
            ],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
