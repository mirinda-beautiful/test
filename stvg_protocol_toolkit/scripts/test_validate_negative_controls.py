#!/usr/bin/env python3
"""Negative controls for validate_outputs.py (temp copies only).

1) Corrupt VidSTG_TubeDETR W CI to [0.1200, 0.1400] → must fail (nonzero).
2) Drop a CI endpoint → must fail (nonzero).
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATE = ROOT / "scripts" / "validate_outputs.py"


def _copy_outputs(dst: Path) -> None:
    shutil.copytree(ROOT / "outputs", dst)


def _rewrite_summary(path: Path, mutator) -> None:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())
    for row in rows:
        mutator(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _run_validate(out_dir: Path, report: Path) -> int:
    proc = subprocess.run(
        [
            sys.executable,
            str(VALIDATE),
            "--root",
            str(ROOT),
            "--out",
            str(out_dir),
            "--write-report",
            str(report),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def main() -> int:
    results = []
    with tempfile.TemporaryDirectory(prefix="stvg_negctrl_") as td:
        td_path = Path(td)

        # Control 1: wrong W CI endpoints for VidSTG_TubeDETR
        out1 = td_path / "out_wrong_w_ci"
        _copy_outputs(out1)

        def corrupt_w(row):
            if row["group"] == "VidSTG_TubeDETR":
                row["W_ci95_lo"] = "0.1200"
                row["W_ci95_hi"] = "0.1400"

        _rewrite_summary(out1 / "frozen_protocol_summary.csv", corrupt_w)
        rc1, so1, se1 = _run_validate(out1, td_path / "report_wrong_w.json")
        ok1 = rc1 != 0
        results.append({"id": "wrong_W_ci_VidSTG_TubeDETR", "ok": ok1, "returncode": rc1})
        if not ok1:
            print("FAIL expected nonzero for wrong W CI")
            print(so1)
            print(se1)

        # Control 2: missing CI endpoint
        out2 = td_path / "out_missing_ci"
        _copy_outputs(out2)

        def drop_endpoint(row):
            if row["group"] == "VidSTG_TubeDETR":
                row["W_ci95_lo"] = ""

        _rewrite_summary(out2 / "frozen_protocol_summary.csv", drop_endpoint)
        rc2, so2, se2 = _run_validate(out2, td_path / "report_missing.json")
        ok2 = rc2 != 0
        results.append({"id": "missing_W_ci_endpoint", "ok": ok2, "returncode": rc2})
        if not ok2:
            print("FAIL expected nonzero for missing CI endpoint")
            print(so2)
            print(se2)

    all_ok = all(r["ok"] for r in results)
    print({"ok": all_ok, "controls": results})
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
