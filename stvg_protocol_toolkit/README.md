# STVG Protocol Toolkit v1.1

CPU-only toolkit to **recompute and audit frozen STVG protocol scores** from archived
prediction boxes and temporal evidence. It does **not** run models, load weights,
or read raw videos.

## Coverage (verified)

Four frozen main evaluations:

| Group | Dataset | Model | Common queries | Strict ranking flips |
|---|---|---|---:|---:|
| VidSTG_TubeDETR | VidSTG | TubeDETR | 200 | 18/200 |
| VidSTG_DeViL | VidSTG | DeViL | 186 | 40/186 |
| HC_TubeDETR | HC-STVG | TubeDETR | 100 | 26/100 |
| HC_DeViL | HC-STVG | DeViL | 100 | 36/100 |

Plus **DeViL 32-query** perturbation regeneration comparison (baseline / R1 / R2).

Acceptance on a clean checkout of this package:

- `scripts/validate_outputs.py` → **149/149** (means + CI endpoints vs `data/reference/`)
- `stvg_protocol_toolkit/tests/test_edge_cases.py` → **9/9**
- `scripts/test_validate_negative_controls.py` → must fail on corrupted/missing CI (temp copies only)

## Requirements

- Python **3.9+** (verified with 3.12)
- `numpy` (see `requirements.txt`)

No GPU, CUDA, or third-party vision stacks are required.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Input format (`data/raw_predictions.jsonl`)

One JSON object per line (query × round). Fields used for recomputation:

- Identifiers: `group`, `dataset`, `model`, `sample_id`, `video_id`, `source_video_id`, `cluster_id`, `round`
- Exclusion: `exclusion_reason` (`no_gt_candidate_slots` for DeViL queries with no GT candidate slots; **not** mechanism class A)
- `arms.Clean|S|T`:
  - `pred_temporal_frames`, `pred_time_sec`, `gt_temporal`
  - `pair_rows[]` with `abs_frame_id`, `gt_box_xywh`, `pred_box_xywh` (absolute-pixel xywh)
  - `sampling` (frame map / fps / slots), `evidence_sha256`, `score_version`
- Optional provenance: `orig_*` / `orig_record_scores` (original run scores; not used for frozen recomputation)

## Run from frozen boxes and times

From this directory (package root):

```bash
python audit_protocol.py \
  --input data/raw_predictions.jsonl \
  --config config/stat_config.json \
  --out outputs/
```

Then verify:

```bash
python scripts/validate_outputs.py
python stvg_protocol_toolkit/tests/test_edge_cases.py
python scripts/test_validate_negative_controls.py
```

`validate_outputs.py` regenerates nothing; run `audit_protocol.py` first so `outputs/`
matches a fresh recomputation. Reference tables under `data/reference/` and
`data/regen32_frozen_reference.json` are used only for acceptance comparison.

## Outputs ↔ paper / closeout tables

| File | Content |
|---|---|
| `outputs/frozen_protocol_summary.csv` | Four-group frozen W, Δ_ov, Δ_GT, CIs |
| `outputs/frozen_per_query.csv` | Per-query frozen scores and mechanism labels |
| `outputs/decomposition_summary.csv` / `decomposition_per_query.csv` | Signed A/B/C contributions |
| `outputs/ranking_summary.csv` / `ranking_per_query.csv` | Strict flips and tie changes |
| `outputs/regeneration_summary.csv` / `regeneration_per_query.csv` | DeViL-32 frozen W and Δw |
| `outputs/exclusion_no_gt_candidates.csv` | 14 DeViL exclusions (`no_gt_candidate_slots`) |
| `outputs/gaps.json` | Scoring gaps (must be `[]` for this package) |
| `validation.json` | Machine-readable acceptance report |

Frozen W definition (see `config/stat_config.json`):

`W = mean(d_GT − d_ov)` with `d = s_T − s_S`, `g = s_GT − s_ov`, `w = g_T − g_S`.

Bootstrap: B=10000, seed=20261008; VidSTG clusters by `video_id`, HC-STVG by `source_video_id`; ranking ε=1e-8.

## Frozen scores vs original-run scores

- **Frozen scores** are recomputed here from boxes + temporal intervals with a single
  explicit mechanism definition (`explicit_gt_interval_v1`).
- **Original-run scores** (`orig_*`) are retained for provenance only. Small
  `W_abs_diff_frozen_minus_orig` values in reference provenance tables reflect
  historical scorer differences, not a second protocol.

## Mechanism labels

- **A**: empty GT∩pred temporal intersection → overlap contribution zeroed
- **B**: nonempty time intersection but no sampled support in the intersection
- **C / C_full**: nonempty selected frames (both roll into aggregate C)
- **M**: sampled support present but no valid box pair → gap (not counted as A/B)
- **`exclusion_reason=no_gt_candidate_slots`**: query-level exclusion (DeViL 14); **do not conflate with A**

## DeViL overlap-box archival note

DeViL **original overlap full boxes were not archived**. Frozen diagnostics in this
toolkit use the **fixed-chain boxes** present in `pair_rows`. Regeneration (32-query)
comparisons are likewise frozen-chain recomputations of archived rounds.

## Layout

```
stvg_protocol_toolkit/
  audit_protocol.py
  config/stat_config.json
  data/raw_predictions.jsonl
  data/reference/…
  data/regen32_frozen_reference.json
  scripts/validate_outputs.py
  stvg_protocol_toolkit/protocol_toolkit/   # score, iou, aggregate, bootstrap
  stvg_protocol_toolkit/tests/test_edge_cases.py
  outputs/                                 # verified reference outputs
  validation.json
  README.md  requirements.txt  LICENSE  THIRD_PARTY_NOTICES.md
  FILE_MANIFEST.csv  SHA256SUMS.txt
```

## License

Toolkit code: MIT (see `LICENSE`). Third-party notices: `THIRD_PARTY_NOTICES.md`.
