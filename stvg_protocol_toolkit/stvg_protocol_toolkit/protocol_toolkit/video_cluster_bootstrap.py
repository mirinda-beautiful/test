#!/usr/bin/env python3
"""Portable video-cluster bootstrap (query-equal mean).

No project imports. Requires numpy.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union


def video_cluster_bootstrap_means(
    rows: Sequence[Dict[str, Any]],
    *,
    key: str = "x",
    video_key: str = "video_id",
    B: int = 10000,
    seed: int = 20261008,
) -> Dict[str, Any]:
    """Cluster bootstrap over videos; within-video queries kept when a video is drawn.

    Mean is query-equal over the flattened resampled queries.
    """
    import numpy as np

    by: Dict[str, List[float]] = {}
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        by.setdefault(str(r[video_key]), []).append(float(v))
    vids = sorted(by)
    if not vids:
        return {
            "n_query": 0,
            "n_video": 0,
            "mean": None,
            "ci95": [None, None],
            "crosses_zero": None,
            "B": B,
            "seed": seed,
        }
    flat = [x for vs in by.values() for x in vs]
    mean = float(np.mean(flat))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        chosen = rng.choice(vids, size=len(vids), replace=True)
        vals = [x for v in chosen for x in by[v]]
        boots.append(float(np.mean(vals)))
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    return {
        "n_query": len(flat),
        "n_video": len(vids),
        "mean": mean,
        "ci95": [lo, hi],
        "crosses_zero": bool(lo <= 0.0 <= hi),
        "B": B,
        "seed": seed,
    }


def boot_from_pairs(
    values: Sequence[float],
    video_ids: Sequence[str],
    *,
    B: int = 10000,
    seed: int = 20261008,
) -> Dict[str, Any]:
    rows = [{"x": float(x), "video_id": str(v)} for x, v in zip(values, video_ids)]
    return video_cluster_bootstrap_means(rows, key="x", video_key="video_id", B=B, seed=seed)
