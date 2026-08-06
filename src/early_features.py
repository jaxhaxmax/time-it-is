"""Early-regime feature extraction over the clean-building list, writes early_features.csv. these four features capture the degenerate early regime (meter offline, vacancy, near-constant load) that poisons the head when 'full' or 'first_15' fine-tuning sees it: early_flat_frac (fraction of first-30% hours in near-constant runs), early_zero_frac (fraction of first-30% hours at or below 1% of the overall median), early_late_mean_ratio (mean of first 30% over mean of last 30%), early_late_std_logratio (log of the same ratio on std). they join the 35 profiler features to make the 39 the router uses."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

TEST_HOURS = 672


def early_regime_features(x: np.ndarray, test_hours: int = TEST_HOURS) -> dict:
    pre = np.asarray(x[: len(x) - test_hours], float)
    n = len(pre)
    e = pre[: int(0.30 * n)]              # what 'full' adds over last_30, approximated by first vs last 30%
    l = pre[n - int(0.30 * n):]
    med = max(np.median(np.abs(pre)), 1e-9)

    flat = np.abs(np.diff(e)) < 0.01 * max(pre.std(), 1e-9)
    zero = np.abs(e) <= 0.01 * med
    return dict(
        early_flat_frac=float(flat.mean()),
        early_zero_frac=float(zero.mean()),
        early_late_mean_ratio=float(e.mean() / max(l.mean(), 1e-9)),
        early_late_std_logratio=float(np.log((e.std() + 1e-9)
                                             / (l.std() + 1e-9))),
    )


def run_early_features(df: pd.DataFrame, clean_json: str | Path,
                       out_dir: str | Path = ".") -> pd.DataFrame:
    out_dir = Path(out_dir)
    buildings = json.loads(Path(clean_json).read_text())["buildings"]

    rows = []
    for i, bid in enumerate(buildings, 1):
        feats = early_regime_features(df[bid].to_numpy(float))
        feats["building_id"] = bid
        rows.append(feats)
        if i % 300 == 0 or i == len(buildings):
            print(f"[EARLY] {i}/{len(buildings)}")
    ef = pd.DataFrame(rows).set_index("building_id")
    csv = out_dir / "early_features.csv"
    ef.to_csv(csv)
    print(f"[EARLY] early features → {csv}  shape={ef.shape}")
    return ef
