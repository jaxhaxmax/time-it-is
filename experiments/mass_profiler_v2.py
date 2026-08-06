"""Runs the v2 profiler over the pinned clean-building list and writes profiles_v2.csv (one row per building, all v2 features plus metadata) and profiles_v2_pruning.json (the |r|>0.9 high-correlation pairs report)."""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.spectral_building_profiler_v2 import (
    profile_building_v2, FEATURE_NAMES)


def run_mass_profiling_v2(df: pd.DataFrame,
                          meta: pd.DataFrame | None,
                          clean_json: str | Path,
                          out_dir: str | Path = ".",
                          progress_every: int = 200) -> pd.DataFrame:
    out_dir = Path(out_dir)
    buildings = json.loads(Path(clean_json).read_text())["buildings"]
    start = str(df.index[0]) if hasattr(df.index, "dtype") else "2017-01-01"
    print(f"[V2] Profiling {len(buildings)} buildings (start={start[:10]}) …")

    rows, fails, t0 = [], 0, time.time()
    for i, bid in enumerate(buildings, 1):
        try:
            feats = profile_building_v2(df[bid].to_numpy(np.float64), start=start)
            feats["building_id"] = bid
            rows.append(feats)
        except Exception as e:
            fails += 1
            print(f"[V2][FAIL] {bid}: {e}")
        if i % progress_every == 0 or i == len(buildings):
            rate = i / max(time.time() - t0, EPS := 1e-9)
            print(f"[V2] {i}/{len(buildings)}  ({rate:.0f} bld/s, fails={fails})")

    profs = pd.DataFrame(rows).set_index("building_id")[FEATURE_NAMES]

    # site/type parsed from the id as a fallback when metadata is absent
    profs["site"] = profs.index.str.split("_").str[0]
    profs["btype"] = profs.index.str.split("_").str[1]
    if meta is not None and "sqm" in getattr(meta, "columns", []):
        profs = profs.join(meta.set_index(meta.columns[0])[["sqm"]], how="left")

    csv_path = out_dir / "profiles_v2.csv"
    profs.to_csv(csv_path)
    print(f"[V2] profiles → {csv_path}  shape={profs.shape}")

    corr = profs[FEATURE_NAMES].corr().abs()
    pairs = [
        {"a": a, "b": b, "abs_r": round(float(corr.loc[a, b]), 3)}
        for i, a in enumerate(FEATURE_NAMES)
        for b in FEATURE_NAMES[i + 1:]
        if corr.loc[a, b] > 0.9
    ]
    rep_path = out_dir / "profiles_v2_pruning.json"
    rep_path.write_text(json.dumps(
        {"n_features": len(FEATURE_NAMES), "high_corr_pairs": pairs}, indent=1))
    print(f"[V2] {len(pairs)} feature pairs with |r|>0.9 → {rep_path}")
    for p_ in pairs:
        print(f"      {p_['a']} ~ {p_['b']}  (|r|={p_['abs_r']})")
    return profs
