"""Draws the P3 fine-tuning sample and freezes the routing predictions before any fine-tuning run exists. the output JSON is committed and its git timestamp is the pre-registration evidence.

design rules: stratification uses metadata only (building type x site, parsed from building_id), never profiler output, to avoid circular sampling. proportional allocation over building types (min 1 per type with n>=10 population), round-robin across sites within type for climate diversity. for each sampled building we record but do not sample on the v2 feature vector, the P2 recurrence class, and the frozen routing prediction under the locked rule drift_fired (ac_divergence>0.15 | trend_shift>0.16 | cv>0.57) -> last_15 else first_15."""

from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CV_THR, AC_DIV_THR, TREND_THR = 0.57, 0.15, 0.16   # locked v1 Option-B


def _allocate(counts: pd.Series, n_total: int, min_pop: int = 10) -> dict:
    # proportional allocation, min 1 for any type with population >= min_pop
    eligible = counts[counts >= min_pop]
    alloc = (eligible / eligible.sum() * n_total).round().astype(int).clip(lower=1)
    # fix rounding drift back to exactly n_total
    while alloc.sum() > n_total:
        alloc[alloc.idxmax()] -= 1
    while alloc.sum() < n_total:
        alloc[(eligible / alloc.replace(0, 1)).idxmax()] += 1
    return alloc.to_dict()


def preregister_p3(profiles_csv: str | Path,
                   recurrence_csv: str | Path | None = None,
                   n_total: int = 45,
                   seed: int = 7,
                   out_json: str | Path = "p3_preregistration.json") -> dict:
    profs = pd.read_csv(profiles_csv).set_index("building_id")
    ids = profs.index.to_series()
    btype, site = ids.str.split("_").str[1], ids.str.split("_").str[0]

    rec = None
    if recurrence_csv is not None and Path(recurrence_csv).exists():
        rec = pd.read_csv(recurrence_csv).set_index("building_id")

    rng = np.random.default_rng(seed)
    alloc = _allocate(btype.value_counts(), n_total)
    print(f"[P3] allocation over types: {alloc}  (Σ={sum(alloc.values())})")

    sampled: list[str] = []
    for t, k in alloc.items():
        pool = profs.index[btype == t]
        pool_sites = site[pool]
        # round-robin over sites for climate diversity
        order = []
        for s in rng.permutation(pool_sites.unique()):
            members = list(rng.permutation(pool[pool_sites == s]))
            order.append(members)
        i, picked = 0, []
        while len(picked) < k and any(order):
            ring = order[i % len(order)]
            if ring:
                picked.append(ring.pop())
            i += 1
        sampled.extend(picked)

    entries = []
    for bid in sampled:
        row = profs.loc[bid]
        fired = bool((row["ac_divergence"] > AC_DIV_THR)
                     or (row["trend_shift"] > TREND_THR)
                     or (row["cv"] > CV_THR))
        p2_class = (str(rec.loc[bid, "p2_class"])
                    if rec is not None and bid in rec.index else "N/A")
        entries.append({
            "building_id": bid,
            "btype": bid.split("_")[1], "site": bid.split("_")[0],
            "drift_fired": fired,
            "predicted_slice": "last_15" if fired else "first_15",
            "p2_class": p2_class,
            "features": {k: round(float(v), 6) for k, v in
                         row.drop(["site", "btype"], errors="ignore").items()
                         if isinstance(v, (int, float, np.floating))},
        })

    payload = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "n_total": len(entries),
        "stratification": "metadata only: building type proportional, "
                          "round-robin over sites within type",
        "routing_rule": f"drift_fired(ac_div>{AC_DIV_THR}|trend>{TREND_THR}"
                        f"|cv>{CV_THR}) -> last_15 else first_15",
        "slice_grid_planned": ["zeroshot", "first_15", "last_15",
                               "last_30", "full"],
        "buildings": entries,
    }
    # hash the payload minus this field, sort_keys so it's reproducible: json.dumps(payload, sort_keys=True)
    blob = json.dumps(payload, sort_keys=True).encode()
    payload["sha256"] = hashlib.sha256(blob).hexdigest()
    Path(out_json).write_text(json.dumps(payload, indent=1))

    df = pd.DataFrame(entries)
    print(f"[P3] sampled {len(df)}: "
          f"{df.btype.value_counts().to_dict()}")
    print(f"[P3] sites covered: {df.site.nunique()} | "
          f"predicted last_15: {(df.predicted_slice=='last_15').sum()}, "
          f"first_15: {(df.predicted_slice=='first_15').sum()}")
    if rec is not None:
        print(f"[P3] p2 classes in sample: {df.p2_class.value_counts().to_dict()}")
    print(f"[P3] sha256={payload['sha256'][:16]}…  → {out_json}")
    print("[P3] COMMIT THIS FILE before any fine-tuning run.")
    return payload


def oversample_true_drift(profiles_csv: str | Path,
                          recurrence_csv: str | Path,
                          primary_json: str | Path,
                          n_extra: int = 12,
                          seed: int = 11,
                          out_json: str | Path = "p3_oversample_true_drift.json") -> dict:
    # secondary sample, TRUE_DRIFT only, excluding the registered primary. gives power for the drift-routing hypothesis since the metadata-stratified primary holds few TRUE_DRIFT buildings by construction. same freezing discipline, analysed separately, labelled the targeted supplement everywhere, never silently pooled with the primary
    profs = pd.read_csv(profiles_csv).set_index("building_id")
    rec = pd.read_csv(recurrence_csv).set_index("building_id")
    primary = {b["building_id"] for b in
               json.loads(Path(primary_json).read_text())["buildings"]}

    pool = rec.index[(rec.p2_class == "TRUE_DRIFT")
                     & (~rec.index.isin(primary))
                     & (rec.index.isin(profs.index))]
    site = pool.to_series().str.split("_").str[0]
    rng = np.random.default_rng(seed)

    rings = [list(rng.permutation(pool[site == s]))
             for s in rng.permutation(site.unique())]
    picked, i = [], 0
    while len(picked) < min(n_extra, len(pool)) and any(rings):
        ring = rings[i % len(rings)]
        if ring:
            picked.append(ring.pop())
        i += 1

    entries = []
    for bid in picked:
        row = profs.loc[bid]
        entries.append({
            "building_id": bid,
            "btype": bid.split("_")[1], "site": bid.split("_")[0],
            "p2_class": "TRUE_DRIFT",
            "predicted_slice": "last_15",     # drift-fired by definition
            "features": {k: round(float(v), 6) for k, v in row.items()
                         if isinstance(v, (int, float, np.floating))},
        })
    payload = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "role": "SECONDARY targeted supplement - TRUE_DRIFT power sample",
        "seed": seed, "n": len(entries),
        "excluded_primary_sha": json.loads(
            Path(primary_json).read_text()).get("sha256"),
        "buildings": entries,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()
    Path(out_json).write_text(json.dumps(payload, indent=1))
    df = pd.DataFrame(entries)
    print(f"[P3+] oversampled {len(df)} TRUE_DRIFT: "
          f"types={df.btype.value_counts().to_dict()}, "
          f"sites={df.site.nunique()}")
    print(f"[P3+] sha256={payload['sha256'][:16]}…  → {out_json}")
    return payload
