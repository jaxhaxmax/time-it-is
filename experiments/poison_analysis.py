"""Analysis of the population-scale poisoning study (PatchTST fine-tuned on all clean buildings x {last_30, full}).

reads the population ft matrix plus profiles_v2.csv, early_features.csv and drift_recurrence.csv, and reports: poison label stats (poison = nmae_full - nmae_last_30, positive means history hurt, catastrophe when poison > 0.10), the policy table (always_full vs always_last_30 vs oracle-of-the-two vs feature-routed, mean/median NMAE and catastrophe counts), routability probes under site-grouped CV (classifier AUC on catastrophe yes/no, out-of-fold Spearman on poison magnitude, with v2 + early-regime feature importances), and catastrophe-rate breakdowns by P2 class and building type.

four buildings with last_30 NMAE > 5.0 are quarantined before any population statistic (last_30 is meant to be the safe fallback, so a last_30 blow-up is a broken evaluation not a routing signal). this drops the population to 1,420 with 54 catastrophes, matching the paper."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

CAT_THR = 0.10        # poison > 0.10 NMAE is a catastrophe, sensitivity at 0.05/0.10/0.20 reported below
QUARANTINE_THR = 5.0  # drop buildings whose last_30 itself blows up past this NMAE


def run_poison_analysis(ft_csv: str | Path,
                        profiles_csv: str | Path,
                        early_csv: str | Path | None = None,
                        recurrence_csv: str | Path | None = None,
                        out_json: str | Path = "poison_analysis.json") -> dict:
    ft = pd.read_csv(ft_csv)
    p = ft.pivot(index="building_id", columns="arm", values="nmae")
    need = {"last_30", "full"}
    if not need.issubset(p.columns):
        raise ValueError(f"ft matrix missing arms {need - set(p.columns)}")
    p = p.dropna(subset=["last_30", "full"])
    n_pre = len(p)
    quarantined = list(p.index[p.last_30 > QUARANTINE_THR])
    p = p[p.last_30 <= QUARANTINE_THR]    # quarantine before any stat
    print(f"[POIS] quarantined {len(quarantined)} (last_30>{QUARANTINE_THR}): "
          f"{quarantined}")
    p["poison"] = p["full"] - p["last_30"]
    rep: dict = {"n": int(len(p)), "n_before_quarantine": int(n_pre),
                 "quarantined": quarantined}

    print(f"[POIS] n={len(p)} buildings with both arms (post-quarantine)")
    print("[POIS] poison distribution: median=%.4f P90=%.4f P99=%.4f max=%.2f"
          % tuple(np.percentile(p.poison, [50, 90, 99, 100])))
    for thr in (0.05, 0.10, 0.20):
        print(f"[POIS] catastrophes (poison>{thr}): "
              f"{(p.poison > thr).sum()} ({100*(p.poison > thr).mean():.1f}%)")
    rep["poison_pcts"] = {f">{t}": int((p.poison > t).sum())
                          for t in (0.05, 0.10, 0.20)}
    cat = p.poison > CAT_THR

    oracle = p[["last_30", "full"]].min(axis=1)
    pol = {"always_full": p.full, "always_last_30": p.last_30,
           "oracle_2arm": oracle}
    print("\n[POIS] policy table:")
    rep["policies"] = {}
    for k, v in pol.items():
        rep["policies"][k] = {"mean": round(float(v.mean()), 4),
                              "median": round(float(v.median()), 4)}
        print(f"  {k:16s} mean={v.mean():.4f} median={v.median():.4f}")

    btype = p.index.to_series().str.split("_").str[1]
    print("\n[POIS] catastrophe rate by type (n≥20):")
    tab = cat.groupby(btype).agg(["mean", "sum", "count"])
    big = tab[tab["count"] >= 20].sort_values("mean", ascending=False)
    print((big.assign(mean=lambda d: d["mean"].round(3))).to_string())
    rep["cat_by_type"] = big["mean"].round(3).to_dict()

    if recurrence_csv and Path(recurrence_csv).exists():
        rec = pd.read_csv(recurrence_csv).set_index("building_id")
        joined = p.join(rec.p2_class, how="left")
        print("\n[POIS] catastrophe rate by P2 class:")
        t = (joined.poison > CAT_THR).groupby(joined.p2_class).agg(
            ["mean", "sum", "count"])
        print(t.round(3).to_string())
        rep["cat_by_p2"] = t["mean"].round(3).to_dict()

    profs = pd.read_csv(profiles_csv).set_index("building_id")
    feat = profs[[c for c in profs.columns
                  if c not in ("site", "btype", "sqm")
                  and profs[c].dtype != object]]
    if early_csv and Path(early_csv).exists():
        feat = feat.join(pd.read_csv(early_csv).set_index("building_id"),
                         how="left")
    common = p.index.intersection(feat.index)
    X = feat.loc[common].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    groups = pd.Series(common, index=common).str.split("_").str[0]

    try:
        from sklearn.ensemble import (RandomForestClassifier,
                                      RandomForestRegressor)
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.metrics import roc_auc_score
        from scipy.stats import spearmanr
    except ImportError:
        print("[POIS] sklearn unavailable — skipping probes")
        Path(out_json).write_text(json.dumps(rep, indent=1, default=str))
        return rep

    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    y_cat = cat.loc[common].astype(int)
    if y_cat.sum() >= 10:
        clf = RandomForestClassifier(n_estimators=400, max_depth=5,
                                     min_samples_leaf=10, random_state=7,
                                     class_weight="balanced")
        proba = cross_val_predict(clf, X, y_cat, cv=cv, groups=groups,
                                  method="predict_proba")[:, 1]
        auc = roc_auc_score(y_cat, proba)
        clf.fit(X, y_cat)
        imp = pd.Series(clf.feature_importances_, index=X.columns)
        print(f"\n[POIS] catastrophe routability: site-grouped AUC = {auc:.3f} "
              f"({int(y_cat.sum())} positives)")
        print("       top features:",
              imp.nlargest(8).round(3).to_dict())
        rep["catastrophe_auc"] = round(float(auc), 4)
        rep["top_features_cat"] = imp.nlargest(8).round(4).to_dict()

        # feature-routed policy: send to last_30 when predicted risky, else full
        routed = np.where(proba > 0.5, p.loc[common, "last_30"],
                          p.loc[common, "full"])
        print("       feature-routed policy: mean=%.4f median=%.4f"
              % (np.mean(routed), np.median(routed)))
        rep["policies"]["feature_routed"] = {
            "mean": round(float(np.mean(routed)), 4),
            "median": round(float(np.median(routed)), 4)}
    else:
        print(f"\n[POIS] only {int(y_cat.sum())} catastrophes — "
              f"classifier probe underpowered, skipping")

    reg = RandomForestRegressor(n_estimators=400, max_depth=6,
                                min_samples_leaf=10, random_state=7)
    pred = cross_val_predict(reg, X, p.loc[common, "poison"],
                             cv=cv, groups=groups)
    rho = spearmanr(pred, p.loc[common, "poison"])[0]
    print(f"[POIS] poison-magnitude routability: out-of-fold Spearman = {rho:.3f}")
    rep["poison_spearman"] = round(float(rho), 4)

    Path(out_json).write_text(json.dumps(rep, indent=1, default=str))
    print(f"[POIS] report → {out_json}")
    return rep
