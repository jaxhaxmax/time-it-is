"""Calibrates the router's operating point. the catastrophe classifier ranks well (AUC 0.9413) but proba>0.5 flags almost nobody on a 3.8% base rate, so this sweeps the threshold and picks the point that actually catches catastrophes.

same setup as robustness_lamb.py (quarantine, RF 400/5/10/balanced/seed7, GroupKFold by site, profiles_v2 + early_features), AUC sanity-gated before proceeding. writes threshold_analysis.json (the sweep figure is rendered by make_figures.py)."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

CAT_THR = 0.10
QUAR_THR = 5.0
RF_KW = dict(n_estimators=400, max_depth=5, min_samples_leaf=10,
             random_state=7, class_weight="balanced")
AUC_SANITY_LO = 0.930   # abort if the reproduced AUC drifts below this


def _load(ft_csv, profiles_csv, early_csv):
    ft = pd.read_csv(ft_csv)
    p = ft.pivot(index="building_id", columns="arm", values="nmae")
    if not {"last_30", "full"}.issubset(p.columns):
        raise ValueError("ft matrix needs both last_30 and full arms")
    n_raw = len(p)
    p = p.dropna(subset=["last_30", "full"])
    return p, n_raw


def _features(profiles_csv, early_csv):
    profs = pd.read_csv(profiles_csv).set_index("building_id")
    feat = profs[[c for c in profs.columns
                  if c not in ("site", "btype", "sqm")
                  and profs[c].dtype != object]]
    if early_csv and Path(early_csv).exists():
        feat = feat.join(pd.read_csv(early_csv).set_index("building_id"),
                         how="left")
    return feat


def _align(p, feat):
    common = p.index.intersection(feat.index)
    X = feat.loc[common].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    poison = p.loc[common, "full"] - p.loc[common, "last_30"]
    y = (poison > CAT_THR).astype(int)
    site = pd.Series(common, index=common).str.split("_").str[0]
    return common, X, y, site, p.loc[common, "last_30"], p.loc[common, "full"]


def _summ(v):
    return dict(mean=round(float(v.mean()), 4),
                median=round(float(v.median()), 4),
                p90=round(float(np.percentile(v, 90)), 4),
                max=round(float(v.max()), 4))


def run_threshold_analysis(
        ft_csv: str | Path,
        profiles_csv: str | Path = "data/profiles_v2.csv",
        early_csv: str | Path | None = "data/early_features.csv",
        zs_bolt_csv: str | Path | None = None,
        quar_thr: float = QUAR_THR,
        out_json: str | Path = "data/threshold_analysis.json") -> dict:

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    p, n_raw = _load(ft_csv, profiles_csv, early_csv)
    feat = _features(profiles_csv, early_csv)

    bad = p["last_30"] > quar_thr
    print(f"[THR] raw={n_raw} both-arm={len(p)} "
          f"quarantine={int(bad.sum())} → n={int((~bad).sum())}")
    p = p[~bad].copy()

    common, X, y, site, last30, full = _align(p, feat)
    n_pos = int(y.sum())
    n_total = len(common)
    print(f"[THR] analysis set: n={n_total} positives={n_pos} "
          f"({100*n_pos/n_total:.1f}%) sites={site.nunique()}")

    cv = GroupKFold(n_splits=min(5, site.nunique()))
    clf = RandomForestClassifier(**RF_KW)
    proba = cross_val_predict(clf, X, y, cv=cv, groups=site,
                              method="predict_proba")[:, 1]
    proba = pd.Series(proba, index=common)

    auc = float(roc_auc_score(y, proba))
    print(f"[THR] reproduced pooled AUC = {auc:.4f}")
    if auc < AUC_SANITY_LO:
        raise RuntimeError(f"AUC {auc:.4f} < {AUC_SANITY_LO} — "
                           f"something changed vs robustness_lamb")

    rep: dict = dict(n=n_total, n_positives=n_pos, reproduced_auc=round(auc, 4))

    cat_mask = y == 1
    proba_cat = proba[cat_mask.values]
    proba_safe = proba[~cat_mask.values]
    print(f"\n[THR] probability distribution:")
    print(f"  catastrophes (n={n_pos}): "
          f"min={proba_cat.min():.4f} median={proba_cat.median():.4f} "
          f"max={proba_cat.max():.4f}")
    print(f"  safe (n={n_total - n_pos}): "
          f"min={proba_safe.min():.4f} median={proba_safe.median():.4f} "
          f"max={proba_safe.max():.4f}")
    rep["proba_distribution"] = dict(
        cat_min=round(float(proba_cat.min()), 4),
        cat_median=round(float(proba_cat.median()), 4),
        cat_max=round(float(proba_cat.max()), 4),
        safe_min=round(float(proba_safe.min()), 4),
        safe_median=round(float(proba_safe.median()), 4),
        safe_max=round(float(proba_safe.max()), 4))

    always_full_mean = float(full.mean())
    always_last30_mean = float(last30.mean())
    oracle = pd.Series(np.minimum(last30.values, full.values), index=common)
    oracle_mean = float(oracle.mean())
    print(f"\n[THR] reference policies (mean NMAE):")
    print(f"  always_full={always_full_mean:.4f}  "
          f"always_last_30={always_last30_mean:.4f}  "
          f"oracle={oracle_mean:.4f}")

    thresholds = np.arange(0.01, 1.00, 0.01)
    sweep = []
    for thr in thresholds:
        flagged = proba > thr
        tp = int((flagged & cat_mask).sum())
        fp = int((flagged & ~cat_mask).sum())
        catch_rate = tp / n_pos if n_pos > 0 else 0.0
        routed = pd.Series(
            np.where(flagged.values, last30.values, full.values), index=common)
        gap = always_last30_mean - oracle_mean
        closure = ((always_last30_mean - float(routed.mean())) / gap * 100
                   if gap > 1e-9 else float("nan"))
        sweep.append(dict(
            threshold=round(float(thr), 2), n_flagged=int(flagged.sum()),
            n_caught=tp, n_missed=n_pos - tp,
            catch_rate=round(catch_rate, 4), unnecessary_routes=fp,
            policy_mean_nmae=round(float(routed.mean()), 4),
            policy_median_nmae=round(float(routed.median()), 4),
            gap_closure_pct=round(float(closure), 2)))
    rep["sweep"] = sweep

    def _best_at(target_catch):
        # cheapest threshold hitting the catch target: fewest unnecessary routes, ties broken by gap closure
        cands = [s for s in sweep if s["catch_rate"] >= target_catch]
        if not cands:
            return None
        cands.sort(key=lambda s: (s["unnecessary_routes"],
                                  -s["gap_closure_pct"]))
        return cands[0]

    rec_90 = _best_at(0.90)
    rec_95 = _best_at(0.95)
    for label, rec in [("90% catch", rec_90), ("95% catch", rec_95)]:
        if rec:
            print(f"\n[THR] recommended operating point ({label}):")
            print(f"  threshold={rec['threshold']}  flagged={rec['n_flagged']}  "
                  f"caught={rec['n_caught']}/{n_pos}  missed={rec['n_missed']}  "
                  f"unnecessary={rec['unnecessary_routes']}")
            print(f"  routed mean NMAE={rec['policy_mean_nmae']}  "
                  f"median={rec['policy_median_nmae']}  "
                  f"gap closure={rec['gap_closure_pct']}%")
        else:
            print(f"\n[THR] no threshold achieves {label}")
    rep["recommended_90"] = rec_90
    rep["recommended_95"] = rec_95

    policies = {}
    if rec_90:
        thr_r = rec_90["threshold"]
        flagged_r = proba > thr_r
        routed_r = pd.Series(
            np.where(flagged_r.values, last30.values, full.values), index=common)
        last30_max = float(last30.max())
        # tail leak: a "safe" route that still exceeds the worst last_30
        leaked = int((routed_r > last30_max).sum()) if routed_r.max() > last30_max else 0
        policies = {"always_full": _summ(full), "always_last_30": _summ(last30),
                    "oracle_2arm": _summ(oracle), "risk_gated_rec": _summ(routed_r)}
        policies["risk_gated_rec"].update(
            threshold=thr_r, n_flagged=int(flagged_r.sum()),
            cats_caught=rec_90["n_caught"], cats_missed=rec_90["n_missed"],
            tail_leaked=leaked)
        print(f"\n[THR] policy table at recommended operating point (thr={thr_r}):")
        print(f"  {'policy':18s} {'mean':>8s} {'median':>8s} {'p90':>8s} {'max':>8s}")
        for k, v in policies.items():
            print(f"  {k:18s} {v['mean']:>8.4f} {v['median']:>8.4f} "
                  f"{v['p90']:>8.4f} {v['max']:>8.4f}")
        if leaked:
            print(f"  WARNING: {leaked} catastrophe(s) leaked past "
                  f"always_last_30 max ({last30_max:.2f})")
        rep["policy_table"] = policies

    if zs_bolt_csv and Path(zs_bolt_csv).exists():
        zs = pd.read_csv(zs_bolt_csv)
        if "model" in zs.columns:
            zb = (zs[zs["model"] == "chronos_bolt_small"]
                  .set_index("building_id")["nmae"])
        else:
            zb = zs.set_index(zs.columns[0])["nmae"]
        c2 = common.intersection(zb.index)
        bolt = zb.loc[c2]
        if policies:
            policies["zs_bolt"] = _summ(bolt)
        ftbest = pd.Series(np.minimum(last30.loc[c2].values, full.loc[c2].values),
                           index=c2)
        win_frac = float((ftbest < bolt).mean())
        print(f"\n[THR] ZS-Bolt on same population (n={len(c2)}): "
              f"mean={bolt.mean():.4f} median={bolt.median():.4f}")
        print(f"  FT-best beats ZS-Bolt on {100*win_frac:.1f}% of buildings")
        rep["cost_framing"] = dict(n=int(len(c2)), zs_bolt=_summ(bolt),
                                   ftbest_win_frac=round(win_frac, 4))
    else:
        print("\n[THR] no ZS-Bolt file — cost framing skipped")

    clf_full = RandomForestClassifier(**RF_KW).fit(X, y)
    imp = pd.Series(clf_full.feature_importances_,
                    index=X.columns).sort_values(ascending=False)
    top12 = imp.head(12)
    print(f"\n[THR] top 12 feature importances:")
    for f, v in top12.items():
        print(f"  {f:30s} {v:.4f}")
    rep["feature_importance_top12"] = {str(k): round(float(v), 4)
                                       for k, v in top12.items()}

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(rep, indent=1, default=str))
    print(f"\n[THR] report → {out_json}")

    return rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ft_csv")
    ap.add_argument("--profiles", default="data/profiles_v2.csv")
    ap.add_argument("--early", default="data/early_features.csv")
    ap.add_argument("--zs_bolt", default=None)
    ap.add_argument("--quar", type=float, default=QUAR_THR)
    ap.add_argument("--out", default="data/threshold_analysis.json")
    a = ap.parse_args()
    run_threshold_analysis(a.ft_csv, a.profiles, a.early, a.zs_bolt,
                           a.quar, a.out)
