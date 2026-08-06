"""Stress-tests the H1 headline (site-grouped catastrophe AUC 0.9413) against the objection that it is really a "this-is-Lamb" detector, since Lamb holds 41 of the 54 catastrophes.

reuses the exact features, RF hyperparameters and site-grouping as poison_analysis.py so numbers are comparable, and reports: baseline reproduction, leave-Lamb-out (drop Lamb entirely, retrain, site-cluster bootstrap CI) which is the paper's honest 0.853, within-Lamb (train off-Lamb, score on Lamb), per-site AUC, and a scale-feature mechanism proxy for the non-Lamb catastrophes. quarantine drops last_30 blow-ups but keeps full-arm detonations since those are the signal."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

# kept identical to poison_analysis.py
CAT_THR = 0.10            # poison > 0.10 NMAE is a catastrophe
QUAR_THR = 5.0            # last_30 NMAE > this is uninterpretable, quarantine
RF_KW = dict(n_estimators=400, max_depth=5, min_samples_leaf=10,
             random_state=7, class_weight="balanced")
N_BOOT = 1000
BOOT_SEED = 7

# non-Lamb catastrophe exemplars diagnosed visually
KNOWN_NONLAMB = ["Cockatoo_education_Christi", "Bull_office_Trevor",
                 "Gator_public_Nettie"]

SCALE_FEATS = ["early_late_mean_ratio", "level_traj_trend", "dynamic_range",
               "flat_spot_frac", "early_late_std_logratio",
               "early_flat_frac", "early_zero_frac", "halves_var_logratio"]


def _load(ft_csv, profiles_csv, early_csv):
    ft = pd.read_csv(ft_csv)
    p = ft.pivot(index="building_id", columns="arm", values="nmae")
    need = {"last_30", "full"}
    missing_arm = sorted(need - set(p.columns))
    if missing_arm:
        raise ValueError(f"ft matrix missing arms {missing_arm}")
    n_raw = len(p)
    p = p.dropna(subset=["last_30", "full"])      # drop buildings missing an arm
    n_botharm = len(p)

    profs = pd.read_csv(profiles_csv).set_index("building_id")
    feat = profs[[c for c in profs.columns
                  if c not in ("site", "btype", "sqm")
                  and profs[c].dtype != object]]
    if early_csv and Path(early_csv).exists():
        feat = feat.join(pd.read_csv(early_csv).set_index("building_id"),
                         how="left")
    return p, feat, dict(n_raw=n_raw, n_botharm=n_botharm)


def _xy(p, feat):
    common = p.index.intersection(feat.index)
    X = feat.loc[common].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    poison = p.loc[common, "poison"]
    y = (poison > CAT_THR).astype(int)
    site = pd.Series(common, index=common).str.split("_").str[0]
    return common, X, y, poison, site


def _grouped_oof_proba(X, y, site):
    # pooled out-of-fold P(catastrophe) under GroupKFold by site, exactly the poison_analysis procedure
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict
    cv = GroupKFold(n_splits=min(5, site.nunique()))
    clf = RandomForestClassifier(**RF_KW)
    return cross_val_predict(clf, X, y, cv=cv, groups=site,
                             method="predict_proba")[:, 1]


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def _cluster_bootstrap_auc(y, s, groups, rng, n=N_BOOT):
    # resample sites (clusters) with replacement, recompute AUC. returns (lo, hi, valid_frac) over reps where both classes are present
    y = np.asarray(y); s = np.asarray(s)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    aucs = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        a = _auc(y[idx], s[idx])
        if not np.isnan(a):
            aucs.append(a)
    if not aucs:
        return (np.nan, np.nan, 0.0)
    return (float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)),
            len(aucs) / n)


def _building_bootstrap_auc(y, s, rng, n=N_BOOT):
    # anti-conservative building-level bootstrap (ignores site clustering), reported only as a secondary clearly-labelled CI
    y = np.asarray(y); s = np.asarray(s)
    m = len(y)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        a = _auc(y[idx], s[idx])
        if not np.isnan(a):
            aucs.append(a)
    if not aucs:
        return (np.nan, np.nan, 0.0)
    return (float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)),
            len(aucs) / n)


def run_robustness(ft_csv: str | Path,
                   profiles_csv: str | Path = "data/profiles_v2.csv",
                   early_csv: str | Path | None = "data/early_features.csv",
                   dominant_site: str | None = None,
                   quar_thr: float = QUAR_THR,
                   out_json: str | Path = "data/robustness_lamb.json") -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    p, feat, counts = _load(ft_csv, profiles_csv, early_csv)
    p["poison"] = p["full"] - p["last_30"]
    rep: dict = {"counts": counts, "cat_thr": CAT_THR, "quar_thr": quar_thr}

    bad_safe = p["last_30"] > quar_thr
    cat_pre = p["poison"] > CAT_THR
    dropped_cats = int((bad_safe & cat_pre).sum())
    print(f"[ROB] raw={counts['n_raw']} both-arm={counts['n_botharm']}")
    print(f"[ROB] quarantine last_30 NMAE>{quar_thr}: dropping "
          f"{int(bad_safe.sum())} buildings "
          f"({dropped_cats} of them were catastrophes by full-poison)")
    rep["quarantine_sens"] = {
        str(t): int((p["last_30"] > t).sum()) for t in (2.0, 5.0, 10.0)}
    p = p[~bad_safe].copy()
    rep["n_post_quarantine"] = int(len(p))

    common, X, y, poison, site = _xy(p, feat)
    n_pos = int(y.sum())
    print(f"[ROB] post-quarantine analysis set: n={len(common)} "
          f"positives={n_pos} sites={site.nunique()}")
    rep["n_analysis"] = int(len(common))
    rep["n_positives"] = n_pos

    cat_by_site = y.groupby(site).sum().sort_values(ascending=False)
    if dominant_site is None:
        dominant_site = str(cat_by_site.index[0])
    dom_pos = int(cat_by_site.get(dominant_site, 0))
    print(f"[ROB] dominant site = {dominant_site}: {dom_pos}/{n_pos} "
          f"catastrophes ({100*dom_pos/max(n_pos,1):.0f}% of positives)")
    rep["dominant_site"] = dominant_site
    rep["dominant_site_positives"] = dom_pos
    rep["cat_by_site_top"] = cat_by_site[cat_by_site > 0].astype(int).to_dict()

    if n_pos < 10:
        print("[ROB] <10 positives after quarantine — probes underpowered.")

    proba = _grouped_oof_proba(X, y, site)
    auc_full = _auc(y, proba)
    lo_c, hi_c, vf_c = _cluster_bootstrap_auc(y, proba, site, rng)
    print(f"\n[ROB] (1) BASELINE pooled site-grouped AUC = {auc_full:.3f}  "
          f"site-cluster 95% CI [{lo_c:.3f}, {hi_c:.3f}] (valid {vf_c:.0%})")
    rep["baseline"] = dict(auc=round(auc_full, 4),
                           cluster_ci=[round(lo_c, 4), round(hi_c, 4)],
                           cluster_valid_frac=round(vf_c, 3))

    keep = site != dominant_site
    Xo, yo, so = X[keep.values], y[keep.values], site[keep.values]
    n_pos_o = int(yo.sum())
    print(f"\n[ROB] (2) LEAVE-{dominant_site.upper()}-OUT: n={len(Xo)} "
          f"positives={n_pos_o} sites={so.nunique()}")
    if n_pos_o >= 2 and len(np.unique(yo)) == 2:
        proba_o = _grouped_oof_proba(Xo, yo, so)
        auc_o = _auc(yo, proba_o)
        lo_o, hi_o, vf_o = _cluster_bootstrap_auc(yo, proba_o, so, rng)
        lo_b, hi_b, vf_b = _building_bootstrap_auc(yo, proba_o, rng)
        print(f"      pooled site-grouped AUC = {auc_o:.3f}  "
              f"site-cluster 95% CI [{lo_o:.3f}, {hi_o:.3f}] (valid {vf_o:.0%})")
        print(f"      [secondary, anti-conservative] building 95% CI "
              f"[{lo_b:.3f}, {hi_b:.3f}]")
        rep["leave_dominant_out"] = dict(
            n=int(len(Xo)), positives=n_pos_o, auc=round(auc_o, 4),
            cluster_ci=[round(lo_o, 4), round(hi_o, 4)],
            cluster_valid_frac=round(vf_o, 3),
            building_ci=[round(lo_b, 4), round(hi_b, 4)])
        print(f"      >>> verdict: {'SURVIVES' if auc_o >= 0.75 else 'WEAKENS/COLLAPSES'} "
              f"(baseline {auc_full:.3f} → {auc_o:.3f}); "
              f"note only {n_pos_o} positives → CI is the real story, not point est.")
    else:
        print(f"      only {n_pos_o} positives off-{dominant_site} — AUC undefined/underpowered.")
        rep["leave_dominant_out"] = dict(n=int(len(Xo)), positives=n_pos_o,
                                         auc=None, note="underpowered")

    dom = ~keep
    Xd, yd = X[dom.values], y[dom.values]
    print(f"\n[ROB] (3) WITHIN-{dominant_site.upper()}: "
          f"{int(yd.sum())}/{len(yd)} catastrophes")
    if len(np.unique(yd)) == 2 and len(np.unique(yo)) == 2:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(**RF_KW).fit(Xo, yo)   # trained off-site only
        sd = clf.predict_proba(Xd)[:, 1]
        auc_w = _auc(yd, sd)
        lo_w, hi_w, vf_w = _building_bootstrap_auc(yd, sd, rng)
        print(f"      AUC (off-{dominant_site} model, scored within {dominant_site}) "
              f"= {auc_w:.3f}  95% CI [{lo_w:.3f}, {hi_w:.3f}]")
        print(f"      >>> {'fingerprints distinguish WHICH ' + dominant_site + ' buildings detonate' if auc_w >= 0.70 else 'NO within-site signal — predictor is largely a ' + dominant_site + ' detector'}")
        rep["within_dominant"] = dict(
            n=int(len(yd)), positives=int(yd.sum()), auc=round(auc_w, 4),
            building_ci=[round(lo_w, 4), round(hi_w, 4)])
    else:
        print(f"      {dominant_site} lacks both classes — within-site AUC undefined.")
        rep["within_dominant"] = dict(n=int(len(yd)), positives=int(yd.sum()),
                                      auc=None, note="single class")

    per_site = {}
    for g, idx in pd.Series(range(len(common)), index=site).groupby(level=0):
        ii = idx.values
        if len(np.unique(np.asarray(y)[ii])) == 2:
            per_site[g] = round(_auc(np.asarray(y)[ii], proba[ii]), 3)
    print(f"\n[ROB] (4) per-site OOF AUC computable for {len(per_site)} site(s) "
          f"(rest have one class): {per_site}")
    rep["per_site_auc"] = per_site

    cats = y[y == 1].index
    nonlamb_cats = [b for b in cats if not b.startswith(dominant_site + "_")]
    lamb_cats = [b for b in cats if b.startswith(dominant_site + "_")]
    avail = [f for f in SCALE_FEATS if f in X.columns]
    summ = {}
    for f in avail:
        col = X[f]
        summ[f] = dict(
            pop_median=round(float(col.median()), 4),
            lamb_cat_median=round(float(col.loc[lamb_cats].median()), 4)
                if lamb_cats else None,
            nonlamb_cat_median=round(float(col.loc[nonlamb_cats].median()), 4)
                if nonlamb_cats else None)
    print(f"\n[ROB] (5) mechanism proxy — {len(nonlamb_cats)} non-{dominant_site} "
          f"catastrophes vs {len(lamb_cats)} {dominant_site} catastrophes:")
    print(f"      {'feature':24s} {'pop':>10s} {dominant_site[:8]+'_cat':>12s} {'other_cat':>12s}")
    for f in avail:
        s = summ[f]
        print(f"      {f:24s} {s['pop_median']:>10} "
              f"{str(s['lamb_cat_median']):>12} {str(s['nonlamb_cat_median']):>12}")
    rep["mechanism_proxy"] = summ
    rep["nonlamb_catastrophes"] = sorted(nonlamb_cats)
    known_here = [b for b in KNOWN_NONLAMB if b in set(common)]
    flagged = {b: int(b in set(cats)) for b in known_here}
    print(f"      known non-{dominant_site} exemplars present & flagged-as-cat: {flagged}")
    rep["known_nonlamb_flagged"] = flagged

    rep["combined_headline"] = dict(
        description=f"post-quarantine (last_30>{quar_thr} dropped) + "
                    f"leave-{dominant_site}-out, pooled site-grouped AUC",
        auc=rep["leave_dominant_out"].get("auc"),
        cluster_ci=rep["leave_dominant_out"].get("cluster_ci"),
        n_positives=rep["leave_dominant_out"].get("positives"))

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(rep, indent=1, default=str))
    print(f"\n[ROB] report → {out_json}")
    return rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ft_csv")
    ap.add_argument("--profiles", default="data/profiles_v2.csv")
    ap.add_argument("--early", default="data/early_features.csv")
    ap.add_argument("--dominant", default=None)
    ap.add_argument("--quar", type=float, default=QUAR_THR)
    ap.add_argument("--out", default="data/robustness_lamb.json")
    a = ap.parse_args()
    run_robustness(a.ft_csv, a.profiles, a.early, a.dominant, a.quar, a.out)
