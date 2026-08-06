"""All robustness tables for the paper in one run, written under results/robustness/ (single-feature baseline, reduced feature set, CV-scheme and threshold sensitivity, missed catastrophes, detonation verification, classifier comparison, plus robustness_summary.json).

setup matches every other script: quarantine last_30>5.0, RF(400/5/10/balanced/seed7), GroupKFold by site. the leave-Lamb column here is the pooled-OOF diagnostic, the paper's headline leave-Lamb-out (0.853) is the stricter protocol in robustness_lamb.py."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd


CAT_THR = 0.10
QUAR_THR = 5.0
RF_KW = dict(n_estimators=400, max_depth=5, min_samples_leaf=10,
             random_state=7, class_weight="balanced")


def _load(ft_csv, profiles_csv, early_csv):
    ft = pd.read_csv(ft_csv)
    p = ft.pivot(index="building_id", columns="arm", values="nmae")
    p = p.dropna(subset=["last_30", "full"])
    p = p[p["last_30"] <= QUAR_THR].copy()
    p["poison"] = p["full"] - p["last_30"]

    profs = pd.read_csv(profiles_csv).set_index("building_id")
    feat_cols = [c for c in profs.columns
                 if c not in ("site", "btype", "sqm") and profs[c].dtype != object]
    feat = profs[feat_cols].copy()
    if early_csv and Path(early_csv).exists():
        feat = feat.join(pd.read_csv(early_csv).set_index("building_id"),
                         how="left")

    common = p.index.intersection(feat.index)
    X = feat.loc[common].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    y = (p.loc[common, "poison"] > CAT_THR).astype(int)
    site = pd.Series(common, index=common).str.split("_").str[0]
    return p, common, X, y, site


def run_robustness_suite(
        ft_csv: str | Path,
        profiles_csv: str | Path = "data/profiles_v2.csv",
        early_csv: str | Path | None = "data/early_features.csv",
        zs_csv: str | Path | None = "data/zs_label_matrix_dec_full.csv",
        out_dir: str | Path = "data/robustness") -> dict:

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import (GroupKFold, LeaveOneGroupOut,
                                         cross_val_predict)
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    p, common, X, y, site = _load(ft_csv, profiles_csv, early_csv)
    n, n_pos = len(common), int(y.sum())
    cv5 = GroupKFold(n_splits=min(5, site.nunique()))
    print(f"[ROB] n={n} pos={n_pos} features={X.shape[1]} sites={site.nunique()}")

    rep: dict = dict(n=n, n_pos=n_pos)

    proba_rf = cross_val_predict(RandomForestClassifier(**RF_KW), X, y,
                                  cv=cv5, groups=site,
                                  method="predict_proba")[:, 1]
    proba_rf = pd.Series(proba_rf, index=common)
    auc_rf = roc_auc_score(y, proba_rf)

    print(f"\n[F1] SINGLE-FEATURE BASELINE")
    lamb_mask = site == "Lamb"
    nonlamb_mask = ~lamb_mask

    rows_f1 = []
    for col in X.columns:
        vals = X[col].values.copy()
        if np.std(vals) < 1e-12:
            continue
        if np.corrcoef(vals, y.values)[0, 1] < 0:
            vals = -vals                       # flip sign so higher means riskier
        auc_all = roc_auc_score(y, vals)

        if len(np.unique(y[nonlamb_mask.values])) == 2:
            auc_nl = roc_auc_score(y[nonlamb_mask.values],
                                   vals[nonlamb_mask.values])
        else:
            auc_nl = np.nan
        if len(np.unique(y[lamb_mask.values])) == 2:
            auc_wl = roc_auc_score(y[lamb_mask.values],
                                   vals[lamb_mask.values])
        else:
            auc_wl = np.nan
        rows_f1.append(dict(feature=col, auc_overall=round(auc_all, 4),
                            auc_leave_lamb=round(auc_nl, 4),
                            auc_within_lamb=round(auc_wl, 4)))

    df_f1 = pd.DataFrame(rows_f1).sort_values("auc_overall", ascending=False)
    if len(np.unique(y[nonlamb_mask.values])) == 2:
        rf_nl = roc_auc_score(y[nonlamb_mask.values],
                              proba_rf.values[nonlamb_mask.values])
    else:
        rf_nl = np.nan
    if len(np.unique(y[lamb_mask.values])) == 2:
        rf_wl = roc_auc_score(y[lamb_mask.values],
                              proba_rf.values[lamb_mask.values])
    else:
        rf_wl = np.nan

    rf_row = dict(feature="RF_39features", auc_overall=round(auc_rf, 4),
                  auc_leave_lamb=round(rf_nl, 4),
                  auc_within_lamb=round(rf_wl, 4))
    df_f1 = pd.concat([df_f1, pd.DataFrame([rf_row])], ignore_index=True)
    df_f1.to_csv(out / "single_feature_table.csv", index=False)

    top5_f1 = df_f1.head(6)
    print(f"  {'feature':30s} {'overall':>8s} {'leave-Lamb':>10s} {'within-Lamb':>12s}")
    for _, r in top5_f1.iterrows():
        print(f"  {r['feature']:30s} {r['auc_overall']:>8.4f} "
              f"{r['auc_leave_lamb']:>10.4f} {r['auc_within_lamb']:>12.4f}")
    print(f"  {'RF_39features':30s} {rf_row['auc_overall']:>8.4f} "
          f"{rf_row['auc_leave_lamb']:>10.4f} {rf_row['auc_within_lamb']:>12.4f}")
    rep["single_feature_top5"] = top5_f1.to_dict("records")
    rep["rf_baseline"] = rf_row

    print(f"\n[F2] FEATURE CORRELATION")
    top_feats = df_f1[df_f1["feature"] != "RF_39features"].head(10)["feature"].tolist()
    corr = X[top_feats].corr()
    high_pairs = []
    for i, f1 in enumerate(top_feats):
        for f2 in top_feats[i+1:]:
            r = abs(corr.loc[f1, f2])
            if r > 0.4:
                high_pairs.append(dict(f1=f1, f2=f2, abs_r=round(r, 3)))
    print(f"  {len(high_pairs)} pairs with |r| > 0.4")
    rep["high_corr_pairs"] = high_pairs

    print(f"\n[F3] REDUCED FEATURE SET")
    clf_full = RandomForestClassifier(**RF_KW).fit(X, y)
    mdi = pd.Series(clf_full.feature_importances_,
                     index=X.columns).sort_values(ascending=False)
    rows_f3 = []
    for k in [3, 5, 10, 15, 20, X.shape[1]]:
        top_k = mdi.head(k).index.tolist()
        pr_k = cross_val_predict(RandomForestClassifier(**RF_KW),
                                  X[top_k], y, cv=cv5, groups=site,
                                  method="predict_proba")[:, 1]
        a_k = roc_auc_score(y, pr_k)
        rows_f3.append(dict(n_features=k, auc=round(a_k, 4)))
        print(f"  top-{k:2d}: AUC={a_k:.4f}")
    pd.DataFrame(rows_f3).to_csv(out / "reduced_features.csv", index=False)
    rep["reduced_features"] = rows_f3

    print(f"\n[F4] CV FOLD SENSITIVITY")
    rows_f4 = []
    for n_splits in [3, 4, 5, 7, 10]:
        if n_splits > site.nunique():
            continue
        cv_k = GroupKFold(n_splits=n_splits)
        pr_k = cross_val_predict(RandomForestClassifier(**RF_KW), X, y,
                                  cv=cv_k, groups=site,
                                  method="predict_proba")[:, 1]
        a_k = roc_auc_score(y, pr_k)
        rows_f4.append(dict(scheme=f"{n_splits}-fold", auc=round(a_k, 4)))
        print(f"  {n_splits}-fold: AUC={a_k:.4f}")

    logo = LeaveOneGroupOut()
    pr_logo = cross_val_predict(RandomForestClassifier(**RF_KW), X, y,
                                 cv=logo, groups=site,
                                 method="predict_proba")[:, 1]
    a_logo = roc_auc_score(y, pr_logo)
    rows_f4.append(dict(scheme="LOGO", auc=round(a_logo, 4)))
    print(f"  LOGO:   AUC={a_logo:.4f}")
    pd.DataFrame(rows_f4).to_csv(out / "cv_sensitivity.csv", index=False)
    rep["cv_sensitivity"] = rows_f4

    print(f"\n[F5] THRESHOLD SENSITIVITY")
    rows_f5 = []
    for thr in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        y_t = (p.loc[common, "poison"] > thr).astype(int)
        np_t = int(y_t.sum())
        if np_t < 5:
            continue
        pr_t = cross_val_predict(RandomForestClassifier(**RF_KW), X, y_t,
                                  cv=cv5, groups=site,
                                  method="predict_proba")[:, 1]
        a_t = roc_auc_score(y_t, pr_t)
        rows_f5.append(dict(threshold=thr, n_pos=np_t,
                            rate_pct=round(100*np_t/n, 1),
                            auc=round(a_t, 4)))
        print(f"  thr={thr:.2f}: {np_t} pos, AUC={a_t:.4f}")
    pd.DataFrame(rows_f5).to_csv(out / "threshold_sensitivity.csv", index=False)
    rep["threshold_sensitivity"] = rows_f5

    print(f"\n[F6] MISSED CATASTROPHES")
    flagged = proba_rf > 0.10
    cat = y == 1
    missed = cat & ~flagged
    rows_f6 = []
    for b in common[missed]:
        poison = p.loc[b, "poison"]
        prob = proba_rf[b]
        feat_row = dict(building_id=b, site=b.split("_")[0],
                        poison=round(poison, 3), predicted_prob=round(prob, 4))
        for col in ["early_late_mean_ratio", "early_flat_frac", "dynamic_range",
                    "level_traj_trend", "flat_spot_frac", "halves_var_logratio"]:
            if col in X.columns:
                feat_row[col] = round(float(X.loc[b, col]), 4)
        rows_f6.append(feat_row)
        print(f"  {b}: poison={poison:.3f} prob={prob:.4f}")
    pd.DataFrame(rows_f6).to_csv(out / "missed_catastrophes.csv", index=False)
    rep["missed_catastrophes"] = rows_f6

    # F7: on catastrophe buildings, split by whether Bolt zero-shot already nails them, compare arm medians
    print(f"\n[F7] DETONATION VERIFICATION")
    zs_w = None
    if zs_csv and Path(zs_csv).exists():
        zs = pd.read_csv(zs_csv)
        zs_w = zs.pivot(index="building_id", columns="model", values="nmae")
        cats = common[y == 1]
        c_zs = cats.intersection(zs_w.index)
        bolt = zs_w.loc[c_zs, "chronos_bolt_small"]
        last30 = p.loc[c_zs, "last_30"]
        full = p.loc[c_zs, "full"]

        low_bolt = bolt < 0.01
        groups = {"bolt_low (n={})".format(int(low_bolt.sum())): c_zs[low_bolt],
                  "bolt_high (n={})".format(int((~low_bolt).sum())): c_zs[~low_bolt]}
        rows_f7 = []
        for label, idx in groups.items():
            rows_f7.append(dict(
                group=label,
                bolt_median=round(float(bolt.loc[idx].median()), 4),
                last30_median=round(float(last30.loc[idx].median()), 4),
                full_median=round(float(full.loc[idx].median()), 4)))
            print(f"  {label}: bolt={bolt.loc[idx].median():.4f} "
                  f"last30={last30.loc[idx].median():.4f} "
                  f"full={full.loc[idx].median():.4f}")
        pd.DataFrame(rows_f7).to_csv(out / "detonation_verification.csv",
                                      index=False)
        rep["detonation_verification"] = rows_f7

    # F8: can the fingerprint predict which buildings fine-tuning helps (FT-best beats Bolt)
    print(f"\n[F8] FT-WINS POCKET")
    if zs_w is not None:
        c2 = common.intersection(zs_w.index)
        ftbest = p.loc[c2, ["last_30", "full"]].min(axis=1)
        bolt2 = zs_w.loc[c2, "chronos_bolt_small"]
        ft_wins = (ftbest < bolt2).astype(int)
        n_wins = int(ft_wins.sum())
        print(f"  FT-best < ZS-Bolt: {n_wins}/{len(c2)} ({100*n_wins/len(c2):.1f}%)")
        if n_wins >= 10:
            cv2 = GroupKFold(n_splits=min(5, site.loc[c2].nunique()))
            pr_w = cross_val_predict(RandomForestClassifier(**RF_KW),
                                      X.loc[c2], ft_wins, cv=cv2,
                                      groups=site.loc[c2],
                                      method="predict_proba")[:, 1]
            auc_w = roc_auc_score(ft_wins, pr_w)
            print(f"  Fingerprint AUC for FT-wins: {auc_w:.4f}")
            rep["ft_wins_pocket"] = dict(n_wins=n_wins, n_total=len(c2),
                                         auc=round(auc_w, 4))

    print(f"\n[F9] ALTERNATIVE CLASSIFIERS")
    rows_f9 = []
    rows_f9.append(dict(classifier="Random Forest", auc=round(auc_rf, 4)))
    lr = Pipeline([("scaler", StandardScaler()),
                   ("lr", LogisticRegression(class_weight="balanced",
                                             max_iter=1000, random_state=7))])
    pr_lr = cross_val_predict(lr, X, y, cv=cv5, groups=site,
                              method="predict_proba")[:, 1]
    auc_lr = roc_auc_score(y, pr_lr)
    rows_f9.append(dict(classifier="Logistic Regression", auc=round(auc_lr, 4)))
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     min_samples_leaf=10, random_state=7)
    pr_gb = cross_val_predict(gb, X, y, cv=cv5, groups=site,
                              method="predict_proba")[:, 1]
    auc_gb = roc_auc_score(y, pr_gb)
    rows_f9.append(dict(classifier="Gradient Boosting", auc=round(auc_gb, 4)))
    rows_f9.append(dict(classifier=f"Best single ({df_f1.iloc[0]['feature']})",
                        auc=round(df_f1.iloc[0]["auc_overall"], 4)))

    for r in rows_f9:
        print(f"  {r['classifier']:30s} AUC={r['auc']:.4f}")
    pd.DataFrame(rows_f9).to_csv(out / "classifier_comparison.csv", index=False)
    rep["classifier_comparison"] = rows_f9

    Path(out / "robustness_summary.json").write_text(
        json.dumps(rep, indent=1, default=str))
    print(f"[ROB] summary → {out}/robustness_summary.json")
    return rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ft_csv")
    ap.add_argument("--profiles", default="data/profiles_v2.csv")
    ap.add_argument("--early", default="data/early_features.csv")
    ap.add_argument("--zs", default="data/zs_label_matrix_dec_full.csv")
    ap.add_argument("--out_dir", default="data/robustness")
    a = ap.parse_args()
    run_robustness_suite(a.ft_csv, a.profiles, a.early, a.zs, a.out_dir)
