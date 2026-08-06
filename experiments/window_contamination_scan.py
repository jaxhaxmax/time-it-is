"""CPU diagnostic for the window-selection limitation, uses existing data only. Part A mines ft_results_population.csv for reverse-poison buildings (full-window beat last_30, poison < -0.10). Part B scans each raw 2017 pre-test series week by week, flags anomalous weeks, groups them into contiguous segments, and classifies the contamination position as CLEAN, EARLY_EDGE, LATE_EDGE, MIDDLE (single interior segment, out of scope for the early-anchored router), MULTI_SEGMENT (disjoint segments), or WIDESPREAD (>60% of weeks weird). writes the scan CSV, the reverse-poison list, a summary with the pattern-vs-poison crosstab, and an exemplar figure of MIDDLE/MULTI buildings."""

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEST_HOURS = 28 * 24
PRETEST_HOURS = 8760 - TEST_HOURS          # 8088 hours available for training
WEEK = 168
N_WEEKS = PRETEST_HOURS // WEEK            # 48 full weeks, remainder discarded
EDGE_FRAC = 0.30                           # matches the early/late 30% framing
EARLY_EDGE_END = int(round(N_WEEKS * EDGE_FRAC))       # weeks [0, 14)
LATE_EDGE_START = N_WEEKS - EARLY_EDGE_END             # weeks [34, 48)

Z_THRESH = 3.5        # robust z on log weekly mean
FLAT_THRESH = 0.80    # fraction of near-constant hours in the week
ZERO_THRESH = 0.50    # fraction of near-zero readings
NAN_THRESH = 0.50     # fraction of missing readings
GAP_MERGE = 1         # merge anomalous segments separated by <= 1 clean week
EDGE_TOUCH_WEEKS = 2  # a segment touches an edge if within 2 weeks of it
WIDESPREAD_FRAC = 0.60

NEG_POISON_THRESH = -0.10                   # reverse of the catastrophe threshold
CATASTROPHE_THRESH = 0.10


def _detect_col(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"no {what} column in ft matrix. tried {candidates}, "
                   f"got {list(df.columns)}")


def load_poison_table(results_dir):
    path = os.path.join(results_dir, "ft_results_population.csv")
    df = pd.read_csv(path)
    bcol = _detect_col(df, ["building", "building_id", "name", "bldg",
                            "series_id"], "building")
    acol = _detect_col(df, ["arm", "strategy", "window", "config"], "arm")
    mcol = _detect_col(df, ["nmae", "NMAE", "test_nmae", "nmae_dec"], "NMAE")

    df = df[[bcol, acol, mcol]].dropna()
    df[acol] = df[acol].astype(str).str.strip().str.lower()
    df = df[df[acol].isin(["full", "last_30", "last30"])].copy()
    df[acol] = df[acol].replace({"last30": "last_30"})

    pivot = df.pivot_table(index=bcol, columns=acol, values=mcol,
                           aggfunc="first").dropna(subset=["full", "last_30"])
    pivot["poison"] = pivot["full"] - pivot["last_30"]

    def label(p):
        if p > CATASTROPHE_THRESH:
            return "catastrophe"
        if p < NEG_POISON_THRESH:
            return "reverse_poison"
        return "normal"

    pivot["label"] = pivot["poison"].apply(label)
    pivot = pivot.rename(columns={"full": "nmae_full", "last_30": "nmae_last_30"})
    pivot.index.name = "building"
    return pivot


def load_raw_2017(raw_csv, buildings=None):
    df = pd.read_csv(raw_csv)
    tcol = next((c for c in df.columns
                 if c.lower() in ("timestamp", "time", "datetime", "date")), None)
    if tcol is None:
        raise KeyError(f"no timestamp column in {raw_csv}")
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.set_index(tcol).sort_index().loc["2017-01-01":"2017-12-31"]
    if len(df) < PRETEST_HOURS:
        raise ValueError(f"2017 slice has {len(df)} rows, need >= {PRETEST_HOURS}")
    df = df.iloc[:PRETEST_HOURS]            # pre-test only, matches FT training data
    if buildings is not None:
        cols = [b for b in buildings if b in df.columns]
        missing = len(buildings) - len(cols)
        if missing:
            print(f"[warn] {missing} clean-list buildings absent from raw CSV")
        df = df[cols]
    return df


def weekly_flags(series):
    x = series.values[: N_WEEKS * WEEK].astype(float)
    weeks = x.reshape(N_WEEKS, WEEK)

    nan_frac = np.isnan(weeks).mean(axis=1)
    with np.errstate(invalid="ignore"):
        wmean = np.nanmean(weeks, axis=1)
        diffs = np.abs(np.diff(weeks, axis=1))
        scale = np.nanmedian(np.abs(x[~np.isnan(x)])) if np.any(~np.isnan(x)) else 1.0
        tiny = max(scale * 1e-3, 1e-6)
        flat_frac = np.nanmean(diffs < tiny, axis=1)
        zero_frac = np.nanmean(np.abs(weeks) < tiny, axis=1)

    # robust z on log weekly mean, log absorbs multiplicative scale shifts
    lw = np.log(np.where(wmean > 0, wmean, np.nan) + 1e-9)
    med = np.nanmedian(lw)
    mad = np.nanmedian(np.abs(lw - med))
    if not np.isfinite(mad) or mad < 1e-9:
        z = np.zeros(N_WEEKS)
    else:
        z = 0.6745 * (lw - med) / mad
    z = np.where(np.isfinite(z), z, np.inf)   # a week with no valid mean is anomalous

    flags = ((np.abs(z) > Z_THRESH) | (flat_frac > FLAT_THRESH)
             | (zero_frac > ZERO_THRESH) | (nan_frac > NAN_THRESH))
    stats = {"z": z, "flat_frac": flat_frac, "zero_frac": zero_frac,
             "nan_frac": nan_frac}
    return flags, stats


def segments_from_flags(flags, gap=GAP_MERGE):
    # contiguous flagged weeks, merging gaps <= gap, returns (start, end_inclusive)
    idx = np.where(flags)[0]
    if len(idx) == 0:
        return []
    segs = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev <= gap + 1:
            prev = i
        else:
            segs.append((start, prev))
            start = prev = i
    segs.append((start, prev))
    return segs


def classify_pattern(flags):
    frac = flags.mean()
    if frac == 0:
        return "CLEAN", []
    segs = segments_from_flags(flags)
    if frac > WIDESPREAD_FRAC:
        return "WIDESPREAD", segs

    touches_start = lambda s: s[0] <= EDGE_TOUCH_WEEKS
    touches_end = lambda s: s[1] >= N_WEEKS - 1 - EDGE_TOUCH_WEEKS
    in_early = lambda s: s[1] < EARLY_EDGE_END
    in_late = lambda s: s[0] >= LATE_EDGE_START

    all_early = all(touches_start(s) or in_early(s) for s in segs)
    all_late = all(touches_end(s) or in_late(s) for s in segs)

    if len(segs) == 1:
        s = segs[0]
        if touches_start(s) or in_early(s):
            return "EARLY_EDGE", segs
        if touches_end(s) or in_late(s):
            return "LATE_EDGE", segs
        return "MIDDLE", segs

    if all_early:
        return "EARLY_EDGE", segs
    if all_late:
        return "LATE_EDGE", segs
    return "MULTI_SEGMENT", segs


def run_scan(raw_csv, results_dir="data", out_dir="scan_out",
             clean_list_json="clean_buildings_2017.json"):
    os.makedirs(out_dir, exist_ok=True)

    poison = load_poison_table(results_dir)
    neg = poison[poison["label"] == "reverse_poison"].sort_values("poison")
    neg.to_csv(os.path.join(out_dir, "negative_poison_buildings.csv"))
    print(f"[Part A] reverse-poison (poison < {NEG_POISON_THRESH}): {len(neg)}")

    buildings = None
    for cand in (clean_list_json,
                 os.path.join(results_dir, clean_list_json),
                 os.path.join("results", clean_list_json)):
        if os.path.exists(cand):
            obj = json.load(open(cand))
            buildings = obj if isinstance(obj, list) else obj.get("buildings")
            print(f"[Part B] clean list from {cand} (n={len(buildings)})")
            break
    if buildings is None:
        print("[warn] clean_buildings_2017.json not found, scanning all columns")

    raw = load_raw_2017(raw_csv, buildings)
    print(f"[Part B] scanning {raw.shape[1]} buildings × {N_WEEKS} weeks")

    rows, flag_store = [], {}
    for b in raw.columns:
        flags, _ = weekly_flags(raw[b])
        pattern, segs = classify_pattern(flags)
        flag_store[b] = flags
        rows.append({"building": b, "pattern": pattern,
                     "n_anom_weeks": int(flags.sum()),
                     "anom_frac": round(float(flags.mean()), 3),
                     "n_segments": len(segs),
                     "segments": ";".join(f"{a}-{z}" for a, z in segs)})
    scan = pd.DataFrame(rows).set_index("building")
    scan = scan.join(poison[["poison", "label"]], how="left")
    scan.to_csv(os.path.join(out_dir, "window_contamination_scan.csv"))

    lines = ["=== PART A: reverse-poison (Level 1 evidence) ===",
             f"poison < {NEG_POISON_THRESH}: {len(neg)} buildings"]
    for b, r in neg.head(20).iterrows():
        lines.append(f"  {b}: poison={r['poison']:.3f} "
                     f"(full={r['nmae_full']:.3f}, last_30={r['nmae_last_30']:.3f})")
    if len(neg) > 20:
        lines.append(f"  ... and {len(neg) - 20} more")

    lines += ["", "=== PART B: contamination-position patterns ==="]
    counts = scan["pattern"].value_counts()
    for p in ["CLEAN", "EARLY_EDGE", "LATE_EDGE", "MIDDLE",
              "MULTI_SEGMENT", "WIDESPREAD"]:
        n = int(counts.get(p, 0))
        lines.append(f"  {p:<14} {n:>5}  ({100.0 * n / len(scan):.1f}%)")

    lines += ["", "=== Crosstab: pattern x poison label ==="]
    lines.append(pd.crosstab(scan["pattern"],
                             scan["label"].fillna("no_ft_data")).to_string())

    lines += ["", "=== Level 2/3 candidates (MIDDLE or MULTI_SEGMENT) ==="]
    lvl23 = scan[scan["pattern"].isin(["MIDDLE", "MULTI_SEGMENT"])].sort_values(
        "anom_frac", ascending=False)
    for b, r in lvl23.head(30).iterrows():
        lines.append(f"  {b}: {r['pattern']}, weeks={r['segments']}, "
                     f"poison={r['poison']}")

    summary = "\n".join(lines)
    open(os.path.join(out_dir, "scan_summary.txt"), "w").write(summary)
    print(summary)

    ex = lvl23.head(12)
    if len(ex) > 0:
        ncols = 3
        nrows = int(np.ceil(len(ex) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows),
                                 squeeze=False)
        for ax, (b, r) in zip(axes.flat, ex.iterrows()):
            s = raw[b].values[: N_WEEKS * WEEK]
            ax.plot(s, lw=0.4, color="#2166AC")
            for w in np.where(flag_store[b])[0]:
                ax.axvspan(w * WEEK, (w + 1) * WEEK, color="#B2182B",
                           alpha=0.25, lw=0)
            ax.set_title(f"{b}\n{r['pattern']}, poison={r['poison']}", fontsize=8)
            ax.tick_params(labelsize=6)
        for ax in axes.flat[len(ex):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "middle_multi_exemplars.png"), dpi=200)
        print(f"[fig] middle_multi_exemplars.png ({len(ex)} panels)")
    else:
        print("[fig] no MIDDLE/MULTI_SEGMENT buildings, no exemplar figure")

    print(f"\nAll outputs in: {out_dir}/")
    return scan


if __name__ == "__main__":
    run_scan(raw_csv="electricity.csv", results_dir="data",
             out_dir="scan_out")
