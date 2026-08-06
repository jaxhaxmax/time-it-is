"""P2: recurring-seasonality vs true-drift disambiguation using 2016. the drift detector only sees 2017, so a school's semester break inside that window looks like drift even though it recurs every year. this scores, per building, how similar the 2017 within-year trajectory is to 2016's. drift signals with high recurrence mean RECURRING_SEASONAL, drift signals with low recurrence mean TRUE_DRIFT.

both recurrence scores are Pearson r of z-scored 52-week trajectories: recurrence_level on the weekly mean-load trajectory, recurrence_band24 on the weekly 24 h spectral-power-fraction trajectory (the frequency-drift analogue).

classification, thresholds documented and sensitivity reported in output: drift_fired = ac_divergence > 0.15 or trend_shift > 0.16 (locked v1 Option-B values applied to the v2 continuous features). RECURRING_SEASONAL if drift_fired and max(recurrence) >= 0.5, TRUE_DRIFT if drift_fired and max(recurrence) < 0.5, STABLE if not drift_fired, UNKNOWN_2016 if 2016 is too incomplete to judge (nan_frac > 0.5)."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

REC_THRESHOLD = 0.5
AC_DIV_THR, TREND_THR = 0.15, 0.16        # locked v1 Option-B values


def _weekly_trajectories(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 52-point weekly mean-level and 24 h-band-fraction trajectories
    w = 168
    nb = min(len(x) // w, 52)
    levels, fracs = np.full(nb, np.nan), np.full(nb, np.nan)
    for i in range(nb):
        seg = x[i * w:(i + 1) * w]
        if np.isnan(seg).mean() > 0.5:
            continue
        seg = pd.Series(seg).ffill().bfill().to_numpy()
        levels[i] = seg.mean()
        sd = seg - seg.mean()
        ps = np.abs(np.fft.rfft(sd)) ** 2
        fr = np.fft.rfftfreq(w, d=1.0)
        m24 = np.abs(fr - 1 / 24) <= (1 / 24) * 0.25
        tot = ps.sum()
        fracs[i] = ps[m24].sum() / tot if tot > 0 else np.nan
    return levels, fracs


def _zcorr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:                      # need 20 overlapping weeks for a stable r
        return np.nan
    a, b = a[m], b[m]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef((a - a.mean()) / a.std(),
                             (b - b.mean()) / b.std())[0, 1])


def run_p2_recurrence(csv_path: str | Path,
                      profiles_csv: str | Path,
                      out_csv: str | Path = "drift_recurrence.csv",
                      progress_every: int = 200) -> pd.DataFrame:
    profs = pd.read_csv(profiles_csv).set_index("building_id")
    buildings = list(profs.index)

    print(f"[P2] Loading raw csv (both years) …")
    raw = pd.read_csv(csv_path, index_col=0)
    raw.index = pd.to_datetime(raw.index, utc=True, errors="coerce")
    raw = raw[raw.index.notna()].sort_index()
    raw = raw.apply(pd.to_numeric, errors="coerce")
    y16 = raw[raw.index.year == 2016]
    y17 = raw[raw.index.year == 2017]
    print(f"[P2] 2016: {len(y16)} rows | 2017: {len(y17)} rows | "
          f"{len(buildings)} buildings to score")

    rows = []
    for i, bid in enumerate(buildings, 1):
        x16 = y16[bid].to_numpy(np.float64)
        x17 = y17[bid].to_numpy(np.float64)
        nan16 = float(np.isnan(x16).mean())
        if nan16 > 0.5:
            rec_level = rec_band = np.nan
        else:
            l16, f16 = _weekly_trajectories(x16)
            l17, f17 = _weekly_trajectories(x17)
            rec_level = _zcorr(l16, l17)
            rec_band = _zcorr(f16, f17)

        ac_div = profs.loc[bid, "ac_divergence"]
        tshift = profs.loc[bid, "trend_shift"]
        fired = (ac_div > AC_DIV_THR) or (tshift > TREND_THR)
        rec_max = np.nanmax([rec_level, rec_band])
        if not fired:
            cls = "STABLE"
        elif nan16 > 0.5 or not np.isfinite(rec_max):
            cls = "UNKNOWN_2016"
        elif rec_max >= REC_THRESHOLD:
            cls = "RECURRING_SEASONAL"
        else:
            cls = "TRUE_DRIFT"

        rows.append(dict(building_id=bid, nan_frac_2016=round(nan16, 3),
                         recurrence_level=rec_level, recurrence_band24=rec_band,
                         ac_divergence=ac_div, trend_shift=tshift,
                         drift_fired=fired, p2_class=cls))
        if i % progress_every == 0 or i == len(buildings):
            print(f"[P2] {i}/{len(buildings)}")

    rec = pd.DataFrame(rows).set_index("building_id")
    rec.to_csv(out_csv)
    print(f"[P2] → {out_csv}")

    print("\n[P2] Class counts:")
    print(rec.p2_class.value_counts().to_string())
    btype = rec.index.to_series().str.split("_").str[1]
    print("\n[P2] Class share by building type (n≥20):")
    tab = pd.crosstab(btype, rec.p2_class, normalize="index").round(3)
    big = btype.value_counts()
    print(tab.loc[big[big >= 20].index].to_string())
    fired = rec[rec.drift_fired & rec.recurrence_level.notna()]
    if len(fired):
        print("\n[P2] Sensitivity: share RECURRING among fired at thresholds "
              "0.4/0.5/0.6 = %.2f / %.2f / %.2f" % tuple(
                  (np.nanmax(fired[["recurrence_level", "recurrence_band24"]],
                             axis=1) >= t).mean() for t in (0.4, 0.5, 0.6)))
    return rec
