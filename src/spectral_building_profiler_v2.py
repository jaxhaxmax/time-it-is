"""Expanded building fingerprint (P1), 35 features, supersedes the 11-dim v1 vector for router consumption. v1 stays untouched so earlier results still reproduce. drift signals are folded in as continuous features here, the router doesn't need a separate detector output. numpy/scipy/pandas only, all CPU-cheap (<50 ms/building).

groups: legacy (cv, ac_24, ac_168, spectral_entropy, baseload_ratio, peak_to_trough, ramp_steepness, dominant_period, energy_ratio), decomposition (trend + 24 h + 168 h strengths, Hyndman-style), calendar (weekend_weekday_ratio, dow_range, needs a start timestamp, defaults to 2017-01-01 for the BDG-2 slice), day-shape (profile_consistency mean/std, IS, IV), variability (stability, lumpiness), complexity (permutation_entropy m=4, dfa_alpha), distribution (skew, kurtosis, dynamic_range, flat_spot_frac), harmonics (harmonic_ratio, n_spectral_peaks), annual (annual_amplitude as a weather proxy), drift (ac_divergence, trend_shift, band24_traj_std/trend, level_traj_trend, halves_var_logratio)."""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy import stats as spst

EPS = 1e-9

FEATURE_NAMES = [
    "cv", "ac_24", "ac_168", "dominant_period", "energy_ratio",
    "spectral_entropy", "baseload_ratio", "peak_to_trough", "ramp_steepness",
    "trend_strength", "seasonal_strength_24", "seasonal_strength_168",
    "weekend_weekday_ratio", "dow_range",
    "profile_consistency_mean", "profile_consistency_std",
    "interdaily_stability", "intradaily_variability",
    "stability", "lumpiness",
    "permutation_entropy", "dfa_alpha",
    "skewness", "kurtosis", "dynamic_range", "flat_spot_frac",
    "harmonic_ratio", "n_spectral_peaks",
    "annual_amplitude",
    "ac_divergence", "trend_shift", "band24_traj_std", "band24_traj_trend",
    "level_traj_trend", "halves_var_logratio",
]


def _ac(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 1:
        return 0.0
    a, b = x[:-lag], x[lag:]
    sa, sb = a.std(), b.std()
    if sa < EPS or sb < EPS:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _linslope_norm(y: np.ndarray) -> float:
    # OLS slope of y vs index as total change over the series relative to |mean|, scale-free
    n = len(y)
    if n < 3 or np.all(~np.isfinite(y)):
        return 0.0
    t = np.arange(n, dtype=float)
    m = np.isfinite(y)
    if m.sum() < 3:
        return 0.0
    slope = np.polyfit(t[m], y[m], 1)[0]
    denom = max(abs(np.nanmean(y)), EPS)
    return float(slope * n / denom)


def _legacy_block(x: np.ndarray) -> dict:
    mu = x.mean()
    cv = x.std() / max(abs(mu), EPS)

    # Welch PSD, fs in cycles/hour so periods come out in hours
    nper = min(len(x), 1024)
    f, p = sps.welch(x - mu, fs=1.0, nperseg=nper)
    f, p = f[1:], p[1:]                                  # drop DC
    period = 1.0 / f
    band = (period >= 2) & (period <= 200)
    fb, pb = f[band], p[band]
    if len(pb):
        k = int(np.argmax(pb))
        dominant_period = float(1.0 / fb[k])
        lo, hi = max(k - 1, 0), min(k + 2, len(pb))
        energy_ratio = float(pb[lo:hi].sum() / max(p.sum(), EPS))
    else:
        dominant_period, energy_ratio = 0.0, 0.0
    pn = p / max(p.sum(), EPS)
    spectral_entropy = float(-(pn * np.log(pn + EPS)).sum() / np.log(len(pn)))

    days = x[: (len(x) // 24) * 24].reshape(-1, 24)
    med_prof = np.median(days, axis=0)
    daily_mean = max(med_prof.mean(), EPS)
    baseload_ratio = float(med_prof[0:6].mean() / daily_mean)
    peak_to_trough = float(med_prof.max() / max(med_prof.min(), EPS))
    ramp_steepness = float(np.abs(np.diff(med_prof)).max() / daily_mean)

    return dict(cv=float(cv), ac_24=_ac(x, 24), ac_168=_ac(x, 168),
                dominant_period=dominant_period, energy_ratio=energy_ratio,
                spectral_entropy=spectral_entropy, baseload_ratio=baseload_ratio,
                peak_to_trough=peak_to_trough, ramp_steepness=ramp_steepness)


def _strength(remainder: np.ndarray, other: np.ndarray) -> float:
    # Hyndman strength: 1 - Var(remainder)/Var(remainder + component)
    v = np.var(remainder + other)
    if v < EPS:
        return 0.0
    return float(np.clip(1.0 - np.var(remainder) / v, 0.0, 1.0))


def _decomposition_block(x: np.ndarray) -> dict:
    n = len(x)
    # trend: centred 169 h moving average, covers one weekly cycle
    trend = pd.Series(x).rolling(169, center=True, min_periods=85).mean().to_numpy()
    detr = x - trend
    valid = np.isfinite(detr)

    hod = np.arange(n) % 24
    s24 = np.zeros(n)
    for h in range(24):
        m = valid & (hod == h)
        s24[hod == h] = detr[m].mean() if m.any() else 0.0
    resid1 = detr - s24

    how = np.arange(n) % 168
    s168 = np.zeros(n)
    for h in range(168):
        m = valid & (how == h)
        s168[how == h] = resid1[m].mean() if m.any() else 0.0
    remainder = resid1 - s168

    r = remainder[valid]
    return dict(
        trend_strength=_strength(r, (trend - np.nanmean(trend))[valid]),
        seasonal_strength_24=_strength(r, s24[valid]),
        seasonal_strength_168=_strength(r, s168[valid]),
    )


def _calendar_block(x: np.ndarray, start: pd.Timestamp) -> dict:
    idx = pd.date_range(start, periods=len(x), freq="h")
    dow = idx.dayofweek.to_numpy()
    mu = max(abs(x.mean()), EPS)
    we, wd = x[dow >= 5], x[dow < 5]
    ratio = float(we.mean() / max(wd.mean(), EPS)) if len(we) and len(wd) else 1.0
    dmeans = np.array([x[dow == d].mean() for d in range(7) if (dow == d).any()])
    dow_range = float((dmeans.max() - dmeans.min()) / mu) if len(dmeans) else 0.0
    return dict(weekend_weekday_ratio=ratio, dow_range=dow_range)


def _dayshape_block(x: np.ndarray) -> dict:
    days = x[: (len(x) // 24) * 24].reshape(-1, 24)
    # shape consistency has to be level-immune: z-score each day first, else regime shifts (mid-year level steps) corrupt the median reference profile
    mu_d = days.mean(axis=1, keepdims=True)
    sd_d = days.std(axis=1, keepdims=True)
    ok = sd_d[:, 0] > EPS
    z = (days[ok] - mu_d[ok]) / sd_d[ok]
    if len(z) >= 14:
        ref = np.median(z, axis=0)
        rs = ref.std()
        cors = np.array([np.corrcoef(d, ref)[0, 1] if rs > EPS else 0.0
                         for d in z])
    else:
        cors = np.zeros(1)

    # actigraphy IS / IV on the raw scale, standard definitions
    mu, var = x.mean(), x.var()
    hod_means = days.mean(axis=0)
    IS = float(((hod_means - mu) ** 2).mean() / max(var, EPS))
    IV = float((np.diff(x) ** 2).mean() / max(var, EPS))
    return dict(profile_consistency_mean=float(np.nanmean(cors)),
                profile_consistency_std=float(np.nanstd(cors)),
                interdaily_stability=IS, intradaily_variability=IV)


def _variability_block(x: np.ndarray) -> dict:
    z = (x - x.mean()) / max(x.std(), EPS)
    w = 168
    nb = len(z) // w
    tiles = z[: nb * w].reshape(nb, w)
    return dict(stability=float(tiles.mean(axis=1).var()),
                lumpiness=float(tiles.var(axis=1).var()))


def _complexity_block(x: np.ndarray) -> dict:
    # permutation entropy m=4, tau=3 (tau>1 avoids noise saturation)
    m, tau = 4, 3
    span = (m - 1) * tau
    em = np.stack([x[i * tau: len(x) - span + i * tau] for i in range(m)], axis=1)
    patterns = np.argsort(em, axis=1)
    codes = patterns @ np.array([64, 16, 4, 1])
    _, counts = np.unique(codes, return_counts=True)
    pr = counts / counts.sum()
    pe = float(-(pr * np.log(pr)).sum() / np.log(24))

    y = np.cumsum(x - x.mean())
    scales = np.unique(np.logspace(np.log10(16), np.log10(1024), 12).astype(int))
    flucts = []
    for s in scales:
        nseg = len(y) // s
        if nseg < 4:
            continue
        segs = y[: nseg * s].reshape(nseg, s)
        t = np.arange(s)
        # detrend each segment linearly, vectorised polyfit
        tm = t - t.mean()
        beta = (segs @ tm) / (tm @ tm)
        resid = segs - segs.mean(axis=1, keepdims=True) - np.outer(beta, tm)
        flucts.append((s, np.sqrt((resid ** 2).mean())))
    if len(flucts) >= 4:
        ls = np.log([f[0] for f in flucts]); lf = np.log([f[1] + EPS for f in flucts])
        alpha = float(np.polyfit(ls, lf, 1)[0])
    else:
        alpha = 0.5
    return dict(permutation_entropy=pe, dfa_alpha=alpha)


def _distribution_block(x: np.ndarray) -> dict:
    p995, p50, p5 = np.percentile(x, [99.5, 50, 5])
    dyn = float(np.log((p995 + EPS) / (max(p5, 0) + EPS)))
    # flat spots: longest run of |diff| < 1% of std, as a fraction of the series
    small = np.abs(np.diff(x)) < 0.01 * max(x.std(), EPS)
    runs, cur = 0, 0
    for s in small:
        cur = cur + 1 if s else 0
        runs = max(runs, cur)
    return dict(skewness=float(spst.skew(x)), kurtosis=float(spst.kurtosis(x)),
                dynamic_range=dyn, flat_spot_frac=float(runs / len(x)))


def _harmonics_block(x: np.ndarray) -> dict:
    xd = x - x.mean()
    p = np.abs(np.fft.rfft(xd)) ** 2
    freqs = np.fft.rfftfreq(len(xd), d=1.0)

    def band_power(period, tol=0.06):
        target = 1.0 / period
        m = np.abs(freqs - target) <= target * tol
        return p[m].sum()

    p24 = band_power(24)
    # regularise: without daily seasonality p24->0 and the raw ratio explodes
    harm = (band_power(12) + band_power(8) + band_power(6)) / (p24 + 0.01 * p.sum())

    f, pw = sps.welch(xd, fs=1.0, nperseg=min(len(x), 1024))
    peaks, _ = sps.find_peaks(pw, prominence=np.median(pw) * 10)
    return dict(harmonic_ratio=float(harm), n_spectral_peaks=float(len(peaks)))


def _annual_block(x: np.ndarray, start: pd.Timestamp) -> dict:
    idx = pd.date_range(start, periods=len(x), freq="h")
    monthly = pd.Series(x, index=idx).resample("MS").mean()
    mu = max(abs(x.mean()), EPS)
    return dict(annual_amplitude=float((monthly.max() - monthly.min()) / mu))


def _drift_block(x: np.ndarray) -> dict:
    h = len(x) // 2
    a, b = x[:h], x[h:]
    ac_div = abs(_ac(a, 24) - _ac(b, 24)) + abs(_ac(a, 168) - _ac(b, 168))
    rm = pd.Series(x).rolling(168, min_periods=84).mean().to_numpy()
    mu = max(abs(x.mean()), EPS)
    trend_shift = abs(np.nanmean(rm[h:]) - np.nanmean(rm[:h])) / mu

    # weekly spectral trajectory: fraction of weekly-window power in the 24 h band
    w = 168
    nb = len(x) // w
    fracs, levels = [], []
    for i in range(nb):
        seg = x[i * w:(i + 1) * w]
        segd = seg - seg.mean()
        ps = np.abs(np.fft.rfft(segd)) ** 2
        fr = np.fft.rfftfreq(w, d=1.0)
        m24 = np.abs(fr - 1 / 24) <= (1 / 24) * 0.25
        fracs.append(ps[m24].sum() / max(ps.sum(), EPS))
        levels.append(seg.mean())
    fracs, levels = np.asarray(fracs), np.asarray(levels)

    va, vb = a.var(), b.var()
    return dict(ac_divergence=float(ac_div), trend_shift=float(trend_shift),
                band24_traj_std=float(fracs.std()),
                band24_traj_trend=_linslope_norm(fracs),
                level_traj_trend=_linslope_norm(levels),
                halves_var_logratio=float(np.log((vb + EPS) / (va + EPS))))


def profile_building_v2(x: np.ndarray,
                        start: str | pd.Timestamp = "2017-01-01") -> dict:
    # full v2 feature vector for one building. x is hourly kWh (~8760, min 14 days), start is the timestamp of x[0] for the calendar features
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 336:
        raise ValueError(f"series too short for v2 profile: {len(x)}")
    if not np.all(np.isfinite(x)):
        x = pd.Series(x).ffill().bfill().to_numpy()
    start = pd.Timestamp(start)

    feats: dict[str, float] = {}
    feats.update(_legacy_block(x))
    feats.update(_decomposition_block(x))
    feats.update(_calendar_block(x, start))
    feats.update(_dayshape_block(x))
    feats.update(_variability_block(x))
    feats.update(_complexity_block(x))
    feats.update(_distribution_block(x))
    feats.update(_harmonics_block(x))
    feats.update(_annual_block(x, start))
    feats.update(_drift_block(x))

    ordered = {k: feats[k] for k in FEATURE_NAMES}
    assert len(ordered) == len(FEATURE_NAMES)
    return ordered
