"""Frozen metric definitions shared by every experiment. Scripts import from here and never redefine a metric locally, so changing this file after results exist means re-running everything that produced them.

NMAE is the primary metric (MAE normalised by mean absolute ground truth over the test window). NRMSE and MAPE are secondary, and nCRPS covers the probabilistic models. MAPE is epsilon-guarded because near-zero overnight loads inflate a per-hour percentage error arbitrarily.
"""

from __future__ import annotations
import numpy as np

EPS_FRAC = 0.01


def nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE normalised by mean absolute ground truth over the window."""
    y_true, y_pred = np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel()
    denom = np.mean(np.abs(y_true))
    if denom < 1e-9:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE normalised by mean absolute ground truth, matching BuildingsBench."""
    y_true, y_pred = np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel()
    denom = np.mean(np.abs(y_true))
    if denom < 1e-9:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / denom)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Epsilon-guarded MAPE in percent, reported as a secondary metric only."""
    y_true, y_pred = np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel()
    eps = EPS_FRAC * max(np.mean(np.abs(y_true)), 1e-9)
    denom = np.maximum(np.abs(y_true), eps)
    return float(100.0 * np.mean(np.abs(y_true - y_pred) / denom))


def crps_samples(y_true: np.ndarray, samples: np.ndarray) -> float:
    """Sample-based nCRPS for one origin, using E|X−y| − 0.5 E|X−X'| with the sorted-sample identity for the second term, normalised by mean |y_true|."""
    y_true = np.asarray(y_true, float).ravel()
    samples = np.asarray(samples, float)
    if samples.ndim != 2 or samples.shape[1] != y_true.shape[0]:
        raise ValueError(f"samples shape {samples.shape} vs y_true {y_true.shape}")
    term1 = np.mean(np.abs(samples - y_true[None, :]))
    s_sorted = np.sort(samples, axis=0)
    S = s_sorted.shape[0]
    coeffs = (2 * np.arange(S) - S + 1)[:, None]
    term2 = np.mean(2.0 * np.sum(coeffs * s_sorted, axis=0) / (S * S))
    crps = term1 - 0.5 * term2
    denom = max(np.mean(np.abs(y_true)), 1e-9)
    return float(crps / denom)


def crps_quantiles(y_true: np.ndarray, quants: np.ndarray,
                   levels: np.ndarray) -> float:
    """Quantile-approximated nCRPS via mean pinball loss × 2, for models that emit quantiles such as Chronos-Bolt."""
    y_true = np.asarray(y_true, float).ravel()
    quants = np.asarray(quants, float)
    levels = np.asarray(levels, float).ravel()
    diff = y_true[None, :] - quants
    pinball = np.maximum(levels[:, None] * diff, (levels[:, None] - 1) * diff)
    denom = max(np.mean(np.abs(y_true)), 1e-9)
    return float(2.0 * np.mean(pinball) / denom)


def score_origins(y_true: np.ndarray, y_pred: np.ndarray,
                  samples: np.ndarray | None = None,
                  quants: np.ndarray | None = None,
                  levels: np.ndarray | None = None) -> dict:
    """Score a full rolling-origin evaluation for one building and model, returning per-origin lists for downstream bootstrap and paired tests alongside the window aggregates."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError(f"shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
    N = y_true.shape[0]

    per = {"nmae": [], "nrmse": [], "mape": [], "ncrps": []}
    for i in range(N):
        per["nmae"].append(nmae(y_true[i], y_pred[i]))
        per["nrmse"].append(nrmse(y_true[i], y_pred[i]))
        per["mape"].append(mape(y_true[i], y_pred[i]))
        if samples is not None:
            per["ncrps"].append(crps_samples(y_true[i], samples[i]))
        elif quants is not None and levels is not None:
            per["ncrps"].append(crps_quantiles(y_true[i], quants[i], levels))
    if not per["ncrps"]:
        per.pop("ncrps")

    out = {
        "n_origins": N,
        "agg": {
            "nmae":  nmae(y_true, y_pred),
            "nrmse": nrmse(y_true, y_pred),
            "mape":  mape(y_true, y_pred),
        },
        "per_origin": {k: [round(v, 6) for v in vals] for k, vals in per.items()},
    }
    if "ncrps" in per:
        out["agg"]["ncrps"] = float(np.mean(per["ncrps"]))
    return out


def bootstrap_ci(per_origin_values, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 7) -> tuple[float, float]:
    """Percentile bootstrap confidence interval over forecast origins."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(per_origin_values, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    boots = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))
