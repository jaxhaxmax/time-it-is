"""Loads building series from the BDG-2 wide-format electricity.csv and applies the raw-data quality filter."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


BUILDING_IDS: dict[str, str] = {
    "Fox":     "Fox_lodging_Stephen",
    "Rat":     "Rat_public_Emilee",
    "Panther": "Panther_education_Misty",
    "Bear":    "Bear_public_Orville",
}

PANTHER_FALLBACK = "Panther_parking_Lorriane"


def load_electricity_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"electricity.csv not found at: {csv_path}")

    print(f"[BDG2] Loading {csv_path} …")
    df = pd.read_csv(csv_path, index_col=0)

    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[df.index.notna()]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill().bfill()

    df = df.resample("h").mean()
    df = df.ffill().bfill()

    print(f"[BDG2] Loaded  shape={df.shape}  "
          f"date_range=[{df.index[0]} → {df.index[-1]}]")
    return df


def extract_building_series(
    df: pd.DataFrame,
    archetype: str,
    building_id: Optional[str] = None,
) -> tuple[np.ndarray, str, pd.DatetimeIndex]:
    arch = archetype.strip().title()

    if building_id is not None:
        bid = building_id
    else:
        bid = BUILDING_IDS.get(arch)
        if bid is None:
            raise ValueError(
                f"Unknown archetype '{archetype}'. "
                f"Known: {list(BUILDING_IDS.keys())}"
            )

    if bid not in df.columns:
        if arch == "Panther" and PANTHER_FALLBACK in df.columns:
            print(f"[BDG2] '{bid}' not found; using fallback '{PANTHER_FALLBACK}'")
            bid = PANTHER_FALLBACK
        else:
            available = [c for c in df.columns if arch.lower() in c.lower()]
            if available:
                print(f"[BDG2] '{bid}' not found; using first match: '{available[0]}'")
                bid = available[0]
            else:
                raise KeyError(
                    f"Building '{bid}' not found in CSV. "
                    f"Columns starting with '{arch}': {available}"
                )

    series = df[bid].copy()
    idx    = series.index
    x      = series.to_numpy(dtype=np.float64)

    print(f"[BDG2] {arch:8s} → column='{bid}'  "
          f"n={len(x):,}  mean={x.mean():.2f}  "
          f"CV={x.std()/max(abs(x.mean()),1e-6):.4f}")
    return x, bid, idx


def quality_filter(
    csv_path: str | Path,
    year: int = 2017,
    max_nan_frac: float = 0.5,
    max_zero_frac: float = 0.5,
    min_std: float = 1.0,
    save_json: Optional[str | Path] = None,
) -> list[str]:
    csv_path = Path(csv_path)
    print(f"[QC] Loading raw {csv_path} …")
    raw = pd.read_csv(csv_path, index_col=0)
    raw.index = pd.to_datetime(raw.index, utc=True, errors="coerce")
    raw = raw[raw.index.notna()].sort_index()
    raw = raw[raw.index.year == year]
    raw = raw.apply(pd.to_numeric, errors="coerce")
    n = len(raw)
    print(f"[QC] Year {year}: {n} rows × {raw.shape[1]} buildings")

    nan_frac  = raw.isna().mean()
    zero_frac = (raw == 0).sum() / raw.notna().sum().clip(lower=1)
    std       = raw.std()

    mask = (nan_frac <= max_nan_frac) & (zero_frac <= max_zero_frac) & (std >= min_std)
    clean = sorted(raw.columns[mask])
    print(f"[QC] passed: {len(clean)}  "
          f"(failed nan: {(nan_frac > max_nan_frac).sum()}, "
          f"zero: {(zero_frac > max_zero_frac).sum()}, "
          f"std: {(std < min_std).sum()})")

    if save_json is not None:
        payload = {
            "criteria": {"year": year, "max_nan_frac": max_nan_frac,
                         "max_zero_frac": max_zero_frac, "min_std": min_std},
            "n_buildings": len(clean),
            "buildings": clean,
        }
        Path(save_json).write_text(json.dumps(payload, indent=1))
        print(f"[QC] saved → {save_json}")
    return clean


def load_all_four(
    csv_path: str | Path,
    building_ids: Optional[dict[str, str]] = None,
) -> dict[str, tuple[np.ndarray, str, pd.DatetimeIndex]]:
    df   = load_electricity_csv(csv_path)
    out  = {}
    bids = building_ids or {}
    for arch in ("Fox", "Rat", "Panther", "Bear"):
        bid = bids.get(arch)
        out[arch] = extract_building_series(df, arch, building_id=bid)
    return out
