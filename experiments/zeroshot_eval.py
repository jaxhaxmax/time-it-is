"""Zero-shot label-matrix harness (P0). runs every model adapter over every clean building on a rolling day-ahead protocol and writes checkpointed per-building JSON that survives Kaggle session death.

frozen protocol, do not edit after the first real run: test window is the last 28 days of 2017 (672 hours), horizon 24 h day-ahead with one origin per day at 00:00, 512 h context before each origin, 28 origins per building, metrics from src.metrics (NMAE primary, then NRMSE, MAPE, nCRPS).

checkpointing: one JSON per (building, model) at <out_dir>/<model>/<building_id>.json, re-running skips existing files so it's safe to resume. Kaggle has no persistent disk so download or commit <out_dir> periodically."""

from __future__ import annotations
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metrics import score_origins  # single source of metric truth


@dataclass(frozen=True)
class EvalConfig:
    context_length: int = 512
    horizon: int = 24
    test_days: int = 28
    num_samples: int = 20                      # Chronos-T5 sample paths
    quantile_levels: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    # test window ends at this index (exclusive), None = end of series. Dec window: None. Oct 2017 window: 7296 (= 304 days * 24, Oct 31 23:00)
    end_hour: Optional[int] = None

    @property
    def test_hours(self) -> int:
        return self.test_days * self.horizon   # 672


@dataclass
class ForecastResult:
    # one batched forecast, N origins for a single building
    point: np.ndarray                          # (N, H) median / point forecast
    samples: Optional[np.ndarray] = None       # (N, S, H)
    quants: Optional[np.ndarray] = None        # (N, Q, H)
    levels: Optional[np.ndarray] = None        # (Q,)


class ModelAdapter(Protocol):
    # every model wraps to this, contexts is float64 (N, L) raw kWh
    name: str
    def predict(self, contexts: np.ndarray, horizon: int) -> ForecastResult: ...


class SeasonalNaiveAdapter:
    # x̂[t] = x[t-168], the baseline every FM has to beat to matter
    name = "snaive168"

    def predict(self, contexts: np.ndarray, horizon: int) -> ForecastResult:
        if contexts.shape[1] < 168:
            raise ValueError("context shorter than 168 h")
        # forecast hour h of the next day = context[-168 + h]
        point = np.stack([contexts[:, -168 + h] for h in range(horizon)], axis=1)
        return ForecastResult(point=point)


class ChronosAdapter:
    # wraps chronos-t5-* (sample-based) and chronos-bolt-* (quantile-based) via the official chronos-forecasting package
    def __init__(self, model_id: str, device: str | None = None,
                 num_samples: int = 20,
                 quantile_levels: tuple = (0.1, 0.2, 0.3, 0.4, 0.5,
                                           0.6, 0.7, 0.8, 0.9)):
        import torch
        from chronos import BaseChronosPipeline
        self.torch = torch
        self.model_id = model_id
        self.name = model_id.split("/")[-1].replace("-", "_")
        self.is_bolt = "bolt" in model_id.lower()
        self.num_samples = num_samples
        self.levels = np.asarray(quantile_levels, float)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        print(f"[ZS] Loading {model_id} on {dev} …")
        self.pipe = BaseChronosPipeline.from_pretrained(
            model_id, device_map=dev, torch_dtype=dtype)

    def predict(self, contexts: np.ndarray, horizon: int) -> ForecastResult:
        ctx = [self.torch.tensor(c, dtype=self.torch.float32) for c in contexts]
        if self.is_bolt:
            # positional first arg: named context in chronos 1.x, inputs in newer releases, positional works in both
            q, mean = self.pipe.predict_quantiles(
                ctx, prediction_length=horizon,
                quantile_levels=list(self.levels))
            q = q.float().cpu().numpy()               # (N, H, Q)
            quants = np.transpose(q, (0, 2, 1))       # (N, Q, H)
            med = quants[:, np.searchsorted(self.levels, 0.5), :]
            return ForecastResult(point=med, quants=quants, levels=self.levels)
        fc = self.pipe.predict(ctx, prediction_length=horizon,
                               num_samples=self.num_samples)
        samples = fc.float().cpu().numpy().astype(np.float64)  # (N, S, H)
        point = np.median(samples, axis=1)
        return ForecastResult(point=point, samples=samples)


class PatchTSTAdapter:
    # domain FM, the verified v2_Mature checkpoint (43 keys, strict load). native 512 h -> 96 h, harness uses the first 24 h for day-ahead, RevIN denorm is internal so raw kWh in and out
    name = "patchtst_b900k"

    def __init__(self, checkpoint_path: str, device: str | None = None,
                 batch_size: int = 256):
        import torch
        from src.patchtst_900k import load_patchtst_900k
        self.torch = torch
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_patchtst_900k(checkpoint_path, device=self.device)

    def predict(self, contexts: np.ndarray, horizon: int) -> ForecastResult:
        if horizon > self.model.pred_len:
            raise ValueError(f"horizon {horizon} > native {self.model.pred_len}")
        x = contexts[:, -self.model.context_len:]
        outs = []
        with self.torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                t = self.torch.tensor(x[i:i + self.batch_size],
                                      dtype=self.torch.float32,
                                      device=self.device).unsqueeze(-1)
                o = self.model(t)                      # (b, 96, 1)
                outs.append(o[:, :horizon, 0].float().cpu().numpy())
        return ForecastResult(point=np.concatenate(outs, axis=0))


def build_origins(x: np.ndarray, cfg: EvalConfig
                  ) -> tuple[np.ndarray, np.ndarray]:
    # slice one building's 2017 hourly array into rolling day-ahead origins. x is (8760,), hour 0 = Jan 1 00:00. returns contexts (test_days, context_length) and targets (test_days, horizon)
    n = len(x)
    end = n if cfg.end_hour is None else cfg.end_hour
    if not (0 < end <= n):
        raise ValueError(f"end_hour {cfg.end_hour} outside series of length {n}")
    need = cfg.context_length + cfg.test_hours
    if end < need:
        raise ValueError(f"series too short: end={end} < {need}")
    test_start = end - cfg.test_hours        # aligned to 00:00 when end % 24 == 0
    ctxs, tgts = [], []
    for d in range(cfg.test_days):
        o = test_start + d * cfg.horizon     # origin hour
        ctxs.append(x[o - cfg.context_length:o])
        tgts.append(x[o:o + cfg.horizon])
    return np.asarray(ctxs, float), np.asarray(tgts, float)


def run_zeroshot_eval(df: pd.DataFrame,
                      buildings: list[str],
                      adapters: list,
                      out_dir: str | Path,
                      cfg: EvalConfig = EvalConfig(),
                      progress_every: int = 50) -> None:
    # df is the 2017 hourly wide frame (rows hours, cols building_ids), buildings the 1,447 clean IDs or any subset, one resume-safe JSON per (model, building)
    out_dir = Path(out_dir)
    if len(df) % 24 != 0:
        print(f"[ZS][WARN] df has {len(df)} rows (not divisible by 24) — "
              f"origins may not align to 00:00. Check the 2017 slice.")

    for adapter in adapters:
        model_dir = out_dir / adapter.name
        model_dir.mkdir(parents=True, exist_ok=True)
        done = {p.stem for p in model_dir.glob("*.json")}
        todo = [b for b in buildings if b not in done]
        print(f"[ZS] {adapter.name}: {len(done)} done, {len(todo)} to run")

        t0, n_fail = time.time(), 0
        for i, bid in enumerate(todo, 1):
            try:
                x = df[bid].to_numpy(dtype=np.float64)
                ctxs, tgts = build_origins(x, cfg)
                fc = adapter.predict(ctxs, cfg.horizon)
                res = score_origins(tgts, fc.point,
                                    samples=fc.samples,
                                    quants=fc.quants, levels=fc.levels)
                res.update({"building_id": bid, "model": adapter.name,
                            "cfg": asdict(cfg) | {"quantile_levels":
                                                  list(cfg.quantile_levels)}})
                tmp = model_dir / f"{bid}.json.tmp"
                tmp.write_text(json.dumps(res))
                tmp.rename(model_dir / f"{bid}.json")   # atomic-ish checkpoint via tmp then rename
            except Exception as e:
                n_fail += 1
                print(f"[ZS][FAIL] {adapter.name} / {bid}: {e}")
            if i % progress_every == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-9)
                eta = (len(todo) - i) / max(rate, 1e-9)
                print(f"[ZS] {adapter.name}  {i}/{len(todo)}  "
                      f"({rate:.1f} bld/s, ETA {eta/60:.1f} min, "
                      f"fails={n_fail})")
        print(f"[ZS] {adapter.name} complete in "
              f"{(time.time()-t0)/60:.1f} min, fails={n_fail}")


def consolidate_results(out_dir: str | Path,
                        save_csv: bool = True) -> pd.DataFrame:
    # collect the per-building JSONs into the label matrix, one row per (building, model) with aggregate metrics
    out_dir = Path(out_dir)
    rows = []
    for f in sorted(out_dir.glob("*/*.json")):
        r = json.loads(f.read_text())
        rows.append({"building_id": r["building_id"], "model": r["model"],
                     "n_origins": r["n_origins"], **r["agg"]})
    mat = pd.DataFrame(rows)
    if save_csv and len(mat):
        path = out_dir / "zs_label_matrix.csv"
        mat.to_csv(path, index=False)
        print(f"[ZS] label matrix: {mat.shape[0]} rows → {path}")
    return mat


def smoke_test(df: pd.DataFrame, out_dir: str | Path = "/tmp/zs_smoke") -> pd.DataFrame:
    # seasonal naive on the 4 calibration buildings before burning quota. expect Fox (stable hotel) low, Rat (rigid schedule) low-moderate, Panther (drifting) moderate, Bear (volatile) highest. wrong ordering means the protocol or slice is broken, fix it first
    from src.bdg2_loader import BUILDING_IDS
    buildings = [b for b in BUILDING_IDS.values() if b in df.columns]
    run_zeroshot_eval(df, buildings, [SeasonalNaiveAdapter()], out_dir)
    mat = consolidate_results(out_dir)
    print(mat[["building_id", "model", "nmae", "nrmse", "mape"]]
          .to_string(index=False))
    return mat
