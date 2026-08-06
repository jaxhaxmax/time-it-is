"""P3 adaptation study. for every pre-registered building x arm (zeroshot | first_15 | last_15 | last_30 | full) fine-tune the domain FM fresh from checkpoint on the slice, evaluate day-ahead on the Dec test window under the frozen protocol, and write a resume-safe JSON per (arm, building), same checkpointing discipline as the zero-shot harness.

cost is ~45 buildings x 4 FT arms at 1-4 min each on a T4, one session. always zip and download out_dir before the session ends, Kaggle has no persistent disk."""

from __future__ import annotations
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metrics import score_origins
from src.finetune_patchtst import finetune_on_slice, RECIPE, SLICES
from src.patchtst_900k import load_patchtst_900k
from experiments.zeroshot_eval import EvalConfig, build_origins

ARMS = ("zeroshot",) + SLICES


def _eval_model(model, x: np.ndarray, cfg: EvalConfig) -> dict:
    ctxs, tgts = build_origins(x, cfg)
    xs = ctxs[:, -model.context_len:]
    dev = next(model.parameters()).device
    with torch.no_grad():
        t = torch.tensor(xs, dtype=torch.float32, device=dev).unsqueeze(-1)
        out = model(t)[:, :cfg.horizon, 0].float().cpu().numpy()
    return score_origins(tgts, out)


def run_adaptation_study(df: pd.DataFrame,
                         prereg_json: str | Path,
                         checkpoint: str | Path,
                         out_dir: str | Path = "ft_results",
                         cfg: EvalConfig = EvalConfig(),
                         arms: tuple = ARMS,
                         buildings: list[str] | None = None) -> None:
    out_dir = Path(out_dir)
    reg = json.loads(Path(prereg_json).read_text())
    blds = buildings or [b["building_id"] for b in reg["buildings"]]
    print(f"[P3] {len(blds)} buildings × {len(arms)} arms "
          f"(prereg sha {reg.get('sha256','?')[:12]})")

    for arm in arms:
        arm_dir = out_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        done = {p.stem for p in arm_dir.glob("*.json")}
        todo = [b for b in blds if b not in done]
        print(f"[P3] arm={arm}: {len(done)} done, {len(todo)} to run")

        for i, bid in enumerate(todo, 1):
            t0 = time.time()
            try:
                x = df[bid].to_numpy(np.float64)
                trace = None
                if arm == "zeroshot":
                    model = load_patchtst_900k(str(checkpoint))
                else:
                    model, trace = finetune_on_slice(
                        str(checkpoint), x, arm,
                        test_hours=cfg.test_hours,
                        end_hour=cfg.end_hour)
                res = _eval_model(model, x, cfg)
                res.update({"building_id": bid, "arm": arm,
                            "ft_loss_trace": trace,
                            "recipe": RECIPE if arm != "zeroshot" else None,
                            "cfg": asdict(cfg) | {"quantile_levels":
                                                  list(cfg.quantile_levels)}})
                tmp = arm_dir / f"{bid}.json.tmp"
                tmp.write_text(json.dumps(res))
                tmp.rename(arm_dir / f"{bid}.json")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()      # free the FT copy before the next building
                print(f"[P3] {arm} {i}/{len(todo)} {bid}: "
                      f"nmae={res['agg']['nmae']:.4f} "
                      f"({time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"[P3][FAIL] {arm}/{bid}: {e}")


def consolidate_ft(out_dir: str | Path,
                   out_csv: str = "ft_results_p3_5arm.csv",
                   save_csv: bool = True) -> pd.DataFrame:
    out_dir = Path(out_dir)
    rows = []
    for f in sorted(out_dir.glob("*/*.json")):
        r = json.loads(f.read_text())
        rows.append({"building_id": r["building_id"], "arm": r["arm"],
                     "n_origins": r["n_origins"], **r["agg"]})
    mat = pd.DataFrame(rows)
    if save_csv and len(mat):
        path = out_dir / out_csv
        mat.to_csv(path, index=False)
        print(f"[P3] FT matrix: {mat.shape[0]} rows → {path}")
        piv = mat.pivot(index="building_id", columns="arm", values="nmae")
        print(piv.describe().loc[["mean", "50%"]].round(4).to_string())
    return mat
