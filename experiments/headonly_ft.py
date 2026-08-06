"""Head-only fine-tuning experiment (GPU). tests whether freezing the transformer backbone and training only the projection head prevents catastrophe: if it does, the failure lives in the encoder learning ambiguous features from regime-contaminated history, not in the head. for each building it fine-tunes head-only on the full window, then compares against the existing full-param and last_30 results. runs on all catastrophes plus 50 random safe buildings (seed 42, shared with lora_ft.py)."""

from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")


def _evaluate_model(model, series_2017, test_hours=672, horizon=24,
                    n_origins=28, device="cuda"):
    test_start = len(series_2017) - test_hours
    ctx_len = model.context_len
    errors, actuals = [], []
    model.eval()
    with torch.no_grad():
        for d in range(n_origins):
            origin = test_start + d * 24
            ctx_start = origin - ctx_len
            if ctx_start < 0 or origin + horizon > len(series_2017):
                continue
            context = series_2017[ctx_start:origin]
            actual = series_2017[origin:origin + horizon]
            if np.isnan(context).sum() > ctx_len * 0.3 or np.isnan(actual).any():
                continue
            x = torch.tensor(context.reshape(1, ctx_len, 1),
                             dtype=torch.float32, device=device)
            pred = model(x)[:, :horizon, 0].cpu().numpy().flatten()
            errors.append(np.abs(pred - actual).mean())
            actuals.append(np.abs(actual).mean())
    if not actuals or np.mean(actuals) < 1e-9:
        return float("nan")
    return float(np.mean(errors) / np.mean(actuals))


def _finetune_headonly(checkpoint_path, series_2017, device="cuda",
                       test_hours=672):
    from src.patchtst_900k import load_patchtst_900k
    from src.finetune_patchtst import _make_windows, RECIPE

    test_start = len(series_2017) - test_hours
    pre = np.asarray(series_2017[:test_start], dtype=np.float64)
    model = load_patchtst_900k(checkpoint_path, device=device, eval_mode=False)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.projection_head.parameters():   # only the head trains
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    head-only: {trainable} trainable / {total} total params "
          f"({100*trainable/total:.1f}%)")

    X, Y = _make_windows(pre, model.context_len, model.pred_len, RECIPE["stride"])
    n = len(X)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=RECIPE["lr"], weight_decay=RECIPE["weight_decay"])
    lossf = torch.nn.MSELoss()

    model.train()
    order = np.arange(n)
    trace = []
    for ep in range(RECIPE["epochs"]):
        np.random.shuffle(order)
        tot = 0.0
        for i in range(0, n, RECIPE["batch_size"]):
            idx = order[i:i + RECIPE["batch_size"]]
            xb, yb = X[idx].to(device), Y[idx].to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["grad_clip"])
            opt.step()
            tot += loss.item() * len(idx)
        trace.append(round(tot / n, 6))
    model.eval()
    return model, trace


def run_headonly_experiment(
        raw_csv: str | Path,
        checkpoint_path: str | Path,
        buildings: list[str] | None = None,
        ft_results_csv: str | Path = "data/ft_results_population.csv",
        out_json: str | Path = "data/headonly_results.json",
        device: str = "cuda") -> dict:

    from src.bdg2_loader import load_electricity_csv

    print("[HEAD] loading raw data...")
    df = load_electricity_csv(raw_csv)

    ft = pd.read_csv(ft_results_csv)
    p = ft.pivot(index="building_id", columns="arm", values="nmae")
    p = p.dropna(subset=["last_30", "full"])
    p = p[p["last_30"] <= 5.0].copy()
    p["poison"] = p["full"] - p["last_30"]

    if buildings is None:
        cats = p[p["poison"] > 0.10].index.tolist()
        safe_pool = p[p["poison"] <= 0.10].index.tolist()
        np.random.seed(42)
        safe_sample = list(np.random.choice(safe_pool,
                                             size=min(50, len(safe_pool)),
                                             replace=False))
        buildings = sorted(set(cats + safe_sample))
        print(f"[HEAD] auto-selected {len(cats)} catastrophe + "
              f"{len(safe_sample)} safe = {len(buildings)} buildings")
    else:
        print(f"[HEAD] using {len(buildings)} specified buildings")

    results = {}
    ho_cats = fp_cats = l30_cats = []
    rescued = 0
    for i, bid in enumerate(buildings):
        print(f"\n[HEAD] ({i+1}/{len(buildings)}) {bid}")
        if bid not in df.columns:
            print(f"  NOT in raw data, skipping")
            continue

        series = df[bid].values.astype(np.float64)[:8760]
        is_cat = bid in p.index and p.loc[bid, "poison"] > 0.10
        existing_full = float(p.loc[bid, "full"]) if bid in p.index else None
        existing_l30 = float(p.loc[bid, "last_30"]) if bid in p.index else None

        print(f"  fine-tuning head-only (full window)...")
        try:
            model_ho, trace_ho = _finetune_headonly(
                str(checkpoint_path), series, device=device)
            nmae_ho = _evaluate_model(model_ho, series, device=device)
            print(f"  head-only NMAE: {nmae_ho:.4f}")
            del model_ho
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  head-only FAILED: {e}")
            nmae_ho = None

        results[bid] = {
            "is_catastrophe": is_cat,
            "nmae_full_param": existing_full,
            "nmae_last_30": existing_l30,
            "nmae_head_only": round(nmae_ho, 4) if nmae_ho is not None else None,
            "poison_full": round(existing_full - existing_l30, 4)
                          if existing_full and existing_l30 else None,
        }

        if nmae_ho is not None and existing_full is not None:
            if is_cat:
                # rescue = head-only lands below half the full-param error
                verdict = ("RESCUED (head-only avoids catastrophe)"
                           if nmae_ho < existing_full * 0.5
                           else "STILL DETONATES (head-only doesn't help)")
            else:
                verdict = "safe building (no catastrophe either way)"
            print(f"  full-param={existing_full:.4f} head-only={nmae_ho:.4f} "
                  f"last_30={existing_l30:.4f} → {verdict}")
            results[bid]["verdict"] = verdict

    cats_tested = {b: r for b, r in results.items() if r.get("is_catastrophe")}
    safe_tested = {b: r for b, r in results.items() if not r.get("is_catastrophe")}

    if cats_tested:
        ho_cats = [r["nmae_head_only"] for r in cats_tested.values()
                   if r["nmae_head_only"] is not None]
        fp_cats = [r["nmae_full_param"] for r in cats_tested.values()
                   if r["nmae_full_param"] is not None]
        l30_cats = [r["nmae_last_30"] for r in cats_tested.values()
                    if r["nmae_last_30"] is not None]
        rescued = sum(1 for r in cats_tested.values()
                      if r.get("verdict", "").startswith("RESCUED"))
        print(f"\n[HEAD] SUMMARY — CATASTROPHE BUILDINGS (n={len(cats_tested)})")
        print(f"  full-param median: {np.median(fp_cats):.4f}")
        print(f"  head-only median:  {np.median(ho_cats):.4f}")
        print(f"  last_30 median:    {np.median(l30_cats):.4f}")
        print(f"  rescued by head-only: {rescued}/{len(cats_tested)}")

    summary = {"n_buildings": len(results), "n_catastrophes": len(cats_tested),
               "n_safe": len(safe_tested)}
    if ho_cats:
        summary["cat_fullparam_median"] = round(float(np.median(fp_cats)), 4)
        summary["cat_headonly_median"] = round(float(np.median(ho_cats)), 4)
        summary["cat_last30_median"] = round(float(np.median(l30_cats)), 4)
        summary["cat_rescued_count"] = rescued
        summary["cat_rescued_frac"] = round(rescued / len(cats_tested), 3)

    output = {"summary": summary, "buildings": results}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(output, indent=1, default=str))
    print(f"\n[HEAD] results → {out_json}")
    return output


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_csv")
    ap.add_argument("checkpoint")
    ap.add_argument("--buildings", nargs="+", default=None)
    ap.add_argument("--out", default="data/headonly_results.json")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run_headonly_experiment(a.raw_csv, a.checkpoint, a.buildings, out_json=a.out,
                            device=a.device)
