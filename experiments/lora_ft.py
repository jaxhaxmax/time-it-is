"""LoRA fine-tuning experiment (GPU). head-only freezing rescues most catastrophes, which places the failure in the encoder, so this tests the finer instrument: LoRA constrains encoder updates to a low-rank subspace, checking whether a little constrained adaptation is safe, partially safe, or still detonates. two configs per rank (LoRA on encoder with head frozen, and LoRA plus trainable head), same building set as headonly_ft.py (seed 42).

architecture note: LoRA wraps out_proj, linear1, linear2 in each of the 3 encoder layers, 9,216 params at rank 4 (1.7% of the model). PyTorch's MultiheadAttention calls F.linear(x, out_proj.weight) rather than out_proj.forward(), so LoRALinear exposes a .weight property returning W_frozen + (B @ A)·scaling to make gradients flow through the attention path too."""

from __future__ import annotations
import json, sys, copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, ".")


class LoRALinear(nn.Module):
    # freezes the wrapped Linear, adds trainable A,B. B=0 at init so the delta starts at zero and the model begins from the checkpoint exactly. the .weight property covers both call paths (self(x) and F.linear(x, self.weight))

    def __init__(self, original: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank
        for p in self.original.parameters():
            p.requires_grad = False
        in_f, out_f = original.in_features, original.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))

    @property
    def weight(self):
        return self.original.weight + (self.lora_B @ self.lora_A) * self.scaling

    @property
    def bias(self):
        return self.original.bias

    @property
    def in_features(self):
        return self.original.in_features

    @property
    def out_features(self):
        return self.original.out_features

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return (f"in={self.original.in_features}, out={self.original.out_features}, "
                f"rank={self.rank}, scaling={self.scaling:.2f}")


def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def apply_lora(model, rank: int = 4, alpha: float = 1.0,
               unfreeze_head: bool = False):
    # freeze all, wrap the 3 encoder Linears per layer, optionally unfreeze the head. in-place
    for p in model.parameters():
        p.requires_grad = False

    lora_count = lora_params = 0
    for layer in model.transformer_encoder.layers:
        for attr, mod in (("self_attn", getattr(layer.self_attn, "out_proj", None)),
                          ("linear1", getattr(layer, "linear1", None)),
                          ("linear2", getattr(layer, "linear2", None))):
            if isinstance(mod, nn.Linear):
                wrapped = LoRALinear(mod, rank=rank, alpha=alpha)
                if attr == "self_attn":
                    layer.self_attn.out_proj = wrapped
                else:
                    setattr(layer, attr, wrapped)
                lora_count += 1
                lora_params += wrapped.lora_A.numel() + wrapped.lora_B.numel()

    head_params = 0
    if unfreeze_head:
        for p in model.projection_head.parameters():
            p.requires_grad = True
            head_params += p.numel()

    total, trainable = _count_params(model)
    stats = {
        "lora_rank": rank, "lora_alpha": alpha,
        "lora_layers_wrapped": lora_count, "lora_params": lora_params,
        "head_params_trainable": head_params, "total_trainable": trainable,
        "total_params": total + lora_params,        # LoRA adds new params
        "trainable_pct": round(100 * trainable / (total + lora_params), 2),
    }
    return model, stats


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


def finetune_lora(checkpoint_path: str, series_2017: np.ndarray,
                  rank: int = 4, alpha: float = 1.0,
                  unfreeze_head: bool = False,
                  device: str = "cuda", test_hours: int = 672):
    # same recipe as full-param FT (lr 5e-4, wd 1e-5, 8 epochs, stride 4, batch 32, clip 1.0), only LoRA (and optionally head) params update
    from src.patchtst_900k import load_patchtst_900k
    from src.finetune_patchtst import _make_windows, RECIPE

    test_start = len(series_2017) - test_hours
    pre = np.asarray(series_2017[:test_start], dtype=np.float64)

    model = load_patchtst_900k(checkpoint_path, device=device, eval_mode=False)
    model, lora_stats = apply_lora(model, rank=rank, alpha=alpha,
                                   unfreeze_head=unfreeze_head)
    model = model.to(device)                      # LoRA params are born on CPU
    config_name = f"lora_r{rank}" + ("_plus_head" if unfreeze_head else "_encoder")
    print(f"    {config_name}: {lora_stats['total_trainable']} trainable / "
          f"{lora_stats['total_params']} total params "
          f"({lora_stats['trainable_pct']}%)")

    X, Y = _make_windows(pre, model.context_len, model.pred_len, RECIPE["stride"])
    n = len(X)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=RECIPE["lr"],
                            weight_decay=RECIPE["weight_decay"])
    lossf = torch.nn.MSELoss()
    torch.manual_seed(RECIPE["seed"])
    np.random.seed(RECIPE["seed"])

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
            torch.nn.utils.clip_grad_norm_(trainable_params, RECIPE["grad_clip"])
            opt.step()
            tot += loss.item() * len(idx)
        trace.append(round(tot / n, 6))
    model.eval()
    return model, trace, lora_stats


def run_lora_experiment(
        raw_csv: str | Path,
        checkpoint_path: str | Path,
        buildings: list[str] | None = None,
        ft_results_csv: str | Path = "data/ft_results_population.csv",
        headonly_json: str | Path | None = "data/headonly_results.json",
        ranks: list[int] | None = None,
        out_json: str | Path = "data/lora_results.json",
        device: str = "cuda") -> dict:

    from src.bdg2_loader import load_electricity_csv

    if ranks is None:
        ranks = [4, 8]

    print("[LORA] loading raw data...")
    df = load_electricity_csv(raw_csv)

    ft = pd.read_csv(ft_results_csv)
    p = ft.pivot(index="building_id", columns="arm", values="nmae")
    p = p.dropna(subset=["last_30", "full"])
    p = p[p["last_30"] <= 5.0].copy()
    p["poison"] = p["full"] - p["last_30"]

    headonly = {}
    if headonly_json and Path(headonly_json).exists():
        headonly = json.loads(Path(headonly_json).read_text()).get("buildings", {})
        print(f"[LORA] loaded {len(headonly)} head-only results for comparison")

    if buildings is None:
        cats = p[p["poison"] > 0.10].index.tolist()
        safe_pool = p[p["poison"] <= 0.10].index.tolist()
        np.random.seed(42)                        # same seed as headonly_ft.py
        safe_sample = list(np.random.choice(safe_pool,
                                             size=min(50, len(safe_pool)),
                                             replace=False))
        buildings = sorted(set(cats + safe_sample))
        print(f"[LORA] auto-selected {len(cats)} catastrophe + "
              f"{len(safe_sample)} safe = {len(buildings)} buildings")
    else:
        print(f"[LORA] using {len(buildings)} specified buildings")

    configs = []
    for r in ranks:
        configs.append({"rank": r, "alpha": float(r), "unfreeze_head": False,
                        "label": f"lora_r{r}_encoder"})
        configs.append({"rank": r, "alpha": float(r), "unfreeze_head": True,
                        "label": f"lora_r{r}_plus_head"})
    print(f"[LORA] configs: {[c['label'] for c in configs]}  "
          f"ranks: {ranks}  buildings: {len(buildings)}  "
          f"runs: {len(buildings) * len(configs)}")

    results = {}
    for i, bid in enumerate(buildings):
        print(f"\n[LORA] ({i+1}/{len(buildings)}) {bid}")
        if bid not in df.columns:
            print(f"  NOT in raw data, skipping")
            continue

        series = df[bid].values.astype(np.float64)[:8760]
        is_cat = bid in p.index and p.loc[bid, "poison"] > 0.10
        existing_full = float(p.loc[bid, "full"]) if bid in p.index else None
        existing_l30 = float(p.loc[bid, "last_30"]) if bid in p.index else None
        existing_ho = headonly[bid].get("nmae_head_only") if bid in headonly else None

        tag = "CAT" if is_cat else "safe"
        print(f"  [{tag}] full={existing_full:.4f} l30={existing_l30:.4f}"
              + (f" ho={existing_ho:.4f}" if existing_ho else ""))

        bld_result = {"is_catastrophe": is_cat,
                      "nmae_full_param": existing_full,
                      "nmae_last_30": existing_l30,
                      "nmae_head_only": existing_ho}

        for cfg in configs:
            label = cfg["label"]
            try:
                model, trace, stats = finetune_lora(
                    str(checkpoint_path), series, rank=cfg["rank"],
                    alpha=cfg["alpha"], unfreeze_head=cfg["unfreeze_head"],
                    device=device)
                nmae = _evaluate_model(model, series, device=device)
                print(f"  {label}: NMAE={nmae:.4f} "
                      f"(trainable={stats['total_trainable']}, "
                      f"{stats['trainable_pct']}%)")
                bld_result[f"nmae_{label}"] = round(nmae, 4)
                bld_result[f"trace_{label}"] = trace
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  {label}: FAILED — {e}")
                bld_result[f"nmae_{label}"] = None

        results[bid] = bld_result

    cats_r = {b: r for b, r in results.items() if r.get("is_catastrophe")}
    safe_r = {b: r for b, r in results.items() if not r.get("is_catastrophe")}
    print(f"\n[LORA] SUMMARY")
    summary = {"n_buildings": len(results), "n_catastrophes": len(cats_r),
               "n_safe": len(safe_r), "ranks_tested": ranks,
               "configs": [c["label"] for c in configs]}

    for cfg in configs:
        label = cfg["label"]
        key = f"nmae_{label}"
        cat_vals = [r[key] for r in cats_r.values() if r.get(key) is not None]
        safe_vals = [r[key] for r in safe_r.values() if r.get(key) is not None]
        if cat_vals:
            # rescued = below half full-param, detonated = NMAE still > 1.0
            rescued = sum(1 for r in cats_r.values()
                          if r.get(key) is not None
                          and r.get("nmae_full_param") is not None
                          and r[key] < r["nmae_full_param"] * 0.5)
            detonated = sum(1 for v in cat_vals if v > 1.0)
            print(f"\n  {label} — CATASTROPHE (n={len(cat_vals)}): "
                  f"median={np.median(cat_vals):.4f} max={np.max(cat_vals):.4f} "
                  f"rescued={rescued}/{len(cat_vals)} detonated={detonated}")
            summary[f"{label}_cat_median"] = round(float(np.median(cat_vals)), 4)
            summary[f"{label}_cat_max"] = round(float(np.max(cat_vals)), 4)
            summary[f"{label}_cat_rescued"] = rescued
            summary[f"{label}_cat_detonated"] = detonated
        if safe_vals:
            print(f"  {label} — SAFE (n={len(safe_vals)}): "
                  f"median={np.median(safe_vals):.4f}")
            summary[f"{label}_safe_median"] = round(float(np.median(safe_vals)), 4)

    print(f"\n  COMPARISON TABLE (catastrophe buildings)")
    print(f"  {'Strategy':<25s} {'Median':>8s} {'Max':>8s} "
          f"{'Rescued':>10s} {'Detonated':>10s}")
    fp_cats = [r["nmae_full_param"] for r in cats_r.values()
               if r.get("nmae_full_param") is not None]
    if fp_cats:
        print(f"  {'full_param':<25s} {np.median(fp_cats):8.4f} "
              f"{np.max(fp_cats):8.4f} {'—':>10s} {'—':>10s}")
    ho_cats = [r["nmae_head_only"] for r in cats_r.values()
               if r.get("nmae_head_only") is not None]
    if ho_cats:
        ho_resc = sum(1 for r in cats_r.values()
                      if r.get("nmae_head_only") is not None
                      and r.get("nmae_full_param") is not None
                      and r["nmae_head_only"] < r["nmae_full_param"] * 0.5)
        ho_det = sum(1 for v in ho_cats if v > 1.0)
        print(f"  {'head_only':<25s} {np.median(ho_cats):8.4f} "
              f"{np.max(ho_cats):8.4f} {ho_resc:>10d} {ho_det:>10d}")
    for cfg in configs:
        label = cfg["label"]
        vals = [r[f"nmae_{label}"] for r in cats_r.values()
                if r.get(f"nmae_{label}") is not None]
        if vals:
            print(f"  {label:<25s} {np.median(vals):8.4f} "
                  f"{np.max(vals):8.4f} "
                  f"{summary.get(f'{label}_cat_rescued', 0):>10d} "
                  f"{summary.get(f'{label}_cat_detonated', 0):>10d}")
    l30_cats = [r["nmae_last_30"] for r in cats_r.values()
                if r.get("nmae_last_30") is not None]
    if l30_cats:
        print(f"  {'last_30':<25s} {np.median(l30_cats):8.4f} "
              f"{np.max(l30_cats):8.4f} {'—':>10s} {'—':>10s}")

    output = {"summary": summary, "buildings": results}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(output, indent=1, default=str))
    print(f"\n[LORA] results → {out_json}")
    return output


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_csv")
    ap.add_argument("checkpoint")
    ap.add_argument("--ranks", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--buildings", nargs="+", default=None)
    ap.add_argument("--ft-results", default="data/ft_results_population.csv")
    ap.add_argument("--headonly-json", default="data/headonly_results.json")
    ap.add_argument("--out", default="data/lora_results.json")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run_lora_experiment(a.raw_csv, a.checkpoint, buildings=a.buildings,
                        ft_results_csv=a.ft_results, headonly_json=a.headonly_json,
                        ranks=a.ranks, out_json=a.out, device=a.device)
