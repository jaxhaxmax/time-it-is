"""RevIN corruption diagnostic (GPU). shows why full-window fine-tuning detonates by comparing what RevIN sees during full vs last_30 training. for each catastrophe building it fine-tunes on both arms, extracts the RevIN affine parameters, and computes per-window mean/std over the training slice. full-window training sees a bimodal distribution (an early dead regime plus a late active regime) whose median window statistics no longer match the test window, while last_30 stays aligned. defaults to Kurt, Jo and a non-Lamb case (Dottie)."""

from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")


def _window_stats(series, ctx=512, stride=128):
    # per-window mean/std over sliding windows, the quantities RevIN normalises by
    means, stds = [], []
    for i in range(0, len(series) - ctx + 1, stride):
        w = series[i:i+ctx]
        if np.isnan(w).sum() > ctx * 0.3:
            continue
        means.append(float(np.nanmean(w)))
        stds.append(float(np.nanstd(w)))
    return np.array(means), np.array(stds)


def run_revin_diagnostic(
        raw_csv: str | Path,
        checkpoint_path: str | Path,
        buildings: list[str] | None = None,
        out_dir: str | Path = "data/revin",
        device: str = "cuda") -> dict:

    from src.bdg2_loader import load_electricity_csv
    from src.finetune_patchtst import finetune_on_slice, slice_bounds

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if buildings is None:
        buildings = ["Lamb_assembly_Kurt", "Lamb_office_Jo",
                     "Bull_education_Dottie"]

    print(f"[REVIN] loading raw data...")
    df = load_electricity_csv(raw_csv)

    results = {}
    for bid in buildings:
        print(f"\n[REVIN] {bid}")
        if bid not in df.columns:
            print(f"  NOT in raw data, skipping")
            continue

        series_2017 = df[bid].values.astype(np.float64)[:8760]
        test_hours = 672
        test_start = 8760 - test_hours
        pre_test = series_2017[:test_start]

        bld_result = {"building_id": bid}
        for arm in ["full", "last_30"]:
            print(f"\n  --- arm={arm} ---")
            a, b = slice_bounds(len(pre_test), arm)
            seg = pre_test[a:b]
            w_means, w_stds = _window_stats(seg, ctx=512, stride=128)
            print(f"  Training slice [{a}:{b}] = {len(seg)} hours")
            print(f"  Per-window means: min={w_means.min():.2f} "
                  f"median={np.median(w_means):.2f} max={w_means.max():.2f}")
            print(f"  Per-window stds:  min={w_stds.min():.2f} "
                  f"median={np.median(w_stds):.2f} max={w_stds.max():.2f}")

            # bimodality proxy: fraction of windows sitting near a dead-regime floor
            near_zero = (w_means < np.median(w_means) * 0.1).mean()
            print(f"  Windows with mean < 10% of median: {near_zero:.1%}")

            print(f"  Fine-tuning ({arm})...")
            model, trace = finetune_on_slice(
                str(checkpoint_path), series_2017, arm,
                test_hours=test_hours, device=device, verbose=False)

            revin_params = {}
            for name, module in model.named_modules():
                if getattr(module, "affine_weight", None) is not None:
                    w = module.affine_weight.detach().cpu().numpy()
                    b_param = module.affine_bias.detach().cpu().numpy()
                    revin_params[name] = {
                        "weight_mean": float(w.mean()), "weight_std": float(w.std()),
                        "weight_vals": w.tolist(),
                        "bias_mean": float(b_param.mean()),
                        "bias_std": float(b_param.std()),
                        "bias_vals": b_param.tolist()}
                    print(f"  RevIN {name}: weight {w.mean():.4f}±{w.std():.4f} "
                          f"bias {b_param.mean():.4f}±{b_param.std():.4f}")

            bld_result[arm] = {
                "slice_hours": int(len(seg)),
                "window_mean_stats": {
                    "min": round(float(w_means.min()), 4),
                    "median": round(float(np.median(w_means)), 4),
                    "max": round(float(w_means.max()), 4),
                    "near_zero_frac": round(float(near_zero), 4)},
                "window_std_stats": {
                    "min": round(float(w_stds.min()), 4),
                    "median": round(float(np.median(w_stds)), 4),
                    "max": round(float(w_stds.max()), 4)},
                "revin_params": revin_params}
            del model
            torch.cuda.empty_cache()

        test_means, test_stds = _window_stats(series_2017[test_start:],
                                              ctx=512, stride=128)
        if len(test_means) > 0:
            bld_result["test_window"] = {
                "mean_median": round(float(np.median(test_means)), 4),
                "std_median": round(float(np.median(test_stds)), 4)}
            print(f"\n  Test window: mean={np.median(test_means):.2f} "
                  f"std={np.median(test_stds):.2f}")
        else:
            # test window shorter than one context, fall back to raw stats
            tw = series_2017[test_start:]
            bld_result["test_window"] = {
                "mean_median": round(float(np.nanmean(tw)), 4),
                "std_median": round(float(np.nanstd(tw)), 4)}
            print(f"\n  Test window (raw): mean={np.nanmean(tw):.2f} "
                  f"std={np.nanstd(tw):.2f}")

        full_med = bld_result["full"]["window_mean_stats"]["median"]
        l30_med = bld_result["last_30"]["window_mean_stats"]["median"]
        test_med = bld_result["test_window"]["mean_median"]
        mismatch_full = abs(full_med - test_med) / max(abs(test_med), 1e-9)
        mismatch_l30 = abs(l30_med - test_med) / max(abs(test_med), 1e-9)
        print(f"\n  SUMMARY:")
        print(f"    full training median window mean:   {full_med:.2f}")
        print(f"    last_30 training median window mean: {l30_med:.2f}")
        print(f"    test window median mean:             {test_med:.2f}")
        print(f"    full↔test mismatch:   {mismatch_full:.1%}")
        print(f"    last_30↔test mismatch: {mismatch_l30:.1%}")
        bld_result["mismatch_full_pct"] = round(float(mismatch_full * 100), 1)
        bld_result["mismatch_l30_pct"] = round(float(mismatch_l30 * 100), 1)
        results[bid] = bld_result

    Path(out / "revin_diagnostic.json").write_text(
        json.dumps(results, indent=1, default=str))
    print(f"\n[REVIN] results → {out}/revin_diagnostic.json")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_csv")
    ap.add_argument("checkpoint")
    ap.add_argument("--buildings", nargs="+",
                    default=["Lamb_assembly_Kurt", "Lamb_office_Jo",
                             "Bull_education_Dottie"])
    ap.add_argument("--out_dir", default="data/revin")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run_revin_diagnostic(a.raw_csv, a.checkpoint, a.buildings, a.out_dir, a.device)
