# PROTOCOL

This is the list of decisions we froze and won't touch. The idea is simple: if any one of these changes, the results built on top of it are no longer valid and have to be re-run. So they're locked. The date is when we froze each block.

## Data

- We load from the raw `electricity.csv` (BDG-2) through `bdg2_loader.load_electricity_csv`, which does ffill then bfill and resamples to hourly. We do not use `electricity_cleaned.csv`, because its imputation quietly wrecks the QC NaN criterion and makes part of the test-window ground truth synthetic.
- Population is `clean_buildings_2017.json`, n = 1,447. Criteria run on the raw 2017 data: nan_frac <= 0.5, zero_frac <= 0.5 (of the non-NaN values), std >= 1. This replaces an older unversioned list of 1,471 whose NaN check ran after ffill and was basically meaningless.

## Evaluation

- Task is day-ahead load forecasting. Horizon 24 h, one origin per day at midnight, 28 rolling origins. Context is up to 512 h before each origin. Each model gets its native context (snaive 168, Chronos 512, PatchTST 512 into a 96 h head where we use the first 24).
- Two test windows. Dec is the last 28 days of 2017, and it includes holidays and semester breaks on purpose, that's the regime stress. Oct (Oct 4 to 31, `end_hour=7296`) is the boring-operations control.
- Metrics live in `metrics.py` and that's the only place they're defined. NMAE is primary. NRMSE and eps-guarded MAPE are secondary, nCRPS is for the probabilistic models. Confidence intervals are percentile bootstrap over origins.

## Models

- snaive168 is the baseline. Any foundation model that can't beat it isn't earning its keep.
- chronos-t5-small (sample-based, 20 paths) and chronos-bolt-small (quantile-based), both through the official `chronos-forecasting` package.
- patchtst_b900k is our thing: a RevIN-PatchTST, 512 into 96, patch 16 stride 8, d_model 64, 3 layers, 542K params. We pretrained it 20,000 steps on 50 parquet partitions of Buildings-900K (in the writing we always say "a subset of"). Checkpoint is `RevIN_PatchTST_Foundation_v2_Mature.pth`, 43 keys, and it only ever loads with strict=True. Historical fine-tuning was full-parameter, AdamW lr 5e-4, 8 epochs, window stride 4.

## Profiler

The old v1 (11-dim) is frozen so earlier results still reproduce. The one that actually feeds the router is v2 (`spectral_building_profiler_v2.py`, 35 features, with the drift signals folded in as continuous features instead of a separate detector). Those 35 plus the 4 early-regime features make the 39 the classifier sees. The v2 features were frozen at the pre-registration commit.

## What P3 told us

- Fine-tuning genuinely transforms the model. FT-best beats zero-shot on 41 of 44 pre-registered buildings (93.2%), Wilcoxon p = 5.08e-11, median NMAE going 0.27 -> 0.13.
- The position-routing idea got falsified, and we pre-registered that test so we can't wriggle out of it. first_15 vs last_15 came out at p = 0.656, and the TRUE_DRIFT signal didn't replicate in the supplement. So the routed policy is basically just always-last_15. The old Panther 14.95pp slice gap that motivated all this is protocol-dependent and doesn't generalise.
- And this is where the real finding fell out: on a minority of buildings, fine-tuning on first_15 or full gives NMAE 8 to 24 while last_30 gives 0.07 to 0.14. Willard is the poster child (first_15 = 24.2, last_30 = 0.136). So always_last_30 is the dominant simple policy.

## The population poisoning study (frozen before we ran it)

- PatchTST, historical recipe, fine-tuned on every clean building times {last_30, full}, Dec protocol, zeroshot arm reused from the closed matrix.
- Label: poison = NMAE(full) minus NMAE(last_30). Catastrophe is poison > 0.10 (we also report 0.05 and 0.20 so nobody thinks we cherry-picked the threshold).
- Quarantine: if a building's last_30 NMAE is above 5.0 we drop it before computing anything. last_30 is supposed to be the safe fallback, so if it's also blowing up that's a broken evaluation, not a routing signal. This takes 1,447 down to 1,420 with 54 catastrophes.
- The three hypotheses. H1: catastrophes are predictable from the fingerprints, and they are, site-grouped AUC 0.9413, honest leave-Lamb-out 0.853. H2: they pile up in one site, and they do, Lamb holds 41 of 54. H3: a feature-routed {full | last_30} policy gets close to the two-arm oracle, and it does, 43.6% gap closure at 92.6% catch.
- Early-regime features (`early_features.csv`): early_flat_frac, early_zero_frac, early_late_mean_ratio, early_late_std_logratio.

## Frozen artifacts

- Pre-registration: `p3_preregistration.json` and `p3_oversample_true_drift.json`. The hash is SHA-256 over `json.dumps(payload, sort_keys=True)` with the sha256 field taken out. Check it yourself with `experiments/verify_prereg_hash.py`.
- Mitigations: head-only, LoRA (rank 4 and 8), and the RevIN diagnostic on the catastrophe exemplars (Kurt, Jo, Dottie).

## What's out of scope (a.k.a. the next paper)

The router assumes contamination is early-anchored, which is exactly why last_30 works as a fallback. Reverse-edge, mid-series, and multi-segment contamination are not handled here. We characterise them with `window_contamination_scan.py` and leave the actual segmentation work for later. We're honest about this being a fence around the claim rather than a solved problem.
