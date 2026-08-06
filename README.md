# time-it-is

This is the code and data for our SoCTA 2026 paper, **"When Fine-Tuning Detonates: Predicting Catastrophic Adaptation Failure in Domain-Pretrained Load Forecasting Models."**

Quick heads up on what this repo is. It's the cleaned-up, curated version of the project, just the files you actually need to redo the numbers and figures. The full messy development history lives in a separate repo (`time-series-grind`), and that's also where the pre-registration was first committed with its timestamp. So think of this one as the tidy artifact and that one as the paper trail.

## So what's the paper about

We take a domain foundation model (a RevIN-PatchTST that we pretrained ourselves on a subset of Buildings-900K) and fine-tune it on each building's own history. Most of the time this helps, which is the whole point of having a foundation model. But on a small set of buildings it does the opposite and blows up hard. Full-history fine-tuning sends the error to NMAE 8 all the way up to 24, while the exact same model fine-tuned on only the last 30% of history sits calmly at 0.07 to 0.14.

So the question we ask is basically: can you see this coming? Before you run any fine-tuning, just from cheap statistical fingerprints of the raw meter series, can you tell which buildings are about to detonate and route around them?

Turns out yes, mostly.

## What we found

- Population is 1,447 clean 2017 buildings, 1,420 once you drop the quarantine cases, and 54 of those are catastrophes.
- A Random Forest on 39 features (35 from the profiler plus 4 early-regime ones) predicts catastrophe at site-grouped AUC 0.9413. Now, one site (Lamb) holds 41 of the 54 catastrophes, so the honest number is the leave-Lamb-out one, which is 0.853. We report that as the real headline, not the inflated pooled figure.
- Fine-tuning helps basically everywhere else. FT-best beats zero-shot on 41 of 44 pre-registered buildings (that's 93.2%, Wilcoxon p = 5.08e-11).
- The risk-gated router (send flagged buildings to last_30, everyone else to full history) closes 43.6% of the gap to the two-arm oracle while catching 92.6% of the catastrophes.
- One honest limitation: the router assumes the contamination sits early in the series (think a building that was vacant then came online). Mid-series and multi-segment cases are out of scope, and we quantify that rather than hide it.

## How to reproduce

Everything runs in pipeline order. Each script reads the committed files in `data/` and writes back there. CPU is fine for all the analysis and figure scripts. The fine-tuning and mitigation ones (`adaptation_study.py`, `headonly_ft.py`, `lora_ft.py`, `revin_diagnostic.py`) need a GPU.

Rough order:

1. Core stuff in `src/`: the BDG-2 loader, the frozen metrics, the PatchTST-900K model, the fine-tuning engine.
2. Population: `data/clean_buildings_2017.json`.
3. Fingerprints: the profiler and early-feature scripts -> `profiles_v2.csv`, `early_features.csv`.
4. Zero-shot baseline: `zeroshot_eval.py` -> the two label matrices.
5. Recurrence check: `p2_recurrence.py` -> `drift_recurrence.csv`.
6. Pre-registration: `p3_preregister.py`, then verify it (see below).
7. Adaptation study: `adaptation_study.py` -> `ft_results_p3_5arm.csv`.
8. Population study: `ft_results_population.csv` and `poison_analysis.py`.
9. Classifier and robustness: `robustness_suite.py`, `robustness_lamb.py`, `threshold_analysis.py`, `final_gaps.py`.
10. Mitigations: `headonly_ft.py`, `lora_ft.py`, `revin_diagnostic.py`.
11. Window scan: `window_contamination_scan.py`.
12. Figures: `make_figures.py`.

## About that pre-registration

We froze the routing predictions and hashed them before any fine-tuning result existed. So if you want to check nobody quietly edited the frozen files afterward, just run:

```
python experiments/verify_prereg_hash.py data/
```

It recomputes each SHA-256 from the file's own contents and checks it against the stored one, and confirms the supplement points back at the primary. You don't have to trust us, the file checks itself.

## Data and checkpoint

The BDG-2 `electricity.csv` and the pretrained checkpoint (`RevIN_PatchTST_Foundation_v2_Mature.pth`, 43 keys, 542K params) aren't in here. BDG-2 is public (link below), and the checkpoint lives on Kaggle.

## Requirements

See `requirements.txt`. The top five packages cover everything on CPU. `torch` and `chronos-forecasting` are only needed for the GPU scripts.

## References

- Dataset: Building Data Genome Project 2, https://github.com/buds-lab/building-data-genome-project-2
- Models: our domain-pretrained PatchTST (subset of Buildings-900K), plus Chronos-T5-Small and Chronos-Bolt-Small as baselines.
