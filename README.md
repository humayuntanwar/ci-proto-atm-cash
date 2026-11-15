# ci-proto-atm-cash

Does 64x more history help? ci-proto vs baselines on 111 real ATMs, on a CPU.

[ci-proto](https://huggingface.co/ContinuousIntelligence/ci-proto) (Continuous Intelligence, released 12 Nov 2026)
is a 167M-parameter zero-shot forecaster that returns 9 quantiles and is pitched on a "2M+ context window,
64x the history of any other time-series foundation model". I tested two things a cash-ops team cares about:
is it better than what they already do (repeat last week), and does more history actually help?

Pinned: ci-proto HF revision `ee3031351a151e3f6febf8c62b53e6ea1008bef6`, `continuousintelligence==0.1.1`,
`amazon/chronos-bolt-small` revision `772f3d25d38aec6d914c8949dab4462e2d46f5d8`,
`autogluon/chronos_datasets` revision `eeecad0b82a8c237e212ce6f8d1abecb513e2cec`. CPU only (8 threads), torch 2.15.0+cpu.

## Results

**NN5 ATM cash** (111 UK ATMs, daily, 4 rolling 14-day windows starting 1998-03-23, 04-06, 04-20, 05-04; full history;
mean over ATMs, [95% bootstrap CI over ATMs, 2,000 resamples]):

| method | runs | MASE h14 | MASE h7 | scaled QL h14 | 10-90% coverage | beats seasonal naive |
|---|---|---|---|---|---|---|
| seasonal naive (repeat last week) | 1 | 1.049 [1.001, 1.100] | 1.115 [1.066, 1.165] | - | - | - |
| same-weekday mean, last 4 weeks | 1 | 0.901 [0.863, 0.940] | 0.995 [0.954, 1.036] | - | - | 101/111 |
| AutoETS (statsforecast, season 7) | 1 | 0.856 [0.817, 0.901] | 0.917 [0.875, 0.963] | 0.702 [0.672, 0.735] | 0.86 [0.85, 0.88] | 105/111 |
| Chronos-Bolt-Small | 3 | 0.836 [0.802, 0.870] | 0.900 [0.863, 0.938] | 0.678 [0.652, 0.707] | 0.84 [0.83, 0.85] | 108/111 |
| **ci-proto** (one ATM per call) | 3 | **0.771 [0.742, 0.803]** | **0.861 [0.831, 0.894]** | **0.632 [0.605, 0.657]** | 0.76 [0.75, 0.77] | **111/111** |
| ci-proto, cohorts of 8 ATMs (sensitivity) | 1 | 0.755 [0.723, 0.789] | 0.843 [0.810, 0.880] | 0.619 [0.593, 0.649] | 0.79 [0.78, 0.80] | 111/111 |

ci-proto beats Chronos-Bolt-Small on 103/111 ATMs (0.93 [0.87, 0.97]). Its 10-90% band is a bit narrow: 76% coverage vs 80% nominal.

**Does more history help?** ci-proto MASE, [95% CI]; "diff" = paired difference vs the reference context, bootstrap over series.

| NN5 (n=111 ATMs), days of history | runs | MASE h14 | diff vs all |
|---|---|---|---|
| 14 | 2 | 1.259 [1.211, 1.313] | +0.488 [0.455, 0.523] |
| 28 | 1 | 0.929 [0.891, 0.969] | +0.158 [0.138, 0.178] |
| 56 | 1 | 0.839 [0.805, 0.877] | +0.068 [0.054, 0.083] |
| 112 | 1 | 0.820 [0.787, 0.860] | +0.049 [0.038, 0.060] |
| 224 | 1 | 0.793 [0.761, 0.829] | +0.022 [0.014, 0.032] |
| 448 | 1 | 0.771 [0.741, 0.804] | -0.000 [-0.006, 0.005] |
| all (735-777 days) | 3 | 0.771 [0.742, 0.803] | 0 |

| Electricity 15-min (n=8 clients x 3 days), steps | runs | MASE, 1-day horizon | diff vs 2K | median s / forecast | peak RSS |
|---|---|---|---|---|---|
| 512 (5 days) | 3 | 1.047 [0.905, 1.182] | +0.113 [0.001, 0.263] | 0.26 | 1.6 GB |
| 2,048 (21 days) | 3 | 0.935 [0.757, 1.113] | 0 | 0.69 | 1.6 GB |
| 8,192 | 3 | 0.937 [0.751, 1.123] | +0.002 [-0.014, 0.019] | 2.3 | 1.6 GB |
| 16,384 (last standard-checkpoint size) | 3 | 0.939 [0.750, 1.128] | +0.004 [-0.009, 0.017] | 5.3 | 1.6 GB |
| 32,768 (long checkpoint) | 1 | 0.944 [0.757, 1.142] | +0.009 [-0.013, 0.035] | 12.1 | 2.2 GB |
| 65,536 (683 days) | 1 | 0.946 [0.753, 1.149] | +0.011 [-0.009, 0.034] | 20.3 | 2.6 GB |
| 131,072 | - | stopped: RSS passed my 3,000 MB cap (3,307 MB) | | | |
| repeat last day (baseline) | 1 | 1.222 [1.040, 1.413] | | | |

Single runs are cheap to trust here: across 3 full-history NN5 runs ci-proto's mean MASE was 0.7710 / 0.7709 / 0.7709.
Chronos-Bolt-Small (max context 2,048) on the same NN5 contexts: 1.622, 1.265, 0.939, 0.892, 0.886, 0.846, 0.836 (3 runs each, CIs in `report.py`).

On ATM cash, history helps up to about a year and then stops: 448 days and all ~2 years are identical.
On electricity, the curve is flat from 2K steps (3 weeks) to 64K (almost 2 years); the point estimates drift slightly up.
At 131,072 steps the process passed my 3,000 MB RAM cap (3,307 MB when stopped), so 256K, 1M and 2M were not reached on this box.
Electricity Chronos-Bolt-Small: 1.139 [0.935, 1.398] at 512, 0.924 [0.719, 1.132] at 2,048 steps (3 runs).

**Cash-ops translation (illustrative only).** Stock each ATM each day at a forecast, first 7 days of every window,
n=111 ATMs x 4 windows. "Short" = actual > stocked; "idle" = sum of (stocked - actual) where positive, as % of withdrawals. No costs assumed.

| rule | ATM-days short | idle cash, % of withdrawals |
|---|---|---|
| ci-proto 90% quantile | 13.0% [11.9, 14.0] | 28.4% [26.8, 29.9] |
| seasonal naive x 1.33 (buffer picked on the same windows to match ci-proto's short rate) | 12.9% [11.5, 14.3] | 44.2% [43.0, 45.4] |
| Chronos-Bolt-Small 90% quantile | 8.8% [7.8, 9.9] | 35.4% [33.7, 37.2] |
| AutoETS 90% quantile | 7.4% [6.5, 8.3] | 39.3% [37.5, 41.1] |

At the same shortfall rate, ci-proto leaves about a third less idle cash than "last week x buffer". Note the 90% quantile
ran short 13% of the time, not 10%; the buffer for the baseline was chosen on the test windows, which favours the baseline.

Images: `images/1_head_to_head_mase.png`, `images/2_context_length.png`, `images/3_atm_forecast.png`
(NN5-024: the ATM with the median ci-proto MASE, last window), `images/4_cpu_cost.png`.

## Setup and data

- **NN5**: `autogluon/chronos_datasets` / `nn5`: 111 series x 791 days (1996-03-18 to 1998-05-17), 1,677 NaN (1,673 in history, 4 in test windows), 419 zeros.
  Foundation models get the raw history with NaN (both document NaN support). Baselines get history where each NaN is
  replaced by the value 7 days earlier (then 7 days later, then the series median). NaN actuals are not scored.
- **Long series**: UCI ElectricityLoadDiagrams via `chronos_datasets/electricity_15min`. 8 clients drawn with seed 0 from
  the 154 that cover all 140,256 steps with <1% zeros and no NaN; last 3 days as 3 non-overlapping 96-step windows.
- **CPU/RAM probe**: one forecast per context in a fresh process, WeatherBench hourly 2m temperature (one grid point, 350,640 steps).
- **Metrics**: MASE with seasonal period 7 (NN5) or 96 (electricity), scale from each series' full history before the window origin,
  so it does not change with context length. Scaled quantile loss = mean pinball loss over the 9 quantiles / same scale.
- ci-proto is called with the package defaults (`tta.json` from the pinned revision, untouched), `series_ids`, no timestamps or covariates.

## Is this fair to the authors?

- **What they claim**: #1 on fev-bench, an aggregate over 100 tasks scored by scaled quantile loss, and a 2M+ context window.
  Neither claim is about ATM cash. NN5 is not one of the 100 fev-bench tasks (checked against fev's `tasks.yaml`).
  A public MASE view (TSFM.ai) reportedly ranks it #2 behind TimeRSI-1.5; I did not re-check that.
- **What I measured**: on NN5 it was the most accurate method I ran, on every metric except band coverage, and beat
  seasonal naive on all 111 ATMs. That is consistent with the accuracy claim. The context claim is about capacity, not
  that more history always helps; on my two datasets the extra history stopped paying off early (about a year of daily
  data; about 3 weeks of 15-minute data). Two datasets is not a general result.
- **Our setup, not the claim**: CPU-only (portable torch backend, bf16 per `tta.json`); 3,000 MB RAM cap, so I could not test
  beyond 64K steps; n=8 electricity clients gives wide CIs; 32K and 64K are single runs; I ran ci-proto one ATM per call
  because a single 111-ATM cohort call was impractically slow on CPU. Cohorts of 8 did slightly better
  (0.755 vs 0.771), so the default cohort mode may be the better way to use it.
- **Leakage**: ci-proto's card says only "a large corpus of real-world and synthetic time series"; it does not list datasets,
  so NN5 and the electricity data may be in its training set. Chronos-Bolt documents NN5 as zero-shot (held out), but the
  electricity 15-min data is part of the Chronos training corpus, so Chronos may be advantaged there.
- **Changed after seeing results**: I first ran 8 windows; after seeing ci-proto's CPU cost I cut to 4 (brief allowed 4-8).
  I added the cohort-of-8 run and the paired differences after the main results. Timers were switched to a monotonic clock
  after the VM paused mid-run (the wall clock jumped while the monotonic clock did not); medians are unaffected, but the
  max-time fields in `runs/long/*.json` for 8K/16K/32K include those pauses. I stopped extra repeat runs of the NN5 context
  ablation for time, so those points are single runs (14 days has 2).

## Determinism

ci-proto is not bit-for-bit deterministic with 8 CPU threads (same seed, different outputs; 1 thread was identical), but the
spread is tiny (see above). Chronos-Bolt gave identical results in all 3 runs.

## Run it

```sh
python -m venv venv && . venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
cd scripts
python run_nn5.py --method ci-proto --contexts all,14,28,56,112,224,448 --runs 3  # also seasonal_naive, weekday_mean4, ets, chronos-bolt-small
python run_long.py --method ci-proto --context 2048 --runs 3   # 512 ... 65536
python run_scaling.py 16384,65536,131072,262144
python report.py                                 # every number above, plus images/
```

Data goes to `../data` (outside the repo). Raw per-window, per-series forecasts and metrics are in `runs/`.

**Cost on this box** (8 CPU cores, ~2.5-6 GB free RAM): one full NN5 pass (111 ATMs x 4 windows) took 171-202 s for ci-proto,
1.5-2.0 s for Chronos-Bolt-Small, 66 s for AutoETS. ci-proto peaks at 1.6 GB up to 16K steps, 2.6 GB at 64K. All runs took about 1.5 hours of wall time, including host pauses.

## Licence

ci-proto weights are under the ci-proto Code and Model Licence (Apache-2.0 based, attribution and notice
rules in Section 10 for external products; purely internal use is exempt). This repo is internal-style research and does not
redistribute the weights.

Uses ci-proto, released by Continuous Intelligence. Continuous Intelligence does not endorse or approve this product or service.
