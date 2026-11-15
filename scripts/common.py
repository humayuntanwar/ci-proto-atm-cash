"""Shared data loading, windows and model wrappers."""
import os
import time
import resource
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DATA = Path(os.environ.get("DATA_DIR", ROOT.parent / "data"))

CI_REPO = "ContinuousIntelligence/ci-proto"
CI_REV = "ee3031351a151e3f6febf8c62b53e6ea1008bef6"
CHRONOS_REPO = "amazon/chronos-bolt-small"
CHRONOS_REV = "772f3d25d38aec6d914c8949dab4462e2d46f5d8"
CHRONOS_DATASETS_REV = "eeecad0b82a8c237e212ce6f8d1abecb513e2cec"
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QCOLS = [f"q{int(q * 100)}" for q in QUANTILES]

# NN5 backtest: 4 non-overlapping 14-day windows at the end of each series.
NN5_H = 14
NN5_WINDOWS = 4


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_nn5():
    """111 ATMs x 791 days from autogluon/chronos_datasets (nn5), NaN kept."""
    path = DATA / "nn5.parquet"
    if not path.exists():
        url = (f"https://huggingface.co/datasets/autogluon/chronos_datasets/resolve/"
               f"{CHRONOS_DATASETS_REV}/nn5/train-00000-of-00001.parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_parquet(url).to_parquet(path)
    df = pd.read_parquet(path)
    ids = df["id"].tolist()
    y = np.stack([np.asarray(t, dtype=np.float64) for t in df["target"]])
    dates = pd.DatetimeIndex(df["timestamp"].iloc[0])
    return ids, dates, y


def nn5_origins(T):
    """Index of the first forecast day of each window (history = y[:, :origin])."""
    return [T - NN5_H * k for k in range(NN5_WINDOWS, 0, -1)]


def fill_weekly(y, period=7):
    """Baseline NaN rule: copy the value from 7 days earlier (repeat), then 7 days
    later, then the series median. Applied to history only, never to actuals."""
    y = np.array(y, dtype=np.float64, copy=True)
    for _ in range(len(y) // period + 1):
        idx = np.where(np.isnan(y))[0]
        idx = idx[idx >= period]
        if len(idx) == 0:
            break
        y[idx] = y[idx - period]
    for _ in range(len(y) // period + 1):
        idx = np.where(np.isnan(y))[0]
        idx = idx[idx < len(y) - period]
        if len(idx) == 0:
            break
        y[idx] = y[idx + period]
    y[np.isnan(y)] = np.nanmedian(y)
    return y


def seasonal_scale(hist, m):
    """MASE denominator: mean |y_t - y_{t-m}| over finite pairs of the full history."""
    d = np.abs(hist[m:] - hist[:-m])
    return float(np.nanmean(d))


class CiProto:
    name = "ci-proto"

    def __init__(self, group=1):
        import continuousintelligence as ci
        self.model = ci.load_model(CI_REPO, revision=CI_REV, device="cpu")
        self.group = group

    def forecast(self, hist, h):
        """hist: [V, T] float array (NaN allowed). Returns [V, 9, h].
        group=1: each row is its own call (pure univariate). group=k: rows are sent in
        cohorts of k with series_ids (the package's independent-series mode, which can
        share context inside the cohort; its CPU cost grows superlinearly with k)."""
        x = torch.tensor(np.asarray(hist, dtype=np.float32))
        outs = []
        for s in range(0, x.shape[0], self.group):
            chunk = x[s:s + self.group]
            ids = [str(i) for i in range(chunk.shape[0])]
            outs.append(self.model.forecast(chunk, horizon=h, series_ids=ids))
        return torch.cat(outs).numpy().astype(np.float64)


class ChronosBolt:
    name = "chronos-bolt-small"
    max_context = 2048

    def __init__(self):
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(
            CHRONOS_REPO, revision=CHRONOS_REV, device_map="cpu", torch_dtype=torch.float32)

    def forecast(self, hist, h):
        ctx = [torch.tensor(r[-self.max_context:], dtype=torch.float32) for r in np.asarray(hist)]
        q, _ = self.pipe.predict_quantiles(ctx, prediction_length=h, quantile_levels=QUANTILES)
        return q.permute(0, 2, 1).numpy().astype(np.float64)  # [V, h, 9] -> [V, 9, h]


def timed(fn, *a):
    t = time.perf_counter()
    out = fn(*a)
    return out, time.perf_counter() - t
