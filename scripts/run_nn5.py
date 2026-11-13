"""NN5 ATM cash backtest: 8 non-overlapping 14-day windows, every method sees the
same windows. Writes one gzip CSV per (method, context, run) under runs/nn5/."""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from common import (NN5_H, QCOLS, RUNS, CiProto, ChronosBolt, fill_weekly, load_nn5,
                    nn5_origins, peak_rss_mb, timed)

OUT = RUNS / "nn5"


def baseline(method, hist_raw, h):
    """Point forecast from NaN-filled history. hist_raw: [V, T]."""
    out = np.zeros((hist_raw.shape[0], h))
    for i, r in enumerate(hist_raw):
        f = fill_weekly(r)
        last_week = f[-7:]
        if method == "seasonal_naive":
            out[i] = [last_week[k % 7] for k in range(h)]
        elif method == "weekday_mean4":
            weeks = f[-28:].reshape(4, 7).mean(0)
            out[i] = [weeks[k % 7] for k in range(h)]
    return out


def ets(hist_raw, h):
    """statsforecast AutoETS(season 7) with 20/40/60/80% intervals -> 9 quantiles."""
    from statsforecast.models import AutoETS
    out = np.zeros((hist_raw.shape[0], 9, h))
    for i, r in enumerate(hist_raw):
        fc = AutoETS(season_length=7).forecast(y=fill_weekly(r), h=h, level=[20, 40, 60, 80])
        lo = [fc[f"lo-{l}"] for l in (80, 60, 40, 20)]
        hi = [fc[f"hi-{l}"] for l in (20, 40, 60, 80)]
        out[i] = np.stack(lo + [fc["mean"]] + hi)
    return out


def save(name, ctx, run, ids, dates, y, origins, preds, meta):
    rows = []
    for w, o in enumerate(origins):
        p = preds[w]
        for i, atm in enumerate(ids):
            for k in range(NN5_H):
                r = {"atm": atm, "window": w, "origin": str(dates[o].date()), "h": k + 1,
                     "date": str(dates[o + k].date()), "actual": y[i, o + k]}
                if p.ndim == 3:
                    r.update({c: p[i, j, k] for j, c in enumerate(QCOLS)})
                else:
                    r["q50"] = p[i, k]
                rows.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_ctx{ctx}_run{run}"
    pd.DataFrame(rows).to_csv(OUT / f"{stem}.csv.gz", index=False, float_format="%.4f")
    (OUT / f"{stem}.json").write_text(json.dumps(meta, indent=1))
    print(stem, json.dumps(meta), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["seasonal_naive", "weekday_mean4", "ets", "ci-proto", "chronos-bolt-small"])
    ap.add_argument("--group", type=int, default=1, help="ci-proto rows per call")
    ap.add_argument("--contexts", default="all", help="comma list of days or 'all'")
    ap.add_argument("--runs", type=int, default=1)
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    ids, dates, y = load_nn5()
    origins = nn5_origins(y.shape[1])
    name = a.method if a.group == 1 else f"{a.method}-cohort{a.group}"
    model = {"ci-proto": lambda: CiProto(a.group), "chronos-bolt-small": ChronosBolt}.get(a.method)
    model = model() if model else None
    for ctx in a.contexts.split(","):
        for run in range(a.runs):
            torch.manual_seed(run)
            preds, secs = [], 0.0
            for o in origins:
                hist = y[:, :o] if ctx == "all" else y[:, max(0, o - int(ctx)):o]
                if model is not None:
                    p, s = timed(model.forecast, hist, NN5_H)
                elif a.method == "ets":
                    p, s = timed(ets, hist, NN5_H)
                else:
                    p, s = timed(baseline, a.method, hist, NN5_H)
                preds.append(p)
                secs += s
            meta = {"method": name, "context": ctx, "run": run, "seconds": round(secs, 2),
                    "peak_rss_mb": round(peak_rss_mb()), "threads": torch.get_num_threads(),
                    "n_atms": len(ids), "windows": len(origins), "horizon": NN5_H}
            save(name, ctx, run, ids, dates, y, origins, preds, meta)


if __name__ == "__main__":
    main()
