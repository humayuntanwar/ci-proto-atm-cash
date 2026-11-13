"""Context-length ablation on a long real series: UCI ElectricityLoadDiagrams
(15-minute, 140,256 steps per client, via autogluon/chronos_datasets electricity_15min).
Clients: 8 drawn with seed 0 from the 154 clients that span the full 140,256 steps with
<1% zeros and no NaN. Windows: the last 3 non-overlapping days (96 steps each).
One process per (method, context) so peak RSS is measured per context length."""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from common import DATA, QCOLS, RUNS, CHRONOS_DATASETS_REV, CiProto, ChronosBolt, peak_rss_mb

OUT = RUNS / "long"
H = 96
WINDOWS = 3
N_CLIENTS = 8


def load_clients():
    cache = DATA / "elec15_selected.npz"
    if cache.exists():
        z = np.load(cache)
        return list(z["ids"]), z["y"], int(z["n_ok"])
    path = DATA / "electricity_15min.parquet"
    if not path.exists():
        url = (f"https://huggingface.co/datasets/autogluon/chronos_datasets/resolve/"
               f"{CHRONOS_DATASETS_REV}/electricity_15min/train-00000-of-00001.parquet")
        pd.read_parquet(url).to_parquet(path)
    df = pd.read_parquet(path, columns=["id", "consumption_kW"])
    ok = []
    for cid, v in zip(df["id"], df["consumption_kW"]):
        v = np.asarray(v, dtype=np.float64)
        if len(v) == 140256 and (v == 0).mean() < 0.01 and not np.isnan(v).any():
            ok.append((cid, v))
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(ok), N_CLIENTS, replace=False))
    ids, y = [ok[i][0] for i in pick], np.stack([ok[i][1] for i in pick])
    np.savez(cache, ids=np.array(ids), y=y, n_ok=len(ok))
    return ids, y, len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["ci-proto", "chronos-bolt-small", "seasonal_naive"])
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--runs", type=int, default=1)
    a = ap.parse_args()
    ids, y, n_ok = load_clients()
    T = y.shape[1]
    origins = [T - H * k for k in range(WINDOWS, 0, -1)]
    model = {"ci-proto": CiProto, "chronos-bolt-small": ChronosBolt}.get(a.method)
    model = model() if model else None
    if model is not None and a.method == "ci-proto" and a.context > 16384:
        # Warm-up: load the long checkpoint before timing (it is loaded lazily on the
        # first history longer than 16,384 steps).
        model.forecast(y[:1, -16385 - H:-H], 8)
    rows, timings = [], []
    for run in range(a.runs):
        torch.manual_seed(run)
        for w, o in enumerate(origins):
            for i, cid in enumerate(ids):
                hist = y[i:i + 1, o - a.context:o]
                t = time.perf_counter()
                if model is None:  # seasonal naive: repeat the last day
                    p = np.full((1, 9, H), np.nan)
                    p[0, 4] = hist[0, -H:]
                else:
                    p = model.forecast(hist, H)
                timings.append(time.perf_counter() - t)
                print(f"{cid} run{run} w{w} {timings[-1]:.2f}s", flush=True)
                for k in range(H):
                    r = {"client": cid, "run": run, "window": w, "h": k + 1, "actual": y[i, o + k],
                         "forecast_seconds": round(timings[-1], 3)}
                    r.update({c: p[0, j, k] for j, c in enumerate(QCOLS)})
                    rows.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{a.method}_ctx{a.context}"
    pd.DataFrame(rows).to_csv(OUT / f"{stem}.csv.gz", index=False, float_format="%.4f")
    meta = {"method": a.method, "context": a.context, "runs": a.runs, "clients": ids,
            "eligible_clients": n_ok, "windows": WINDOWS, "horizon": H,
            "seconds_per_forecast_median": round(float(np.median(timings)), 3),
            "seconds_per_forecast_max": round(float(np.max(timings)), 3),
            "seconds_total": round(float(np.sum(timings)), 1),
            "peak_rss_mb": round(peak_rss_mb()), "threads": torch.get_num_threads()}
    (OUT / f"{stem}.json").write_text(json.dumps(meta, indent=1))
    print(stem, json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
