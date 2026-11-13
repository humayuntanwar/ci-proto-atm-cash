"""CPU time and peak RAM of one ci-proto forecast vs context length.
Series: WeatherBench hourly 2m temperature, first grid point in
autogluon/chronos_datasets weatherbench_hourly/2m_temperature shard 0 (id T212993,
350,640 hourly steps, 1979-2018). Each context runs in a fresh process with a warm-up
call on the same checkpoint, then one timed forecast of the next 24 hours."""
import json
import subprocess
import sys
import time

import numpy as np

from common import DATA, RUNS, CHRONOS_DATASETS_REV, CiProto, peak_rss_mb

OUT = RUNS / "scaling"
SERIES = DATA / "wb_t2m_T212993.npy"
H = 24


def load():
    if not SERIES.exists():
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("autogluon/chronos_datasets", "weatherbench_hourly/2m_temperature/train-00000-of-00009.parquet",
                            repo_type="dataset", revision=CHRONOS_DATASETS_REV)
        r = next(pq.ParquetFile(p).iter_batches(batch_size=1)).to_pylist()[0]
        np.save(SERIES, np.asarray(r["target"], dtype=np.float32))
    return np.load(SERIES).astype(np.float64)


def one(ctx):
    y = load()
    m = CiProto()
    warm = 16385 if ctx > 16384 else 512
    m.forecast(y[None, -H - warm:-H], 8)
    t = time.perf_counter()
    p = m.forecast(y[None, -H - ctx:-H], H)
    sec = time.perf_counter() - t
    mae = float(np.abs(p[0, 4] - y[-H:]).mean())
    print(json.dumps({"context": ctx, "status": "ok", "seconds": round(sec, 2), "peak_rss_mb": round(peak_rss_mb()),
                      "mae_24h": round(mae, 3)}))


RSS_CAP_MB = 3000  # hard cap for this shared box (brief budget ~2.5 GB)
TIME_CAP_S = 660


def watch(ctx):
    """Run one context in a child process; kill it above the RAM or time cap."""
    import psutil
    p = subprocess.Popen([sys.executable, __file__, "--one", str(ctx)], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    proc, t0, peak = psutil.Process(p.pid), time.perf_counter(), 0
    while p.poll() is None:
        try:
            peak = max(peak, proc.memory_info().rss / 2**20)
        except psutil.NoSuchProcess:
            break
        if peak > RSS_CAP_MB:
            p.kill()
            return {"context": ctx, "status": f"stopped: RSS passed {RSS_CAP_MB} MB cap", "rss_at_stop_mb": round(peak)}
        if time.perf_counter() - t0 > TIME_CAP_S:
            p.kill()
            return {"context": ctx, "status": f"stopped: over {TIME_CAP_S} s"}
        time.sleep(0.1)
    out, err = p.communicate()
    line = [l for l in out.splitlines() if l.startswith("{")]
    if not line:
        return {"context": ctx, "status": f"failed (exit {p.returncode})", "stderr_tail": err[-300:]}
    return json.loads(line[-1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctxs = [int(c) for c in sys.argv[1].split(",")]
    for ctx in ctxs:
        res = watch(ctx)
        res["series"] = "weatherbench_hourly 2m_temperature T212993"
        (OUT / f"ctx{ctx}.json").write_text(json.dumps(res, indent=1))
        print(res, flush=True)
        if res["status"] != "ok":
            print("stopping: ceiling reached", flush=True)
            break


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(int(sys.argv[2]))
    else:
        main()
