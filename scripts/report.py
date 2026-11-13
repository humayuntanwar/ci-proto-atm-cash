"""Regenerate every number and chart in the README from runs/.
Writes runs/summary.json, runs/metrics_*.csv and images/*.png, prints markdown tables."""
import glob
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import NN5_WINDOWS, QCOLS, QUANTILES, RUNS, ROOT, load_nn5, nn5_origins, seasonal_scale

IMG = ROOT / "images"
B = 2000
RNG = np.random.default_rng(0)
NICE = {"seasonal_naive": "seasonal naive", "weekday_mean4": "same-weekday mean (4 wk)",
        "ets": "AutoETS", "chronos-bolt-small": "Chronos-Bolt-Small", "ci-proto": "ci-proto",
        "ci-proto-cohort8": "ci-proto, cohorts of 8 ATMs"}
plt.rcParams.update({"font.size": 15, "axes.titlesize": 19, "axes.labelsize": 16,
                     "figure.facecolor": "white", "axes.facecolor": "white",
                     "text.color": "#111", "axes.labelcolor": "#111"})


def boot_ci(x, stat=np.mean):
    x = np.asarray(x, dtype=float)
    idx = RNG.integers(0, len(x), (B, len(x)))
    s = np.array([stat(x[i]) for i in idx])
    return float(stat(x)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def pinball(a, q, tau):
    return 2 * np.maximum(tau * (a - q), (tau - 1) * (a - q))


def score(df, scale_of, hmax):
    """Per (unit, window) MASE, SQL and 10-90 coverage over horizons 1..hmax."""
    d = df[(df.h <= hmax) & df.actual.notna()].copy()
    d["scale"] = [scale_of[(u, w)] for u, w in zip(d.unit, d.window)]
    d["ae"] = (d.actual - d.q50).abs() / d.scale
    probabilistic = "q10" in d and d["q10"].notna().all()
    if probabilistic:
        ql = sum(pinball(d.actual, d[c], t) for c, t in zip(QCOLS, QUANTILES)) / len(QCOLS)
        d["sql"] = ql / d.scale
        d["ql"] = ql
        d["cov"] = ((d.actual >= d.q10) & (d.actual <= d.q90)).astype(float)
    g = d.groupby(["unit", "window"])
    out = g.agg(mase=("ae", "mean"))
    if probabilistic:
        out["sql"] = g.sql.mean()
        out["cov"] = g["cov"].mean()
        out["ql_sum"] = g.ql.sum()
        out["abs_sum"] = g.actual.apply(lambda s: s.abs().sum())
    return out.reset_index()


def per_unit(sc):
    return sc.groupby("unit").mean(numeric_only=True)


def summarize(pu):
    r = {"n": int(len(pu))}
    for m in ("mase", "sql", "cov"):
        if m in pu:
            r[m] = boot_ci(pu[m].values)
    if "ql_sum" in pu:  # WQL: ratio of sums, bootstrap over units
        v = pu[["ql_sum", "abs_sum"]].values
        f = lambda i: v[i, 0].sum() / v[i, 1].sum()
        idx = RNG.integers(0, len(v), (B, len(v)))
        s = [f(i) for i in idx]
        r["wql"] = (float(f(np.arange(len(v)))), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))
    return r


def fmt(t, d=3):
    return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


# ---------------------------------------------------------------- NN5
ids, dates, y = load_nn5()
origins = nn5_origins(y.shape[1])
nn5_scale = {(a, w): seasonal_scale(y[i, :o], 7) for i, a in enumerate(ids) for w, o in enumerate(origins)}


def load_nn5_run(path):
    df = pd.read_csv(path).rename(columns={"atm": "unit"})
    return df


files = sorted(glob.glob(str(RUNS / "nn5" / "*.csv.gz")))
runs = {}
for f in files:
    m = re.match(r"(.+)_ctx(\w+)_run(\d+)\.csv\.gz", f.split("/")[-1])
    runs.setdefault((m[1], m[2]), []).append(load_nn5_run(f))

summary = {"nn5": {}, "nn5_context": {}, "long_context": {}, "cash_ops": {}, "runs_spread": {}}
per_atm, ctx_pu, long_pu = {}, {}, {}
for (meth, ctx), dfs in runs.items():
    for hmax in (14, 7):
        scs = [per_unit(score(d, nn5_scale, hmax)) for d in dfs]
        pu = sum(scs) / len(scs)  # metric averaged over repeated runs (not forecasts)
        s = summarize(pu)
        s["runs"] = len(dfs)
        s["run_means_mase"] = [float(x.mase.mean()) for x in scs]
        if ctx == "all":
            summary["nn5"][f"{meth}@h{hmax}"] = s
            if hmax == 14:
                per_atm[meth] = pu
        if hmax == 14:
            summary["nn5_context"][f"{meth}@{ctx}"] = s
            ctx_pu[(meth, ctx)] = pu

pd.concat({k: v for k, v in per_atm.items()}, names=["method"]).to_csv(RUNS / "metrics_nn5_per_atm.csv", float_format="%.4f")

# share of ATMs where each FM beats seasonal naive (MASE, h=14, full history)
sn = per_atm["seasonal_naive"].mase
for meth in ("ci-proto", "ci-proto-cohort8", "chronos-bolt-small", "ets", "weekday_mean4"):
    if meth in per_atm:
        win = (per_atm[meth].mase < sn).astype(float).values
        summary["nn5"][f"{meth}@share_beats_snaive"] = boot_ci(win)
        summary["nn5"][f"{meth}@count_beats_snaive"] = [int(win.sum()), len(win)]
if "chronos-bolt-small" in per_atm:
    win = (per_atm["ci-proto"].mase < per_atm["chronos-bolt-small"].mase).astype(float).values
    summary["nn5"]["ci-proto@share_beats_chronos"] = boot_ci(win)
    summary["nn5"]["ci-proto@count_beats_chronos"] = [int(win.sum()), len(win)]

# ---------------------------------------------------------------- cash ops (illustrative)
def cash(df, col, mult=1.0):
    d = df[(df.h <= 7) & df.actual.notna()]
    stock = d[col] * mult
    short = (d.actual > stock).groupby(d.unit).mean()
    idle = np.maximum(stock - d.actual, 0).groupby(d.unit).sum()
    tot = d.actual.groupby(d.unit).sum()
    return short, idle, tot


def cash_summary(short, idle, tot):
    v = np.c_[short.values, idle.values, tot.values]
    idx = RNG.integers(0, len(v), (B, len(v)))
    sh = [v[i, 0].mean() for i in idx]
    il = [v[i, 1].sum() / v[i, 2].sum() for i in idx]
    return {"shortfall_rate": (float(v[:, 0].mean()), float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))),
            "idle_pct_of_withdrawals": (float(v[:, 1].sum() / v[:, 2].sum()), float(np.percentile(il, 2.5)), float(np.percentile(il, 97.5)))}


ci_df = runs[("ci-proto", "all")][0]
target = cash(ci_df, "q90")[0].mean()
summary["cash_ops"]["ci-proto q90 (run 0)"] = cash_summary(*cash(ci_df, "q90"))
if ("chronos-bolt-small", "all") in runs:
    summary["cash_ops"]["chronos q90"] = cash_summary(*cash(runs[("chronos-bolt-small", "all")][0], "q90"))
summary["cash_ops"]["AutoETS q90"] = cash_summary(*cash(runs[("ets", "all")][0], "q90"))
sn_df = runs[("seasonal_naive", "all")][0]
for b in np.arange(1.0, 4.0, 0.01):
    if cash(sn_df, "q50", b)[0].mean() <= target:
        break
summary["cash_ops"][f"seasonal naive x {b:.2f} (buffer matched to ci-proto shortfall)"] = cash_summary(*cash(sn_df, "q50", b))
summary["cash_ops"]["matched_buffer"] = round(float(b), 2)

# ---------------------------------------------------------------- long series
long_meta = {}
for f in sorted(glob.glob(str(RUNS / "long" / "*.json"))):
    m = json.loads(open(f).read())
    long_meta[(m["method"], m["context"])] = m
long_scale = {}
if long_meta:
    from run_long import load_clients, H, WINDOWS
    cids, ly, _ = load_clients()
    T = ly.shape[1]
    lorig = [T - H * k for k in range(WINDOWS, 0, -1)]
    long_scale = {(c, w): seasonal_scale(ly[i, :o], 96) for i, c in enumerate(cids) for w, o in enumerate(lorig)}
    for (meth, ctx), m in sorted(long_meta.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        df = pd.read_csv(RUNS / "long" / f"{meth}_ctx{ctx}.csv.gz").rename(columns={"client": "unit"})
        scs = [per_unit(score(d, long_scale, H)) for _, d in df.groupby("run")]
        pu = sum(scs) / len(scs)
        s = summarize(pu)
        s.update({"runs": len(scs), "run_means_mase": [float(x.mase.mean()) for x in scs],
                  "sec_per_forecast": m["seconds_per_forecast_median"], "peak_rss_mb": m["peak_rss_mb"]})
        summary["long_context"][f"{meth}@{ctx}"] = s
        long_pu[(meth, ctx)] = pu
for f in sorted(glob.glob(str(RUNS / "scaling" / "*.json"))):
    m = json.loads(open(f).read())
    summary.setdefault("scaling", {})[str(m["context"])] = m

# paired differences in MASE (context c minus reference), bootstrap over series
summary["paired"] = {}
for (meth, ref), store, name in ((("ci-proto", "all"), ctx_pu, "nn5"), (("chronos-bolt-small", "all"), ctx_pu, "nn5"),
                                 (("ci-proto", 2048), long_pu, "long"), (("chronos-bolt-small", 2048), long_pu, "long")):
    if (meth, ref) not in store:
        continue
    for (m2, c), pu in store.items():
        if m2 == meth and c != ref:
            d = (pu.mase - store[(meth, ref)].mase).values
            summary["paired"][f"{name}:{meth}:{c}-{ref}"] = boot_ci(d)

(RUNS / "summary.json").write_text(json.dumps(summary, indent=1))

# ---------------------------------------------------------------- print tables
print(f"## NN5 head to head (full history, n={len(ids)} ATMs x {NN5_WINDOWS} windows x 14 days, mean [95% bootstrap CI over ATMs])")
print("| method | runs | MASE h14 | MASE h7 | SQL h14 | WQL h14 | 10-90% coverage | beats seasonal naive (share of ATMs) |")
print("|---|---|---|---|---|---|---|---|")
order = ["seasonal_naive", "weekday_mean4", "ets", "chronos-bolt-small", "ci-proto"]
for meth in order + ["ci-proto-cohort8"]:
    if f"{meth}@h14" not in summary["nn5"]:
        continue
    s14, s7 = summary["nn5"][f"{meth}@h14"], summary["nn5"][f"{meth}@h7"]
    share = summary["nn5"].get(f"{meth}@share_beats_snaive")
    print(f"| {NICE[meth]} | {s14['runs']} | {fmt(s14['mase'])} | {fmt(s7['mase'])} | "
          f"{fmt(s14['sql']) if 'sql' in s14 else '-'} | {fmt(s14['wql']) if 'wql' in s14 else '-'} | "
          f"{fmt(s14['cov'], 2) if 'cov' in s14 else '-'} | {fmt(share, 2) if share else '-'} |")
if "ci-proto@share_beats_chronos" in summary["nn5"]:
    print("\nci-proto beats Chronos-Bolt-Small on", fmt(summary["nn5"]["ci-proto@share_beats_chronos"], 2), "of ATMs",
          summary["nn5"]["ci-proto@count_beats_chronos"])
print("beats seasonal naive counts:", {k: v for k, v in summary["nn5"].items() if "count_beats_snaive" in k})
print("\nrun-to-run MASE means:", {k: [round(x, 4) for x in v["run_means_mase"]] for k, v in summary["nn5"].items() if "run_means_mase" in v and v["runs"] > 1})

print("\n## NN5 context ablation (MASE h14)")
print("| context (days) | ci-proto | Chronos-Bolt-Small |")
print("|---|---|---|")
ctxs = ["14", "28", "56", "112", "224", "448", "all"]
for c in ctxs:
    a = summary["nn5_context"].get(f"ci-proto@{c}")
    b_ = summary["nn5_context"].get(f"chronos-bolt-small@{c}")
    print(f"| {c} | {fmt(a['mase']) if a else '-'} | {fmt(b_['mase']) if b_ else '-'} |")

print("\n## Cash-ops (illustrative; first 7 days of each window, n=111 ATMs)")
print("| rule | ATM-days short | idle cash as % of withdrawals |")
print("|---|---|---|")
for k, v in summary["cash_ops"].items():
    if isinstance(v, dict):
        print(f"| {k} | {fmt(v['shortfall_rate'], 3)} | {fmt(tuple(100 * x for x in v['idle_pct_of_withdrawals']), 1)} |")

if summary["long_context"]:
    print("\n## Long series context ablation (electricity 15-min, n=8 clients x 3 days, MASE m=96)")
    print("| method | context | runs | MASE | SQL | coverage | s/forecast | peak RSS MB |")
    print("|---|---|---|---|---|---|---|---|")
    for k, s in summary["long_context"].items():
        meth, ctx = k.split("@")
        print(f"| {meth} | {ctx} | {s['runs']} | {fmt(s['mase'])} | {fmt(s['sql']) if 'sql' in s else '-'} | "
              f"{fmt(s['cov'], 2) if 'cov' in s else '-'} | {s['sec_per_forecast']} | {s['peak_rss_mb']} |")
print("\n## Paired MASE differences (context minus reference; negative = context helps)")
for k, v in summary["paired"].items():
    print(k, fmt(v))
if "scaling" in summary:
    print("\n## CPU cost per forecast (1 series, after warm-up)")
    for k, m in summary["scaling"].items():
        print(k, m)

# ---------------------------------------------------------------- images
IMG.mkdir(exist_ok=True)
COL = {"seasonal_naive": "#9e9e9e", "weekday_mean4": "#6d6d6d", "ets": "#8c6d31",
       "chronos-bolt-small": "#2a7ab9", "ci-proto": "#d1495b"}

# 1. head to head
fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
ms = [m for m in order if f"{m}@h14" in summary["nn5"]]
vals = [summary["nn5"][f"{m}@h14"]["mase"] for m in ms]
ax.barh(range(len(ms)), [v[0] for v in vals], xerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
        color=[COL[m] for m in ms], capsize=8)
for i, v in enumerate(vals):
    ax.text(v[2] + 0.01, i, f"{v[0]:.3f}", va="center", fontsize=16)
ax.set_yticks(range(len(ms)), [NICE[m] for m in ms])
ax.set_xlabel("MASE, 14-day horizon (lower is better), mean over ATMs with 95% bootstrap CI")
ax.set_title(f"NN5: {len(ids)} UK ATMs, {NN5_WINDOWS} rolling 14-day windows, full history")
ax.set_xlim(0, max(v[2] for v in vals) * 1.15)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(IMG / "1_head_to_head_mase.png")

# 2. context curves
fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=100)
ax = axes[0]
for meth in ("ci-proto", "chronos-bolt-small"):
    pts = [(c, summary["nn5_context"][f"{meth}@{c}"]["mase"]) for c in ctxs if f"{meth}@{c}" in summary["nn5_context"]]
    if not pts:
        continue
    xs = [int(c) if c != "all" else 777 for c, _ in pts]  # 'all' = 679..777 days depending on window
    ax.errorbar(xs, [p[0] for _, p in pts], yerr=[[p[0] - p[1] for _, p in pts], [p[2] - p[0] for _, p in pts]],
                marker="o", capsize=5, lw=2.5, color=COL[meth], label=NICE[meth])
snv = summary["nn5"]["seasonal_naive@h14"]["mase"][0]
ax.axhline(snv, ls="--", color="#777", label=f"seasonal naive ({snv:.3f})")
ax.set_xscale("log", base=2)
ax.set_xticks([14, 28, 56, 112, 224, 448, 777], ["14", "28", "56", "112", "224", "448", "all"])
ax.set_xlabel("days of history given to the model")
ax.set_ylabel("MASE, 14-day horizon")
ax.set_title("NN5 ATM cash (n=111)")
ax.legend(fontsize=13)
ax = axes[1]
for meth in ("ci-proto", "chronos-bolt-small"):
    pts = sorted((int(k.split("@")[1]), s["mase"]) for k, s in summary["long_context"].items() if k.startswith(meth + "@"))
    if pts:
        ax.errorbar([c for c, _ in pts], [p[0] for _, p in pts], yerr=[[p[0] - p[1] for _, p in pts], [p[2] - p[0] for _, p in pts]],
                    marker="o", capsize=5, lw=2.5, color=COL[meth], label=NICE[meth])
if "seasonal_naive@512" in summary["long_context"]:
    v = summary["long_context"]["seasonal_naive@512"]["mase"][0]
    ax.axhline(v, ls="--", color="#777", label=f"repeat last day ({v:.3f})")
ax.set_xscale("log", base=2)
lt = [512, 2048, 8192, 16384, 32768, 65536]
ax.set_xticks(lt, ["512", "2K", "8K", "16K", "32K", "64K"])
ax.set_xlabel("steps of history (15-min; 2K = 21 days, 64K = 683 days)")
ax.set_ylabel("MASE, 1-day horizon (96 steps)")
ax.set_title("Electricity load, 15-min (n=8 clients)")
ax.axvline(16384, color="#bbb", lw=1)
ax.text(16384 * 1.08, ax.get_ylim()[0], "long checkpoint →", va="bottom", fontsize=12, color="#555")
ax.legend(fontsize=13)
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Does more history help? MASE vs context length (95% CI over series)", fontsize=21)
fig.tight_layout()
fig.savefig(IMG / "2_context_length.png")

# 3. one real ATM: the median-MASE ATM for ci-proto (full history), last window
pu = per_atm["ci-proto"].mase.sort_values()
atm = pu.index[len(pu) // 2]
i = ids.index(atm)
w = ci_df[(ci_df.unit == atm) & (ci_df.window == NN5_WINDOWS - 1)].sort_values("h")
o = origins[-1]
fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
hd = dates[o - 70:o]
ax.plot(hd, y[i, o - 70:o], color="#333", lw=2, label="history (cash withdrawn per day)")
fd = pd.to_datetime(w.date)
ax.fill_between(fd, w.q10, w.q90, color=COL["ci-proto"], alpha=0.2, label="ci-proto 10-90% band")
ax.plot(fd, w.q50, color=COL["ci-proto"], lw=2.5, label="ci-proto median")
ax.plot(fd, w.actual, "o-", color="#111", ms=7, lw=1.5, label="actual (held out)")
ax.axvline(dates[o], color="#999", ls=":")
ax.set_title(f"{atm}: the median-error ATM for ci-proto (MASE {pu[atm]:.2f}), last 14-day window")
ax.set_ylabel("daily withdrawals (NN5 units)")
ax.legend(fontsize=13, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(IMG / "3_atm_forecast.png")
summary["example_atm"] = {"atm": atm, "mase_ci_proto": float(pu[atm]), "window": NN5_WINDOWS - 1, "origin": str(dates[o].date())}
(RUNS / "summary.json").write_text(json.dumps(summary, indent=1))
print("\nexample ATM:", summary["example_atm"])

# 4. CPU cost: median seconds per forecast and peak RSS from the electricity runs,
# plus the scaling probe (fresh process per context) for where the RAM cap stopped it.
cp = sorted((m["context"], m) for (meth, c), m in long_meta.items() if meth == "ci-proto")
if cp:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    xs = [c for c, _ in cp]
    ys = [m["seconds_per_forecast_median"] for _, m in cp]
    ax.plot(xs, ys, "o-", color=COL["ci-proto"], lw=2.5, ms=9, label="ci-proto, median s per forecast (electricity runs)")
    for c, m in cp:
        ax.annotate(f"{m['seconds_per_forecast_median']:.1f}s\n{m['peak_rss_mb'] / 1024:.1f} GB", (c, m["seconds_per_forecast_median"]),
                    textcoords="offset points", xytext=(0, 14), ha="center", fontsize=13)
    for m in summary.get("scaling", {}).values():
        if m.get("status") != "ok":
            ax.axvline(m["context"], color="#999", ls="--")
            ax.text(m["context"] * 0.93, max(ys) * 0.15, f"{m['context']:,} steps:\nstopped, RSS passed 3,000 MB cap", ha="right", fontsize=13)
    ax.axvline(16384, color="#bbb", lw=1)
    ax.text(16384 * 1.05, min(ys), "long checkpoint above 16,384", fontsize=12, color="#555")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(min(ys) / 2, max(ys) * 4)
    ax.set_xticks([512, 2048, 8192, 16384, 32768, 65536, 131072], ["512", "2K", "8K", "16K", "32K", "64K", "128K"])
    ax.set_xlabel("context length (steps), one series per call, 8 CPU threads")
    ax.set_ylabel("seconds per forecast (median)")
    ax.set_title("ci-proto on a CPU box: time and peak RAM per forecast")
    ax.legend(loc="upper left", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG / "4_cpu_cost.png")
