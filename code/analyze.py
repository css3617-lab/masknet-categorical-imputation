"""Aggregate per-setting results into the manuscript's tables, statistics and figures.

python analyze.py  -> results/summary.json, results/tables.md, figures/*.pdf
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = f"{ROOT}/results", f"{ROOT}/figures"
os.makedirs(FIG, exist_ok=True)
ORDER = ["Mode", "kNN", "MICE", "MICE-S", "MissForest", "GAIN", "MIDAS-DAE", "ReMasker", "DAE-complete",
         "DAE-complete-unw", "Noise DNN", "Noise DNN-unw", "MaskNet", "MaskNet-S"]
REF = "MaskNet"
METRICS = ["acc", "f1", "bal", "tvd", "coef"]
HIGHER = {"acc": True, "f1": True, "bal": True, "tvd": False, "coef": False}


def load(mode):
    rows = []
    for f in sorted(glob.glob(f"{RES}/{mode}/*.json")):
        rows += json.load(open(f))
    df = pd.DataFrame(rows)
    df["setting"] = df.dataset + "|" + df.mech + "|" + df.rate.astype(str) + "|" + df.seed.astype(str) + "|" + df.n_train.astype(str)
    df["config"] = df.dataset + "|" + df.mech + "|" + df.rate.astype(str)
    return df


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, (len(p) - i) * p[k])
        adj[k] = min(1.0, run)
    return adj


def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def cluster_boot_ci(diff_by_cfg, B=10000, seed=0):
    """95% CI of the mean paired difference, resampling configurations (seeds kept together)."""
    rng = np.random.default_rng(seed)
    keys = list(diff_by_cfg)
    arr = [np.asarray(diff_by_cfg[k]) for k in keys]
    means = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        means.append(np.concatenate([arr[i] for i in pick]).mean())
    return np.percentile(means, [2.5, 97.5])


def paired(df, metric, methods, ref=REF):
    piv = df.pivot_table(index="setting", columns="method", values=metric)
    cfg = df.drop_duplicates("setting").set_index("setting")["config"]
    out = {}
    for m in methods:
        if m == ref or m not in piv:
            continue
        d = (piv[ref] - piv[m]).dropna()                        # positive = MaskNet larger
        if not HIGHER[metric]:
            d = -d                                              # positive = MaskNet better
        p54 = wilcoxon(d, zero_method="wilcox").pvalue if np.any(d != 0) else 1.0
        dc = d.groupby(cfg.loc[d.index]).mean()
        p18 = wilcoxon(dc, zero_method="wilcox").pvalue if np.any(dc != 0) else 1.0
        r3 = np.round(piv[ref].loc[d.index], 3) - np.round(piv[m].loc[d.index], 3)
        if not HIGHER[metric]:
            r3 = -r3
        ci = cluster_boot_ci({k: v.values for k, v in d.groupby(cfg.loc[d.index])})
        out[m] = dict(mean=float(d.mean()), median=float(d.median()), ci=[float(ci[0]), float(ci[1])],
                      sd=float(d.std(ddof=1)), rb=rank_biserial(d.values), p54=float(p54), p18=float(p18),
                      win=int((r3 > 0).sum()), tie=int((r3 == 0).sum()), loss=int((r3 < 0).sum()), n=int(len(d)))
    ms = list(out)
    for key in ["p54", "p18"]:
        adj = holm([out[m][key] for m in ms])
        for m, a in zip(ms, adj):
            out[m][key + "_holm"] = float(a)
    return out


def marker(p):
    return "‡" if p < 0.001 else "†" if p < 0.01 else "*" if p < 0.05 else ""


def main():
    df = load("main")
    methods = [m for m in ORDER if m in df.method.unique()]
    n_set = df.setting.nunique()
    S = {"n_settings": n_set}
    # ---------------------------------------------------------------- Table 2
    mean = df.groupby("method")[METRICS + ["time"]].mean()
    sd_seed = df.groupby(["method", "config"])[METRICS].std().groupby("method").mean()
    acc = df.pivot_table(index="setting", columns="method", values="acc")
    ranks = acc[methods].rank(axis=1, ascending=False).mean()
    stats = {m_: paired(df, m_, methods) for m_ in METRICS}
    T2 = []
    for m in methods:
        r = dict(method=m, rank=float(ranks[m]), time=float(mean.loc[m, "time"]))
        for k in METRICS:
            r[k] = float(mean.loc[m, k]); r[k + "_sd"] = float(sd_seed.loc[m, k])
            r[k + "_mark"] = marker(stats[k][m]["p54_holm"]) if m in stats[k] else ""
        T2.append(r)
    S["table2"] = T2
    S["paired"] = stats
    # MaskNet-S draw variability
    ms = df[df.method == "MaskNet-S"]
    S["masknet_s_draw_sd"] = {k: float(ms[k + "_sd"].mean()) for k in METRICS}
    S["masknet_s_draw_sd_max"] = {k: float(ms[k + "_sd"].max()) for k in METRICS}
    # ---------------------------------------------------------------- Table 3 (dataset x mechanism)
    df["dm"] = df.dataset.map({"brfss": "B", "diabetes": "D"}) + "-" + df.mech
    T3 = df.pivot_table(index="method", columns="dm", values="acc").loc[methods]
    S["table3"] = T3.round(4).to_dict()
    # by rate and mechanism (Fig. 3 data) and per-rate summaries
    S["acc_by_rate"] = df.groupby(["method", "rate"]).acc.mean().unstack().loc[methods].round(4).to_dict()
    S["coef_by_rate"] = df.groupby(["method", "rate"]).coef.mean().unstack().loc[methods].round(4).to_dict()
    S["tvd_by_rate"] = df.groupby(["method", "rate"]).tvd.mean().unstack().loc[methods].round(4).to_dict()
    S["acc_by_mech"] = df.groupby(["method", "mech"]).acc.mean().unstack().loc[methods].round(4).to_dict()
    S["coef_by_mech"] = df.groupby(["method", "mech"]).coef.mean().unstack().loc[methods].round(4).to_dict()
    S["tvd_by_mech"] = df.groupby(["method", "mech"]).tvd.mean().unstack().loc[methods].round(4).to_dict()
    S["mcar_drop_10_50"] = {m: float(df[(df.method == m) & (df.mech == "MCAR") & (df.rate == 0.1)].acc.mean()
                                     - df[(df.method == m) & (df.mech == "MCAR") & (df.rate == 0.5)].acc.mean())
                            for m in methods}
    # ---------------------------------------------------------------- Table 4 (MCAR re-run)
    mc = df[df.mech == "MCAR"]
    T4 = {}
    for (ds, rate), g in mc.groupby(["dataset", "rate"]):
        a = g.groupby("method")[["acc", "f1"]].mean()
        T4[f"{ds}|{rate}"] = {k: float(a.loc[m, c]) for k, m, c in
                              [("DAEc_acc", "DAE-complete", "acc"), ("NDNN_acc", "Noise DNN", "acc"),
                               ("DAEc_f1", "DAE-complete", "f1"), ("NDNN_f1", "Noise DNN", "f1"),
                               ("DAEcu_acc", "DAE-complete-unw", "acc"), ("NDNNu_acc", "Noise DNN-unw", "acc"),
                               ("MaskNet_acc", "MaskNet", "acc")]}
    S["table4"] = T4
    # ---------------------------------------------------------------- MaskNet vs Noise DNN-unw by condition
    piv = df.pivot_table(index=["dataset", "mech", "rate", "seed"], columns="method", values="acc")
    dd = (piv["MaskNet"] - piv["Noise DNN-unw"]).rename("d").reset_index()
    dd["win"] = np.round(piv["MaskNet"].values, 3) > np.round(piv["Noise DNN-unw"].values, 3)
    dd["loss"] = np.round(piv["MaskNet"].values, 3) < np.round(piv["Noise DNN-unw"].values, 3)
    S["mn_vs_ndnn_cells"] = dd.groupby(["dataset", "mech", "rate"]).agg(d=("d", "mean"), win=("win", "sum"), loss=("loss", "sum")).reset_index().to_dict("records")
    for fac in ["dataset", "mech", "rate"]:
        S[f"mn_vs_ndnn_by_{fac}"] = dd.groupby(fac).agg(d=("d", "mean"), win=("win", "sum"), loss=("loss", "sum")).reset_index().to_dict("records")
    # per-variable: where does each win (30%, pooled over mechanisms/seeds)
    pv = {}
    for ds in ["brfss", "diabetes"]:
        meta = json.load(open(f"{ROOT}/data/{ds}_meta.json"))
        g = df[(df.dataset == ds) & (df.rate == 0.3)]
        av = {}
        for m in ["Mode", "MaskNet", "Noise DNN", "Noise DNN-unw", "GAIN", "MICE"]:
            arr = np.array([[np.nan if v is None else v for v in r] for r in g[g.method == m].acc_var])
            av[m] = np.nanmean(arr, 0)
        pv[ds] = {"names": meta["names"], "K": meta["K"], **{m: v.round(4).tolist() for m, v in av.items()}}
    S["per_variable_30"] = pv
    # ---------------------------------------------------------------- ablation (30 %)
    ab = load("ablation")
    full = ab[ab.method == "full"].set_index("setting")
    AB = []
    for m in ab.method.unique():
        g = ab[ab.method == m].set_index("setting")
        dlt = (g.acc - full.acc.loc[g.index])
        p = wilcoxon(dlt).pvalue if np.any(dlt != 0) else 1.0
        AB.append(dict(method=m, acc=float(g.acc.mean()), delta=float(dlt.mean()), p=float(p), f1=float(g.f1.mean()),
                       tvd=float(g.tvd.mean()), mnar=float(g[g.mech == "MNAR"].acc.mean()), time=float(g.time.mean())))
    # reference methods from the main run at 30 %
    m30 = df[df.rate == 0.3]
    fullmain = m30[m30.method == "MaskNet"].set_index("setting")
    for m in ["MIDAS-DAE", "ReMasker", "Noise DNN-unw", "DAE-complete-unw"]:
        g = m30[m30.method == m].set_index("setting")
        if len(g) == 0:
            continue
        dlt = g.acc - fullmain.acc.loc[g.index]
        AB.append(dict(method="ref: " + m, acc=float(g.acc.mean()), delta=float(dlt.mean()), p=float(wilcoxon(dlt).pvalue),
                       f1=float(g.f1.mean()), tvd=float(g.tvd.mean()), mnar=float(g[g.mech == "MNAR"].acc.mean()),
                       time=float(g.time.mean())))
    S["ablation"] = AB
    S["ablation_full_main30"] = float(fullmain.acc.mean())
    # ---------------------------------------------------------------- training size
    sz = load("size")
    m30 = df[(df.rate == 0.3) & (df.mech.isin(["MCAR", "MNAR"])) & (df.method.isin(sz.method.unique()))]
    allsz = pd.concat([sz, m30])
    S["size"] = allsz.groupby(["method", "n_train", "mech"]).agg(acc=("acc", "mean"), time=("time", "mean")).reset_index().to_dict("records")
    json.dump(S, open(f"{RES}/summary.json", "w"), indent=1, default=float)
    figures(df, S, methods)
    print_tables(S, methods)


def figures(df, S, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    T2 = {r["method"]: r for r in S["table2"]}
    col = {"MaskNet": "#d62728", "MaskNet-S": "#ff9896", "Noise DNN-unw": "#1f77b4", "Noise DNN": "#1f77b4",
           "DAE-complete": "#9467bd", "DAE-complete-unw": "#9467bd", "ReMasker": "#2ca02c", "MICE": "#8c564b",
           "MICE-S": "#c49c94", "MissForest": "#e377c2", "kNN": "#7f7f7f", "Mode": "#000000", "GAIN": "#bcbd22",
           "MIDAS-DAE": "#17becf"}
    weighted = {"Noise DNN", "DAE-complete"}
    # Fig. 2 trade-offs
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for k, (yk, yl) in enumerate([("bal", "Balanced accuracy"), ("tvd", "Distributional error (TVD)")]):
        for m in methods:
            r = T2[m]
            ax[k].scatter(r["acc"], r[yk], s=28, marker="s" if m in weighted else "o", color=col[m],
                          edgecolor="k", linewidth=0.4, zorder=3)
            off = {"MaskNet": (3, -7), "Noise DNN-unw": (-8, 5), "DAE-complete-unw": (3, 4), "MICE": (-14, 5), "MissForest": (3, -6),
                   "MIDAS-DAE": (-30, 4), "ReMasker": (3, 2), "kNN": (3, 2)}.get(m, (3, 2))
            ax[k].annotate(m, (r["acc"], r[yk]), fontsize=5.5, xytext=off, textcoords="offset points")
        ax[k].set_xlabel("Accuracy"); ax[k].set_ylabel(yl); ax[k].grid(alpha=0.3)
        ax[k].set_title(f"({'ab'[k]})", loc="left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_tradeoffs.pdf"); plt.close(fig)
    # Fig. 3 accuracy vs rate
    show = ["Mode", "MICE", "MissForest", "GAIN", "MIDAS-DAE", "ReMasker", "DAE-complete", "Noise DNN", "Noise DNN-unw", "MaskNet"]
    fig, ax = plt.subplots(2, 3, figsize=(7.0, 4.4), sharex=True)
    for i, ds in enumerate(["brfss", "diabetes"]):
        for j, mech in enumerate(["MCAR", "MAR", "MNAR"]):
            a = ax[i, j]
            g = df[(df.dataset == ds) & (df.mech == mech)].groupby(["method", "rate"]).acc.mean().unstack()
            for m in [m for m in show if m in g.index]:
                a.plot([10, 30, 50], g.loc[m].values, marker="o", ms=3, lw=1.2, color=col[m],
                       ls="--" if m in weighted else "-", label=m)
            a.set_title(f"{'BRFSS' if ds == 'brfss' else 'Diabetes 130-US'}, {mech}", fontsize=8)
            a.grid(alpha=0.3)
            if i == 1: a.set_xlabel("Missing rate (%)")
            if j == 0: a.set_ylabel("Accuracy")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{FIG}/fig3_acc_vs_rate.pdf"); plt.close(fig)
    # Fig. 4 per-variable gain over mode
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.6), gridspec_kw={"width_ratios": [22, 27]})
    for k, ds in enumerate(["brfss", "diabetes"]):
        pv = S["per_variable_30"][ds]
        x = np.arange(len(pv["names"]))
        g1 = np.array(pv["MaskNet"]) - np.array(pv["Mode"])
        g2 = np.array(pv["Noise DNN"]) - np.array(pv["Mode"])
        ax[k].bar(x - 0.2, 100 * g1, 0.4, color=col["MaskNet"], label="MaskNet")
        ax[k].bar(x + 0.2, 100 * g2, 0.4, color="#1f77b4", label="Noise DNN (weighted)")
        ax[k].axhline(0, color="k", lw=0.6)
        ax[k].set_xticks(x); ax[k].set_xticklabels(pv["names"], rotation=90, fontsize=5)
        ax[k].set_title("BRFSS 2015" if ds == "brfss" else "Diabetes 130-US", fontsize=8)
        ax[k].grid(axis="y", alpha=0.3)
    ax[0].set_ylabel("Accuracy gain over mode (points)")
    ax[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_per_variable.pdf"); plt.close(fig)


def print_tables(S, methods):
    print(f"settings: {S['n_settings']}")
    print(f"{'method':18s} acc   f1    bal   tvd   coef  time  rank")
    for r in S["table2"]:
        print(f"{r['method']:18s} {r['acc']:.3f}{r['acc_mark']:1s} {r['f1']:.3f}{r['f1_mark']:1s} {r['bal']:.3f}{r['bal_mark']:1s} "
              f"{r['tvd']:.3f}{r['tvd_mark']:1s} {r['coef']:.3f}{r['coef_mark']:1s} {r['time']:5.1f} {r['rank']:.1f}")
    print("\npaired accuracy vs MaskNet (pp):")
    for m, s in S["paired"]["acc"].items():
        print(f"{m:18s} mean {100*s['mean']:+.2f} [{100*s['ci'][0]:+.2f},{100*s['ci'][1]:+.2f}] med {100*s['median']:+.2f} "
              f"W/T/L {s['win']}/{s['tie']}/{s['loss']} rb {s['rb']:+.2f} p54 {s['p54_holm']:.2g} p18 {s['p18_holm']:.2g}")


if __name__ == "__main__":
    main()
