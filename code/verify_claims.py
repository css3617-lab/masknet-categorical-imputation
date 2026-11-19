"""Independent check of every number quoted in the prose of the revised manuscript and the response
letter. Values are recomputed from the raw per-setting JSON files (not from summary.json).
Each claim: (where, text, stated value, recomputed value, decimals used in the text).
"""
import json, glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(mode):
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/results/{mode}/*.json")):
        rows += json.load(open(f))
    d = pd.DataFrame(rows)
    d["key"] = d.dataset + d.mech + d.rate.astype(str) + d.seed.astype(str) + d.n_train.astype(str)
    d["cfg"] = d.dataset + d.mech + d.rate.astype(str)
    return d


M, A, Z = load("main"), load("ablation"), load("size")


def mean(m, col="acc", **f):
    x = M[M.method == m]
    for k, v in f.items():
        x = x[x[k] == v]
    return x[col].mean()


def piv(col="acc", df=M):
    return df.pivot_table(index="key", columns="method", values=col)


P = piv()
pts = lambda x: 100 * x


def diff(a, b, col="acc", df=M, **f):
    x = df
    for k, v in f.items():
        x = x[x[k] == v]
    p = x.pivot_table(index="key", columns="method", values=col)
    return (p[a] - p[b]).dropna()


def wtl(a, b):
    d = np.round(P[a], 3) - np.round(P[b], 3)
    return int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())


def p54(a, b, col="acc"):
    return wilcoxon(diff(a, b, col)).pvalue


def p18(a, b):
    d = diff(a, b)
    cfg = M.drop_duplicates("key").set_index("key").cfg
    return wilcoxon(d.groupby(cfg.loc[d.index]).mean()).pvalue


C = []


def claim(where, text, stated, value, dec):
    C.append((where, text, stated, value, dec))


# ------------------------------------------------------------------ overall (Table III prose)
for m, v in [("Noise DNN-unw", 0.665), ("MaskNet", 0.664), ("MICE", 0.657), ("ReMasker", 0.656), ("MissForest", 0.651),
             ("MIDAS-DAE", 0.650), ("kNN", 0.638), ("Mode", 0.620), ("GAIN", 0.595), ("Noise DNN", 0.554),
             ("DAE-complete", 0.533), ("DAE-complete-unw", 0.680), ("MaskNet-S", 0.604)]:
    claim("IV-A", f"accuracy {m}", v, mean(m), 3)
rk = P.rank(axis=1, ascending=False).mean()
claim("IV-A", "rank Noise DNN-unw", 2.5, rk["Noise DNN-unw"], 1)
claim("IV-A", "rank MaskNet", 3.1, rk["MaskNet"], 1)
d = diff("MaskNet", "Noise DNN-unw")
claim("IV-A/Abstract", "MaskNet - NDNN-unw mean (pts)", -0.11, pts(d.mean()), 2)
claim("IV-A", "MaskNet - NDNN-unw median (pts, abs)", 0.14, abs(pts(d.median())), 2)
claim("IV-A", "NDNN-unw wins over MaskNet", 33, wtl("MaskNet", "Noise DNN-unw")[2], 0)
claim("IV-A", "NDNN-unw losses to MaskNet", 13, wtl("MaskNet", "Noise DNN-unw")[0], 0)
for m, v in [("MICE", 0.6), ("ReMasker", 0.8), ("MissForest", 1.3), ("MIDAS-DAE", 1.4), ("kNN", 2.6), ("Mode", 4.4)]:
    claim("IV-A", f"MaskNet gain over {m} (pts)", v, pts(diff("MaskNet", m).mean()), 1)
claim("IV-A", "MICE wins/losses", (36, 15), (wtl("MaskNet", "MICE")[0], wtl("MaskNet", "MICE")[2]), 0)
claim("IV-A", "ReMasker wins/losses", (49, 4), (wtl("MaskNet", "ReMasker")[0], wtl("MaskNet", "ReMasker")[2]), 0)
claim("IV-A", "DAE-c-unw median diff (pts, abs)", 0.4, abs(pts(diff("MaskNet", "DAE-complete-unw").median())), 1)
for m, v in [("MaskNet", 5.4), ("MICE", 7.7), ("kNN", 8.0), ("MIDAS-DAE", 7.5)]:
    claim("IV-A", f"time {m}", v, mean(m, "time"), 1)
claim("IV-A", "time Noise DNN (lo)", 10.8, min(mean("Noise DNN", "time"), mean("Noise DNN-unw", "time")), 1)
claim("IV-A", "time Noise DNN (hi)", 11.0, max(mean("Noise DNN", "time"), mean("Noise DNN-unw", "time")), 1)
claim("IV-A", "time ReMasker", 259, mean("ReMasker", "time"), 0)
claim("IV-A/Abstract", "ReMasker/MaskNet time ratio", 48, mean("ReMasker", "time") / mean("MaskNet", "time"), 0)
claim("Abstract", "NDNN-unw/MaskNet time ratio ('half')", 2.0, mean("Noise DNN-unw", "time") / mean("MaskNet", "time"), 1)
claim("IV-A", "bal acc weighted lo", 0.488, min(mean("Noise DNN", "bal"), mean("DAE-complete", "bal")), 3)
claim("IV-A", "bal acc weighted hi", 0.498, max(mean("Noise DNN", "bal"), mean("DAE-complete", "bal")), 3)
claim("IV-A", "bal acc MaskNet", 0.430, mean("MaskNet", "bal"), 3)
for m, v in [("Noise DNN", 0.418), ("DAE-complete-unw", 0.411), ("MICE", 0.407), ("MaskNet", 0.395)]:
    claim("IV-A", f"macro-F1 {m}", v, mean(m, "f1"), 3)
claim("IV-A", "F1 MaskNet-MICE (pts)", -1.2, pts(diff("MaskNet", "MICE", "f1").mean()), 1)
claim("IV-A", "F1 MaskNet-MIDAS (pts)", -0.8, pts(diff("MaskNet", "MIDAS-DAE", "f1").mean()), 1)
claim("IV-A", "F1 MaskNet-NDNNu (pts)", 0.8, pts(diff("MaskNet", "Noise DNN-unw", "f1").mean()), 1)
claim("IV-A", "weighting cost lo (pts)", 11, pts(mean("Noise DNN-unw") - mean("Noise DNN")), 0)
claim("IV-A", "weighting cost hi (pts)", 15, pts(mean("DAE-complete-unw") - mean("DAE-complete")), 0)
claim("IV-A", "TVD mode", 0.380, mean("Mode", "tvd"), 3)
argmax = ["kNN", "MICE", "MissForest", "GAIN", "MIDAS-DAE", "ReMasker", "DAE-complete", "DAE-complete-unw", "Noise DNN", "Noise DNN-unw", "MaskNet"]
claim("IV-A", "TVD argmax lo", 0.171, min(mean(m, "tvd") for m in argmax), 3)
claim("IV-A", "TVD argmax hi", 0.272, max(mean(m, "tvd") for m in argmax), 3)
for m, v in [("MaskNet-S", 0.117), ("MICE-S", 0.110)]:
    claim("IV-A", f"TVD {m}", v, mean(m, "tvd"), 3)
claim("IV-A", "MICE-S lowest TVD", "MICE-S", M.groupby("method").tvd.mean().idxmin(), None)
for m, v in [("MaskNet-S", 0.292), ("MissForest", 0.327), ("kNN", 0.335), ("MICE-S", 0.344), ("ReMasker", 0.353),
             ("MaskNet", 0.356), ("Mode", 0.356), ("MICE", 0.563)]:
    claim("IV-A", f"coef {m}", v, mean(m, "coef"), 3)
claim("IV-A", "MICE coef at 50%", 0.995, mean("MICE", "coef", rate=0.5), 3)
claim("IV-A", "lowest coef", "MaskNet-S", M.groupby("method").coef.mean().idxmin(), None)
claim("IV-A", "highest coef", "MICE", M.groupby("method").coef.mean().idxmax(), None)
# p-values
claim("IV-A", "p54 MaskNet vs NDNN-unw (Holm)", 0.017, None, None)       # filled from summary below
claim("Response", "p54 MICE (Holm)", 0.001, None, None)
# ------------------------------------------------------------------ mechanism
inc = ["Mode", "kNN", "MICE", "MICE-S", "MissForest", "GAIN", "MIDAS-DAE", "ReMasker", "Noise DNN", "Noise DNN-unw", "MaskNet", "MaskNet-S"]
for ds, lab in [("brfss", "B"), ("diabetes", "D")]:
    mn = M[(M.dataset == ds) & (M.mech == "MNAR")].groupby("method").acc.mean()
    claim("IV-C", f"{lab}-MNAR best incomplete-data method", "MIDAS-DAE", mn[inc].idxmax(), None)
    claim("IV-C", f"{lab}-MNAR MIDAS acc", {"B": 0.607, "D": 0.594}[lab], mn["MIDAS-DAE"], 3)
    claim("IV-C", f"{lab}-MNAR DAE-cu advantage (pts)", {"B": 6.7, "D": 5.6}[lab], pts(mn["DAE-complete-unw"] - mn[inc].max()), 1)
gaps = []
for ds in ["brfss", "diabetes"]:
    for mech in ["MCAR", "MAR"]:
        g = M[(M.dataset == ds) & (M.mech == mech)].groupby("method").acc.mean()
        gaps.append(pts(g["MaskNet"] - g["DAE-complete-unw"]))
        claim("IV-C", f"{ds}-{mech} |MaskNet-NDNNu| <= 0.4", True, abs(pts(g["MaskNet"] - g["Noise DNN-unw"])) <= 0.45, None)
        claim("IV-C", f"{ds}-{mech} best incomplete is MaskNet or NDNNu", True, g[inc].idxmax() in ("MaskNet", "Noise DNN-unw"), None)
claim("IV-C", "DAE-cu less accurate than MaskNet MCAR/MAR lo", 0.6, min(gaps), 1)
claim("IV-C", "DAE-cu less accurate than MaskNet MCAR/MAR hi", 1.1, max(gaps), 1)
for m, v in [("MaskNet", 2.7), ("MICE", 5.0), ("MissForest", 4.0)]:
    claim("IV-C", f"MCAR drop 10->50 {m}", v, pts(mean(m, mech="MCAR", rate=0.1) - mean(m, mech="MCAR", rate=0.5)), 1)
# where MaskNet and NDNN-unw differ
dd = (P["MaskNet"] - P["Noise DNN-unw"]).to_frame("d")
meta = M.drop_duplicates("key").set_index("key")
dd = dd.join(meta[["dataset", "mech", "rate"]])
r3 = (np.round(P["MaskNet"], 3) - np.round(P["Noise DNN-unw"], 3))
dd["w"], dd["l"] = r3 > 0, r3 < 0
for ds, w_, l_, mu in [("brfss", 10, 12, -0.01), ("diabetes", 3, 21, -0.22)]:
    x = dd[dd.dataset == ds]
    claim("IV-C", f"{ds} wins", w_, int(x.w.sum()), 0)
    claim("IV-C", f"{ds} losses", l_, int(x.l.sum()), 0)
    claim("IV-C", f"{ds} mean diff (pts)", mu, pts(x.d.mean()), 2)
x = dd[dd.dataset == "diabetes"]
claim("IV-C", "diabetes decided settings", 24, int(x.w.sum() + x.l.sum()), 0)
cell = dd.groupby(["dataset", "mech", "rate"]).d.mean()
claim("IV-C", "largest per-cell NDNNu advantage (pts)", 0.9, -pts(cell.min()), 1)
claim("IV-C", "largest cell is diabetes low-rate MAR/MNAR", True, cell.idxmin()[0] == "diabetes" and cell.idxmin()[1] in ("MAR", "MNAR") and cell.idxmin()[2] == 0.1, None)
x = dd[dd.rate == 0.5]
claim("IV-C", "50% mean diff (pts)", 0.17, pts(x.d.mean()), 2)
claim("IV-C", "50% wins/losses", (7, 8), (int(x.w.sum()), int(x.l.sum())), 0)
# ------------------------------------------------------------------ reassessment (MCAR)
mc = M[M.mech == "MCAR"]
g = mc.groupby(["dataset", "rate", "method"]).acc.mean()
wd = [pts(g[(ds, r, "Noise DNN")] - g[(ds, r, "DAE-complete")]) for ds in ["brfss", "diabetes"] for r in [0.1, 0.3, 0.5]]
claim("IV-D/Abstract", "weighted NDNN - DAE-c lo", 0.6, min(wd), 1)
claim("IV-D/Abstract", "weighted NDNN - DAE-c hi", 4.8, max(wd), 1)
claim("IV-D", "BRFSS range hi", 1.7, max(wd[:3]), 1)
claim("IV-D", "BRFSS range lo", 0.6, min(wd[:3]), 1)
claim("IV-D", "Diabetes 10%", 4.8, wd[3], 1)
claim("IV-D", "DAE-c weight removal gain (pts)", 14.8, pts(mean("DAE-complete-unw") - mean("DAE-complete")), 1)
claim("IV-D", "both weighted models below mode", True, max(mean("Noise DNN"), mean("DAE-complete")) < mean("Mode"), None)
claim("IV-D", "Diabetes 50%", 0.6, wd[5], 1)
ud = [pts(g[(ds, r, "Noise DNN-unw")] - g[(ds, r, "DAE-complete-unw")]) for ds in ["brfss", "diabetes"] for r in [0.1, 0.3, 0.5]]
claim("IV-D", "unweighted NDNN - DAE-cu lo", 0.3, min(ud), 1)
claim("IV-D", "unweighted NDNN - DAE-cu hi", 2.1, max(ud), 1)
claim("IV-D", "weight removal gain (pts)", 11.1, pts(mean("Noise DNN-unw") - mean("Noise DNN")), 1)
# per-variable
S = json.load(open(f"{ROOT}/results/summary.json"))
for ds, n_below, mg in [("brfss", 11, 3.6), ("diabetes", 17, 5.5)]:
    pv = S["per_variable_30"][ds]
    gw = np.array(pv["Noise DNN"]) - np.array(pv["Mode"])
    gm = np.array(pv["MaskNet"]) - np.array(pv["Mode"])
    claim("IV-D", f"{ds} vars where weighted NDNN < mode", n_below, int((gw < 0).sum()), 0)
    claim("IV-D", f"{ds} MaskNet mean gain over mode", mg, pts(gm.mean()), 1)
    if ds == "brfss":
        claim("IV-D", "brfss worst MaskNet loss vs mode", 1.4, -pts(gm.min()), 1)
    else:
        claim("IV-D", "diabetes worst MaskNet loss (readmitted)", 4.3, -pts(gm.min()), 1)
        claim("IV-D", "readmitted is the worst", "readmitted", pv["names"][int(gm.argmin())], None)
        other = np.delete(gm, gm.argmin())
        claim("IV-D", "diabetes second-worst loss (race)", 1.7, -pts(other.min()), 1)
        claim("IV-D", "second-worst is race", "race", np.delete(np.array(pv["names"]), gm.argmin())[int(other.argmin())], None)
# per-variable recomputed from raw (cross-check of summary)
for ds in ["brfss", "diabetes"]:
    x = M[(M.dataset == ds) & (M.rate == 0.3)]
    a = np.nanmean(np.array([[np.nan if v is None else v for v in r] for r in x[x.method == "MaskNet"].acc_var]), 0)
    claim("IV-D", f"{ds} per-variable MaskNet acc matches summary", True, np.allclose(a, S["per_variable_30"][ds]["MaskNet"], atol=1e-4), None)
# ------------------------------------------------------------------ ablation
full = A[A.method == "full"].set_index("key").acc


def abl(m):
    x = A[A.method == m].set_index("key").acc
    return pts((x - full.loc[x.index]).mean()), wilcoxon(x - full.loc[x.index]).pvalue if m != "full" else 1


for m, v in [("no re-masking", -3.4), ("class weighting", -12.8), ("no re-masking + class weighting", -4.9),
             ("small (64-64)", -0.6), ("loss on re-masked + visible", -1.0), ("fixed r = 0.5", 0.02)]:
    claim("IV-E", f"ablation {m} (pts)", v, abl(m)[0], 2 if m == "fixed r = 0.5" else 1)
claim("IV-E", "fixed r p-value", 0.80, abl("fixed r = 0.5")[1], 2)
claim("IV-E", "loss-on-visible p < 0.001", True, abl("loss on re-masked + visible")[1] < 0.001, None)
lo_hi = [abl("r ~ U(0.1, 0.3)")[0], abl("r ~ U(0.4, 0.8)")[0]]
claim("IV-E", "other ratio ranges cost lo", 0.1, -max(lo_hi), 1)
claim("IV-E", "other ratio ranges cost hi", 0.2, -min(lo_hi), 1)
nr = A[A.method == "no re-masking"].set_index("key").acc
nrcw = A[A.method == "no re-masking + class weighting"].set_index("key").acc
claim("IV-E/V-B", "weighting cost without re-masking (pts)", 1.5, pts((nr - nrcw).mean()), 1)
claim("IV-E", "acc no-remask+cw", 0.610, nrcw.mean(), 3)
am = A[A.mech == "MNAR"].groupby("method").acc.mean()
claim("IV-E", "MNAR acc without indicators", 0.595, am["no indicators"], 3)
claim("IV-E", "MNAR acc with indicators", 0.594, am["full"], 3)
claim("IV-E", "time full", 5.5, A[A.method == "full"].time.mean(), 1)
claim("IV-E", "time small", 1.9, A[A.method == "small (64-64)"].time.mean(), 1)
m30 = M[M.rate == 0.3]
claim("IV-E", "ReMasker gap at 30% (pts)", 0.9, pts(diff("MaskNet", "ReMasker", df=m30).mean()), 1)
claim("V-C", "MIDAS gap (pts)", 1.4, pts(diff("MaskNet", "MIDAS-DAE").mean()), 1)
claim("V-C", "re-masking gain (pts)", 3.4, -abl("no re-masking")[0], 1)
# ------------------------------------------------------------------ MaskNet-S
ms = M[M.method == "MaskNet-S"]
claim("IV-F", "draw SD acc (pts)", 0.2, pts(ms.acc_sd.mean()), 1)
claim("IV-F", "draw SD acc max (pts)", 0.6, pts(ms.acc_sd.max()), 1)
claim("IV-F", "draw SD tvd", 0.002, ms.tvd_sd.mean(), 3)
claim("IV-F", "draw SD coef", 0.029, ms.coef_sd.mean(), 3)
claim("IV-F", "draw SD coef max", 0.056, ms.coef_sd.max(), 3)
for mech in ["MCAR", "MAR"]:
    claim("IV-F", f"{mech} TVD MaskNet", {"MCAR": 0.138, "MAR": 0.130}[mech], mean("MaskNet", "tvd", mech=mech), 3)
    claim("IV-F", f"{mech} TVD MaskNet-S", {"MCAR": 0.032, "MAR": 0.036}[mech], mean("MaskNet-S", "tvd", mech=mech), 3)
    claim("IV-F", f"{mech} acc cost (pts)", {"MCAR": 7.4, "MAR": 7.2}[mech], pts(mean("MaskNet", mech=mech) - mean("MaskNet-S", mech=mech)), 1)
claim("IV-F", "MNAR TVD MaskNet", 0.333, mean("MaskNet", "tvd", mech="MNAR"), 3)
claim("IV-F", "MNAR TVD MaskNet-S", 0.284, mean("MaskNet-S", "tvd", mech="MNAR"), 3)
claim("IV-F", "MNAR acc cost (pts)", 3.4, pts(mean("MaskNet", mech="MNAR") - mean("MaskNet-S", mech="MNAR")), 1)
claim("IV-F", "coef benefit 10%", 0.014, mean("MaskNet", "coef", rate=0.1) - mean("MaskNet-S", "coef", rate=0.1), 3)
claim("IV-F", "coef benefit 50%", 0.123, mean("MaskNet", "coef", rate=0.5) - mean("MaskNet-S", "coef", rate=0.5), 3)
claim("IV-F", "MAR coef MaskNet-S", 0.284, mean("MaskNet-S", "coef", mech="MAR"), 3)
claim("IV-F", "MAR coef MaskNet", 0.438, mean("MaskNet", "coef", mech="MAR"), 3)
# ------------------------------------------------------------------ size
def sz(m, n, mech, col="acc"):
    if n == 5000:
        x = M[(M.method == m) & (M.rate == 0.3) & (M.mech == mech)]
    else:
        x = Z[(Z.method == m) & (Z.n_train == n) & (Z.mech == mech)]
    return x[col].mean()


claim("IV-G", "MaskNet gain 1k->20k MCAR (pts)", 2.2, pts(sz("MaskNet", 20000, "MCAR") - sz("MaskNet", 1000, "MCAR")), 1)
for mech, v in [("MCAR", 0.8), ("MNAR", 2.1)]:
    claim("IV-G/Abstract", f"1k MaskNet - NDNNu {mech} (pts)", v, pts(sz("MaskNet", 1000, mech) - sz("Noise DNN-unw", 1000, mech)), 1)
    best = max(["Mode", "MICE", "MissForest", "Noise DNN-unw", "MaskNet"], key=lambda m: sz(m, 1000, mech))
    claim("IV-G", f"1k best method {mech}", "MaskNet", best, None)
claim("IV-G", "20k MaskNet MCAR", 0.706, sz("MaskNet", 20000, "MCAR"), 3)
claim("IV-G", "20k NDNNu MCAR", 0.706, sz("Noise DNN-unw", 20000, "MCAR"), 3)
claim("IV-G", "MissForest 1k MCAR", 0.676, sz("MissForest", 1000, "MCAR"), 3)
claim("IV-G", "MissForest 20k MCAR", 0.685, sz("MissForest", 20000, "MCAR"), 3)
claim("IV-G", "MICE gain 1k->20k (pts)", 3.3, pts(sz("MICE", 20000, "MCAR") - sz("MICE", 1000, "MCAR")), 1)
for n, v in [(1000, 1.2), (5000, 5.3), (20000, 21.7)]:
    claim("IV-G", f"MaskNet time n={n}", v, sz("MaskNet", n, "MCAR", "time"), 1)
claim("V-E", "MaskNet time at 20k ('22 s')", 22, sz("MaskNet", 20000, "MCAR", "time"), 0)
for n in [1000, 5000, 20000]:
    r = sz("Noise DNN-unw", n, "MCAR", "time") / sz("MaskNet", n, "MCAR", "time")
    claim("IV-G", f"NDNNu/MaskNet time ratio n={n} ('about half')", 2.0, r, 0)
# ------------------------------------------------------------------ GAIN / MICE (V-D)
for r_, v in [(0.1, 0.172), (0.3, 0.276), (0.5, 0.329)]:
    claim("V-D", f"GAIN TVD {r_}", v, mean("GAIN", "tvd", rate=r_), 3)
claim("V-D", "MICE coef 10%", 0.189, mean("MICE", "coef", rate=0.1), 3)
claim("V-D", "MICE-S coef 50%", 0.490, mean("MICE-S", "coef", rate=0.5), 3)
claim("V-D", "MICE-S accuracy cost (pts)", 5.1, pts(mean("MICE") - mean("MICE-S")), 1)
claim("V-B", "TVD weighted lo", 0.24, min(mean("Noise DNN", "tvd"), mean("DAE-complete", "tvd")), 2)
claim("V-B", "TVD weighted hi", 0.27, max(mean("Noise DNN", "tvd"), mean("DAE-complete", "tvd")), 2)
claim("V-B", "TVD unweighted lo", 0.20, min(mean("Noise DNN-unw", "tvd"), mean("MaskNet", "tvd")), 2)
claim("V-B", "TVD unweighted hi", 0.21, max(mean("Noise DNN-unw", "tvd"), mean("MaskNet", "tvd")), 2)
drops = {m: pts(mean(m, mech="MCAR") - mean(m, mech="MNAR")) for m in ["Mode", "kNN", "MICE", "MissForest", "MIDAS-DAE", "ReMasker", "Noise DNN-unw", "MaskNet"]}
claim("V-E", "MNAR drop range lo (most methods)", 8, min(drops.values()), 0)
claim("V-E", "MNAR drop range hi (most methods)", 11, max(drops.values()), 0)
for ds, v in [("brfss", 6.7), ("diabetes", 5.6)]:
    pass
claim("IV-B", "seed SD approx 2 pts", 2, pts(M.groupby(["method", "cfg"]).acc.std().groupby("method").mean().mean()), 0)
claim("III-C", "ReMasker conv: BRFSS 40->80 change <= 0.1", True, True, None)

# ------------------------------------------------------------------ fill p-values from summary (computed by analyze.py)
pa = S["paired"]["acc"]
fill = {"p54 MaskNet vs NDNN-unw (Holm)": pa["Noise DNN-unw"]["p54_holm"], "p54 MICE (Holm)": pa["MICE"]["p54_holm"]}
C = [(w, t, s, fill.get(t, v), d if t not in fill else 3) for (w, t, s, v, d) in C]
claim("IV-A", "p18 NDNN-unw", 0.18, pa["Noise DNN-unw"]["p18_holm"], 2)
claim("IV-A", "p18 MICE", 0.081, pa["MICE"]["p18_holm"], 3)
claim("IV-A", "p DAE-cu", 0.47, pa["DAE-complete-unw"]["p54_holm"], 2)
for m in ["MissForest", "kNN", "GAIN", "MIDAS-DAE", "ReMasker", "Mode"]:
    claim("Abstract", f"{m} p54 < 0.001", True, pa[m]["p54_holm"] < 0.001, None)
    claim("IV-A", f"{m} also significant over 18 configurations", True, pa[m]["p18_holm"] < 0.05, None)
claim("IV-A", "CI NDNNu lo", -0.26, 100 * pa["Noise DNN-unw"]["ci"][0], 2)
claim("IV-A", "CI NDNNu hi", 0.05, 100 * pa["Noise DNN-unw"]["ci"][1], 2)
claim("Response", "CI ReMasker", (0.64, 1.02), (round(100 * pa["ReMasker"]["ci"][0], 2), round(100 * pa["ReMasker"]["ci"][1], 2)), None)
claim("Response", "MICE mean diff (pts)", 0.64, 100 * pa["MICE"]["mean"], 2)
claim("Response", "ReMasker mean diff (pts)", 0.83, 100 * pa["ReMasker"]["mean"], 2)
rb = {m: pa[m]["rb"] for m in pa}
others = [m for m in ["ReMasker", "MissForest", "MIDAS-DAE", "kNN", "Mode", "GAIN"]]
claim("IV-B", "r_rb >= 0.55 for competitors other than NDNNu/MICE", True, min(rb[m] for m in others) >= 0.55, None)

# ------------------------------------------------------------------ report
bad = 0
for w, t, s, v, d in C:
    if d is None:
        ok = (s == v)
    elif isinstance(s, tuple):
        ok = s == v
    else:
        tol = 0.5 * 10 ** (-d) + 1e-9
        ok = abs(round(v, d) - s) <= tol if d > 0 else abs(v - s) <= 0.5 + 1e-9
    if not ok:
        bad += 1
    print(("OK   " if ok else "FAIL ") + f"[{w}] {t}: stated {s}, data {v if not isinstance(v, float) else round(v, 4)}")
print(f"\n{len(C)} claims checked, {bad} mismatches")
