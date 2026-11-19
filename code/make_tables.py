"""Write LaTeX tables for the revised manuscript from results/summary.json."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/tables"
os.makedirs(OUT, exist_ok=True)
S = json.load(open(f"{ROOT}/results/summary.json"))
MK = {"‡": r"$^\ddagger$", "†": r"$^\dagger$", "*": r"$^{*}$", "": ""}
NEW = {"MICE-S", "ReMasker", "DAE-complete-unw"}
DISP = {"DAE-complete": r"DAE-complete$^\S$", "DAE-complete-unw": r"DAE-complete-unw$^\S$"}


def f3(x):
    return f"{x:.3f}"


def w(name, s):
    open(f"{OUT}/{name}.tex", "w").write(s)


# ---------------------------------------------------------------- Table: main results
rows = []
T2 = S["table2"]
best = {k: (max if k in ("acc", "f1", "bal") else min)(r[k] for r in T2) for k in ["acc", "f1", "bal", "tvd", "coef"]}
for r in T2:
    name = DISP.get(r["method"], r["method"])
    if r["method"] in NEW:
        name = r"\textcolor{blue}{" + name + "}"
    cells = []
    for k in ["acc", "f1", "bal", "tvd", "coef"]:
        v = f3(r[k])
        if abs(r[k] - best[k]) < 5e-4:
            v = r"\textbf{" + v + "}"
        cells.append(v + MK[r[k + "_mark"]])
    rows.append(f"{name} & " + " & ".join(cells) + f" & {r['time']:.1f} & {r['rank']:.1f}\\\\")
w("tab_main", r"""\begin{table*}[!t]
\caption{Mean performance over the 54 settings (2 datasets $\times$ 3 mechanisms $\times$ 3 missing rates $\times$ 3 seeds). Acc., accuracy; F1, macro-F1; Bal., balanced accuracy (higher is better); TVD, distributional error; Coef., coefficient RMSE of the downstream regression (lower is better); Time, seconds for fitting and imputation on one CPU core; Rank, average rank by accuracy within settings (1 = best of 14). Markers: two-sided Wilcoxon signed-rank test against MaskNet over the 54 paired settings with Holm correction, $^{*}p<0.05$, $^\dagger p<0.01$, $^\ddagger p<0.001$. Bold: best in column. $^\S$Requires complete training data (reference only). \rev{Methods in blue were added in the revision; MaskNet-S values are means over ten independent draws.}}\label{tab:main}
\centering\small
\begin{tabular}{@{}lccccccc@{}}
\toprule
Method & Acc. & F1 & Bal. & TVD & Coef. & Time & Rank\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- Table: effect sizes
P = S["paired"]["acc"]
order = ["Noise DNN-unw", "DAE-complete-unw", "ReMasker", "MICE", "MissForest", "MIDAS-DAE", "kNN", "Mode", "GAIN", "MICE-S", "MaskNet-S", "Noise DNN", "DAE-complete"]
rows = []
for m in order:
    if m not in P:
        continue
    s = P[m]
    p54 = s["p54_holm"]; p18 = s["p18_holm"]
    fp = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
    rows.append(f"{DISP.get(m, m)} & {100*s['mean']:+.2f} & [{100*s['ci'][0]:+.2f}, {100*s['ci'][1]:+.2f}] & {100*s['median']:+.2f} & "
                f"{s['win']}/{s['tie']}/{s['loss']} & {s['rb']:+.2f} & {fp(p54)} & {fp(p18)}\\\\")
w("tab_effect", r"""\begin{table*}[!t]
\caption{\rev{Paired accuracy differences between MaskNet and each competitor (positive: MaskNet more accurate), in percentage points. CI, 95\% cluster-bootstrap confidence interval of the mean (configurations resampled); W/T/L, settings in which MaskNet won, tied (three decimals) or lost; $r_{\mathrm{rb}}$, matched-pairs rank-biserial correlation; $p_{54}$, Wilcoxon test over 54 settings; $p_{18}$, over 18 seed-averaged configurations; both Holm-adjusted.}}\label{tab:effect}
\centering\small
\begin{tabular}{@{}lccccccc@{}}
\toprule
Competitor & Mean & 95\% CI & Median & W/T/L & $r_{\mathrm{rb}}$ & $p_{54}$ & $p_{18}$\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- Table: dataset x mechanism
T3 = S["table3"]
cols = ["B-MCAR", "B-MAR", "B-MNAR", "D-MCAR", "D-MAR", "D-MNAR"]
methods = [r["method"] for r in T2]
bestc = {c: max(T3[c][m] for m in methods if m not in ("DAE-complete", "DAE-complete-unw")) for c in cols}
rows = []
for m in methods:
    cells = []
    for c in cols:
        v = T3[c][m]
        s = f3(v)
        if m not in ("DAE-complete", "DAE-complete-unw") and abs(v - bestc[c]) < 5e-4:
            s = r"\textbf{" + s + "}"
        cells.append(s)
    name = DISP.get(m, m)
    if m in NEW:
        name = r"\textcolor{blue}{" + name + "}"
    rows.append(f"{name} & " + " & ".join(cells) + r"\\")
w("tab_mech", r"""\begin{table*}[!t]
\caption{Mean accuracy by dataset (B, BRFSS; D, Diabetes 130-US) and missingness mechanism, averaged over missing rates and seeds. Bold: best in column among methods that use only incomplete training data. $^\S$Requires complete training data.}\label{tab:mech}
\centering\small
\begin{tabular}{@{}lcccccc@{}}
\toprule
Method & B-MCAR & B-MAR & B-MNAR & D-MCAR & D-MAR & D-MNAR\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- Table: reassessment (MCAR)
rows = []
for key in ["brfss|0.1", "brfss|0.3", "brfss|0.5", "diabetes|0.1", "diabetes|0.3", "diabetes|0.5"]:
    v = S["table4"][key]
    ds, r = key.split("|")
    rows.append(f"{'B' if ds == 'brfss' else 'D'} & {int(float(r)*100)} & {f3(v['DAEc_acc'])} & {f3(v['NDNN_acc'])} & {f3(v['DAEc_f1'])} & {f3(v['NDNN_f1'])} & "
                f"\\textcolor{{blue}}{{{f3(v['DAEcu_acc'])}}} & {f3(v['NDNNu_acc'])} & {f3(v['MaskNet_acc'])}\\\\")
w("tab_reassess", r"""\begin{table}[!t]
\caption{Re-run of the preliminary comparison under MCAR. B, BRFSS; D, Diabetes 130-US; rate in \%. DAE-c, class-weighted DAE trained on complete data; NDNN, class-weighted Noise DNN; \rev{DAE-cu, unweighted DAE trained on complete data;} NDNN-u, unweighted Noise DNN; F1, macro-F1.}\label{tab:reassess}
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llccccccc@{}}
\toprule
 & & DAE-c & NDNN & DAE-c & NDNN & \rev{DAE-cu} & NDNN-u & MaskNet\\
Data & Rate & acc. & acc. & F1 & F1 & \rev{acc.} & acc. & acc.\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Table: ablation
lab = {"full": "MaskNet (full)", "no indicators": "without indicators", "no re-masking": "without re-masking",
       "class weighting": "with class weighting", "no re-masking + class weighting": r"\rev{without re-masking, with class weighting}",
       "small (64-64)": "small (64--64)", "fixed r = 0.5": r"\rev{fixed ratio $r=0.5$}", "r ~ U(0.1, 0.3)": r"\rev{$r\sim U(0.1,0.3)$}",
       "r ~ U(0.4, 0.8)": r"\rev{$r\sim U(0.4,0.8)$}", "loss on re-masked + visible": r"\rev{loss on re-masked $+$ visible}",
       "ref: MIDAS-DAE": "MIDAS-DAE", "ref: ReMasker": r"\rev{ReMasker}", "ref: Noise DNN-unw": "Noise DNN-unw",
       "ref: DAE-complete-unw": r"\rev{DAE-complete-unw}$^\S$"}
order = ["full", "no indicators", "no re-masking", "class weighting", "no re-masking + class weighting", "small (64-64)",
         "fixed r = 0.5", "r ~ U(0.1, 0.3)", "r ~ U(0.4, 0.8)", "loss on re-masked + visible",
         "ref: MIDAS-DAE", "ref: ReMasker", "ref: Noise DNN-unw", "ref: DAE-complete-unw"]
A = {r["method"]: r for r in S["ablation"]}
rows = []
for m in order:
    if m not in A:
        continue
    r = A[m]
    if m == "ref: MIDAS-DAE":
        rows.append(r"\midrule\multicolumn{7}{@{}l}{\emph{Reference methods (main benchmark, 30\%; $\Delta$ against MaskNet of the main run)}}\\")
    sig = "" if m == "full" else (r"$^\ddagger$" if r["p"] < 0.001 else r"$^\dagger$" if r["p"] < 0.01 else r"$^{*}$" if r["p"] < 0.05 else "")
    d = "+0.000" if m == "full" else f"{r['delta']:+.3f}{sig}"
    rows.append(f"{lab[m]} & {f3(r['acc'])} & {d} & {f3(r['f1'])} & {f3(r['tvd'])} & {f3(r['mnar'])} & {r['time']:.1f}\\\\")
w("tab_ablation", r"""\begin{table}[!t]
\caption{Ablation of MaskNet at 30\% missingness (2 datasets $\times$ 3 mechanisms $\times$ 3 seeds). Acc., accuracy; $\Delta$acc., mean paired difference from the full model (Wilcoxon, $^{*}p<0.05$, $^\dagger p<0.01$, $^\ddagger p<0.001$); F1, macro-F1; MNAR, accuracy under MNAR only; Time, seconds. \rev{Rows in blue were added in the revision.} $^\S$Requires complete training data.}\label{tab:ablation}
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
Variant & Acc. & $\Delta$acc. & F1 & TVD & MNAR & Time\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Table: training size
sz = {(r["method"], r["n_train"], r["mech"]): r for r in S["size"]}
rows = []
for m in ["Mode", "MICE", "MissForest", "Noise DNN-unw", "MaskNet"]:
    cells = []
    for mech in ["MCAR", "MNAR"]:
        for n in [1000, 5000, 20000]:
            cells.append(f3(sz[(m, n, mech)]["acc"]))
    t = [f"{sz[(m, n, 'MCAR')]['time']:.1f}" for n in [1000, 5000, 20000]]
    rows.append(f"{m} & " + " & ".join(cells) + " & " + " / ".join(t) + r"\\")
w("tab_size", r"""\begin{table*}[!t]
\caption{\rev{Sensitivity to the training size at 30\% missingness (both datasets, three seeds; test size 5,000). Accuracy for 1,000, 5,000 and 20,000 training records, and fitting time in seconds (MCAR).}}\label{tab:size}
\centering\small
\begin{tabular}{@{}lccccccc@{}}
\toprule
 & \multicolumn{3}{c}{MCAR} & \multicolumn{3}{c}{MNAR} & Time (s)\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Method & 1k & 5k & 20k & 1k & 5k & 20k & 1k / 5k / 20k\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- Table: MaskNet vs MaskNet-S by condition
am, tm, cm = S["acc_by_mech"], S["tvd_by_mech"], S["coef_by_mech"]
ar, tr, cr = S["acc_by_rate"], S["tvd_by_rate"], S["coef_by_rate"]
rows = []
for lab_, a, t, c, key in [("MCAR", am, tm, cm, "MCAR"), ("MAR", am, tm, cm, "MAR"), ("MNAR", am, tm, cm, "MNAR"),
                          ("10\\%", ar, tr, cr, "0.1"), ("30\\%", ar, tr, cr, "0.3"), ("50\\%", ar, tr, cr, "0.5")]:
    if lab_ == "10\\%":
        rows.append(r"\midrule")
    rows.append(f"{lab_} & {f3(a[key]['MaskNet'])} & {f3(a[key]['MaskNet-S'])} & {f3(t[key]['MaskNet'])} & {f3(t[key]['MaskNet-S'])} & "
                f"{f3(c[key]['MaskNet'])} & {f3(c[key]['MaskNet-S'])} & {f3(c[key]['MICE'])} & {f3(c[key]['MICE-S'])}\\\\")
sd = S["masknet_s_draw_sd"]; mx = S["masknet_s_draw_sd_max"]
w("tab_masknets", r"""\begin{table*}[!t]
\caption{\rev{Accuracy--fidelity trade-off of argmax versus sampled imputation by mechanism and by missing rate (means over the other factors). M, MaskNet; M-S, MaskNet-S (mean of ten draws). Standard deviation across the ten draws, averaged over settings (maximum): accuracy """ + f"{sd['acc']:.4f} ({mx['acc']:.4f}), TVD {sd['tvd']:.4f} ({mx['tvd']:.4f}), coefficient RMSE {sd['coef']:.3f} ({mx['coef']:.3f})" + r""".}}\label{tab:masknets}
\centering\small
\begin{tabular}{@{}lcccccccc@{}}
\toprule
 & \multicolumn{2}{c}{Accuracy} & \multicolumn{2}{c}{TVD} & \multicolumn{4}{c}{Coefficient RMSE}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-9}
Condition & M & M-S & M & M-S & M & M-S & MICE & MICE-S\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
print("tables written to", OUT)
