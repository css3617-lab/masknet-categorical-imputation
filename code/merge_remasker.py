"""Evaluate ReMasker imputations computed on the second machine (remasker_vm) or later on the primary
machine (remasker_cloud) and append them to results/main/<key>.json.

Times from the second machine are rescaled by MaskNet's time on the two machines in the same setting,
so that every reported time refers to the primary machine.
"""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmark as R
import imputers as I

ROOT = R.ROOT
src = sys.argv[1:] or [f"{ROOT}/results/remasker_vm", f"{ROOT}/results/remasker_cloud"]
n = 0
for d in src:
    for f in sorted(glob.glob(f"{d}/*.npz")):
        key = os.path.basename(f)[:-4]
        ds, mech, p, seed, ntr = key.split("_")
        fn = f"{ROOT}/results/main/{key}.json"
        rows = json.load(open(fn))
        if any(r["method"] == "ReMasker" for r in rows):
            continue
        z = np.load(f)
        Xtr, Mtr, Xte, Mte, K, outcome = R.make_setting(ds, mech, float(p), int(seed))
        Ximp = z["Ximp"].astype(np.int64)
        assert np.array_equal(Ximp[~Mte], Xte[~Mte]), key      # same setting on both machines
        t = float(z["t_remasker"])
        if "t_masknet" in z:
            t_mn_here = next(r["time"] for r in rows if r["method"] == "MaskNet")
            t = t * t_mn_here / float(z["t_masknet"])
        beta = R.coefs(Xte, K, outcome, R.POSITIVE[ds])
        base = {k: rows[0][k] for k in ["dataset", "mech", "rate", "seed", "n_train", "miss_tr", "miss_te"]}
        rows.append({**base, **R.evaluate("ReMasker", Ximp, Xte, Mte, K, outcome, beta, R.POSITIVE[ds], t)})
        json.dump(rows, open(fn, "w"))
        n += 1
print("merged", n)
