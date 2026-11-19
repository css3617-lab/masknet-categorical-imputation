"""Inductive benchmark runner.

Usage:
  python run_benchmark.py main  <worker_id> <n_workers>    # 54 settings x all methods
  python run_benchmark.py ablation <worker_id> <n_workers> # 30% missingness, MaskNet variants
  python run_benchmark.py size  <worker_id> <n_workers>    # training-size sensitivity
Each setting writes one JSON file into results/<mode>/, so runs can be resumed.
"""
import os
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
import sys, json, time, itertools, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imputers as I
from masks import make_mechanism
from metrics import pointwise, coefs, coef_rmse
from nn import Blocks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = {"brfss": 1, "diabetes": 2}
POSITIVE = {"brfss": 1, "diabetes": 2}          # Diabetes_binary = 1 ; readmitted = '<30'
MECHS = {"MCAR": 1, "MAR": 2, "MNAR": 3}
RATES = {0.1: 1, 0.3: 2, 0.5: 3}
SEEDS = [0, 1, 2]
N_DRAWS = 10                                    # stochastic draws for MaskNet-S


def load(ds):
    z = np.load(f"{ROOT}/data/{ds}.npz")
    return z["X"].astype(np.int64), z["K"], int(z["outcome"])


def mrng(*key):
    return np.random.default_rng([int(k) for k in key])


def make_setting(ds, mech, p, seed, n_train=5000, n_test=5000):
    X, K, outcome = load(ds)
    r = mrng(seed, DATASETS[ds])
    idx = r.choice(len(X), n_train + n_test, replace=False)
    S = X[idx]
    Xtr, Xte = S[:n_train], S[n_train:]
    rm = mrng(seed, DATASETS[ds], MECHS[mech], RATES[p], n_train)
    f = make_mechanism(S, K, mech, p, rm)
    Mtr, Mte = f(Xtr, rm), f(Xte, rm)
    return Xtr, Mtr, Xte, Mte, K, outcome


def evaluate(name, Ximp, Xte, Mte, K, outcome, beta, positive, t, extra=None):
    m = pointwise(Xte, Ximp, Mte, K)
    m["coef"] = coef_rmse(beta, Ximp, K, outcome, positive)
    m["time"] = t
    m["method"] = name
    if extra:
        m.update(extra)
    return m


def run_methods(ds, mech, p, seed, methods, n_train=5000):
    Xtr, Mtr, Xte, Mte, K, outcome = make_setting(ds, mech, p, seed, n_train)
    pos = POSITIVE[ds]
    beta = coefs(Xte, K, outcome, pos)
    b = Blocks(K)
    rows = []
    base = dict(dataset=ds, mech=mech, rate=p, seed=seed, n_train=n_train,
                miss_tr=float(Mtr.mean()), miss_te=float(Mte.mean()))
    for mi, (name, fn) in enumerate(methods):
        rng = mrng(seed, DATASETS[ds], MECHS[mech], RATES[p], zlib.crc32(name.encode()))
        t0 = time.perf_counter()
        out = fn(Xtr, Mtr, Xte, Mte, K, rng)
        t = time.perf_counter() - t0
        if isinstance(out, dict):                  # MaskNet: argmax + stochastic draws
            P = out["P"]
            rows.append({**base, **evaluate(name, I.fill(Xte, Mte, b.argmax(P)), Xte, Mte, K, outcome, beta, pos, t)})
            if out.get("sample"):
                draws = []
                for k in range(N_DRAWS):
                    Xs = I.fill(Xte, Mte, b.sample(P, rng))
                    draws.append(evaluate(name + "-S", Xs, Xte, Mte, K, outcome, beta, pos, t))
                agg = {k: float(np.mean([d_[k] for d_ in draws])) for k in ["acc", "f1", "bal", "tvd", "coef"]}
                sd = {k + "_sd": float(np.std([d_[k] for d_ in draws], ddof=1)) for k in ["acc", "f1", "bal", "tvd", "coef"]}
                rows.append({**base, **agg, **sd, "time": t, "method": name + "-S"})
        else:
            Ximp = I.fill(Xte, Mte, b.argmax(out)) if out.dtype.kind == "f" else out
            rows.append({**base, **evaluate(name, Ximp, Xte, Mte, K, outcome, beta, pos, t)})
        print(f"  {ds} {mech} {p} s{seed} {name:22s} acc={rows[-1]['acc']:.3f} t={t:.1f}s", flush=True)
    return rows


# ----------------------------------------------------------------------------- method tables
def dae(weighted, complete):
    def f(Xtr, Mtr, Xte, Mte, K, rng):
        M = np.zeros_like(Mtr) if complete else Mtr
        bb, net = I.train_dae(Xtr, M, K, rng, weighted)
        return I.dae_impute(bb, net, Xte, Mte)
    return f


def masknet(sample=False, **kw):
    def f(Xtr, Mtr, Xte, Mte, K, rng):
        model = I.train_masknet(Xtr, Mtr, K, rng, **kw)
        return {"P": I.masknet_proba(model, Xte, Mte), "sample": sample}
    return f


REMASKER_EPOCHS = int(os.environ.get("REMASKER_EPOCHS", 100))


def remasker(epochs=None):
    def f(Xtr, Mtr, Xte, Mte, K, rng):
        m = I.ReMasker(K, rng).fit(Xtr, Mtr, rng, epochs=epochs or REMASKER_EPOCHS)
        return m.proba(Xte, Mte)
    return f


MAIN = [
    ("Mode", I.impute_mode),
    ("kNN", I.impute_knn),
    ("MICE", I.impute_mice),
    ("MICE-S", lambda *a: I.impute_mice(*a, stochastic=True)),
    ("MissForest", I.impute_missforest),
    ("GAIN", I.impute_gain),
    ("MIDAS-DAE", dae(False, False)),
    ("DAE-complete", dae(True, True)),
    ("DAE-complete-unw", dae(False, True)),
    ("Noise DNN", lambda *a: I.impute_noise_dnn(*a, weighted=True)),
    ("Noise DNN-unw", lambda *a: I.impute_noise_dnn(*a, weighted=False)),
    ("MaskNet", masknet(sample=True)),
    ("ReMasker", remasker()),
]

ABLATION = [
    ("full", masknet()),
    ("no indicators", masknet(indicators=False)),
    ("no re-masking", masknet(remask=False)),
    ("class weighting", masknet(weighted=True)),
    ("no re-masking + class weighting", masknet(remask=False, weighted=True)),
    ("small (64-64)", masknet(hidden=64)),
    ("fixed r = 0.5", masknet(r_range=(0.5, 0.5))),
    ("r ~ U(0.1, 0.3)", masknet(r_range=(0.1, 0.3))),
    ("r ~ U(0.4, 0.8)", masknet(r_range=(0.4, 0.8))),
    ("loss on re-masked + visible", masknet(loss_on="observed")),
]

SIZE = [
    ("Mode", I.impute_mode),
    ("MICE", I.impute_mice),
    ("MissForest", I.impute_missforest),
    ("Noise DNN-unw", lambda *a: I.impute_noise_dnn(*a, weighted=False)),
    ("MaskNet", masknet()),
]


def settings(mode):
    if mode == "main":
        return list(itertools.product(DATASETS, MECHS, RATES, SEEDS, [5000]))
    if mode == "ablation":
        return list(itertools.product(DATASETS, MECHS, [0.3], SEEDS, [5000]))
    if mode == "size":
        return list(itertools.product(DATASETS, ["MCAR", "MNAR"], [0.3], SEEDS, [1000, 20000]))
    raise ValueError(mode)


if __name__ == "__main__":
    mode, wid, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    only = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    methods = {"main": MAIN, "ablation": ABLATION, "size": SIZE}[mode]
    if only:
        methods = [m for m in methods if m[0] in only]
    if os.environ.get("EXCLUDE"):
        methods = [m for m in methods if m[0] not in os.environ["EXCLUDE"].split(",")]
    outdir = f"{ROOT}/results/{mode}"
    os.makedirs(outdir, exist_ok=True)
    todo = settings(mode)
    # heavy settings (Diabetes, 50 %, n=20000) first, interleaved across workers
    for i, (ds, mech, p, seed, n) in enumerate(todo):
        if i % nw != wid:
            continue
        fn = f"{outdir}/{ds}_{mech}_{p}_{seed}_{n}.json"
        if os.path.exists(fn):
            continue
        rows = run_methods(ds, mech, p, seed, methods, n)
        with open(fn, "w") as fh:
            json.dump(rows, fh)
