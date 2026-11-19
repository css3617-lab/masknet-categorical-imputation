"""ReMasker runs on a second Linux machine (numpy only). Writes imputed test codes per setting.
Also times MaskNet on the same machine so that ReMasker's time can be expressed relative to MaskNet.
python vm_remasker.py <worker_id> <n_workers>
"""
import os
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
import sys, time, itertools, zlib, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imputers as I
from masks import make_mechanism
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = {"brfss": 1, "diabetes": 2}; MECHS = {"MCAR": 1, "MAR": 2, "MNAR": 3}; RATES = {0.1: 1, 0.3: 2, 0.5: 3}
mrng = lambda *k: np.random.default_rng([int(x) for x in k])

def make_setting(ds, mech, p, seed, n_train=5000, n_test=5000):
    z = np.load(f"{ROOT}/data/{ds}.npz"); X, K = z["X"].astype(np.int64), z["K"]
    r = mrng(seed, DATASETS[ds]); idx = r.choice(len(X), n_train + n_test, replace=False); S = X[idx]
    Xtr, Xte = S[:n_train], S[n_train:]
    rm = mrng(seed, DATASETS[ds], MECHS[mech], RATES[p], n_train)
    f = make_mechanism(S, K, mech, p, rm)
    return Xtr, f(Xtr, rm), Xte, f(Xte, rm), K

if __name__ == "__main__":
    import pickle
    wid, nw, budget = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    skip = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else set()
    T0 = time.perf_counter()
    out = f"{ROOT}/results/remasker_vm"; os.makedirs(out, exist_ok=True)
    ck = os.path.expanduser("~/rm_ckpt"); os.makedirs(ck, exist_ok=True)
    todo = list(itertools.product(DATASETS, MECHS, RATES, [0, 1, 2]))
    left = lambda: budget - (time.perf_counter() - T0)
    for i, (ds, mech, p, seed) in enumerate(todo):
        key = f"{ds}_{mech}_{p}_{seed}_5000"
        if i % nw != wid or key in skip or os.path.exists(f"{out}/{key}.npz"):
            continue
        if left() < 20:
            break
        Xtr, Mtr, Xte, Mte, K = make_setting(ds, mech, p, seed)
        cf = f"{ck}/{key}.pkl"
        if os.path.exists(cf):
            st = pickle.load(open(cf, "rb"))
        else:
            rng = mrng(seed, DATASETS[ds], MECHS[mech], RATES[p], zlib.crc32(b"ReMasker"))
            st = {"model": I.ReMasker(K, rng), "rng": rng, "ep": 0, "t": 0.0, "ept": None}
        m, rng = st["model"], st["rng"]
        while st["ep"] < 40:
            if st["ept"] and left() < st["ept"] * 1.3 + 5:
                break
            t0 = time.perf_counter(); m.fit(Xtr, Mtr, rng, epochs=1); dt = time.perf_counter() - t0
            st["t"] += dt; st["ep"] += 1; st["ept"] = dt
        if st["ep"] < 40:
            pickle.dump(st, open(cf, "wb")); print(key, "epoch", st["ep"], flush=True)
            break
        t0 = time.perf_counter(); P = m.proba(Xte, Mte); st["t"] += time.perf_counter() - t0
        Ximp = I.fill(Xte, Mte, m.b.argmax(P))
        rng2 = mrng(seed, DATASETS[ds], MECHS[mech], RATES[p], zlib.crc32(b"MaskNet"))
        t0 = time.perf_counter(); mn = I.train_masknet(Xtr, Mtr, K, rng2); I.masknet_proba(mn, Xte, Mte)
        t_mn = time.perf_counter() - t0
        np.savez_compressed(f"{out}/{key}.npz", Ximp=Ximp.astype(np.int8), t_remasker=st["t"], t_masknet=t_mn)
        print(key, f"done remasker {st['t']:.0f}s masknet {t_mn:.1f}s", flush=True)
