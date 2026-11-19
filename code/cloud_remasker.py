"""ReMasker for the remaining main settings on the primary machine, processed from the END of the
setting list (the second machine works from the start). Writes results/remasker_cloud/<key>.npz.
python cloud_remasker.py <worker_id> <n_workers> <stop_file>
"""
import os
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
import sys, time, itertools, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imputers as I
from vm_remasker import make_setting, DATASETS, MECHS, RATES, mrng, ROOT

if __name__ == "__main__":
    wid, nw, claimed = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    out = f"{ROOT}/results/remasker_cloud"; os.makedirs(out, exist_ok=True)
    todo = list(itertools.product(DATASETS, MECHS, RATES, [0, 1, 2]))[::-1]
    for i, (ds, mech, p, seed) in enumerate(todo):
        key = f"{ds}_{mech}_{p}_{seed}_5000"
        if i % nw != wid or os.path.exists(f"{out}/{key}.npz"):
            continue
        done_elsewhere = set(open(claimed).read().split()) if os.path.exists(claimed) else set()
        if key in done_elsewhere:
            break                                     # met the other machine
        Xtr, Mtr, Xte, Mte, K = make_setting(ds, mech, p, seed)
        rng = mrng(seed, DATASETS[ds], MECHS[mech], RATES[p], zlib.crc32(b"ReMasker"))
        t0 = time.perf_counter()
        m = I.ReMasker(K, rng).fit(Xtr, Mtr, rng, epochs=40)
        P = m.proba(Xte, Mte)
        t = time.perf_counter() - t0
        np.savez_compressed(f"{out}/{key}.npz", Ximp=I.fill(Xte, Mte, m.b.argmax(P)).astype(np.int8), t_remasker=t)
        print(key, f"{t:.0f}s", flush=True)
