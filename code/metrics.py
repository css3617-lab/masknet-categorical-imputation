"""Evaluation metrics on the artificially masked test entries."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def pointwise(Xtrue, Ximp, M, K):
    acc = (Xtrue[M] == Ximp[M]).mean()
    f1s, bals, tvds, accv = [], [], [], []
    for j, k in enumerate(K):
        m = M[:, j]
        if m.sum() == 0:
            continue
        t, p = Xtrue[m, j], Ximp[m, j]
        accv.append(float((t == p).mean()))
        present = np.unique(t)
        f1, rec = [], []
        for c in present:
            tp = np.sum((t == c) & (p == c))
            fp = np.sum((t != c) & (p == c))
            fn = np.sum((t == c) & (p != c))
            f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
            rec.append(tp / (tp + fn))
        f1s.append(np.mean(f1)); bals.append(np.mean(rec))
        ht = np.bincount(t, minlength=k) / len(t)
        hp = np.bincount(p, minlength=k) / len(p)
        tvds.append(0.5 * np.abs(ht - hp).sum())
    if len(accv) < len(K):                     # keep per-variable accuracy aligned with variables
        full, it = [], iter(accv)
        for j in range(len(K)):
            full.append(next(it) if M[:, j].any() else None)
        accv = full
    return dict(acc=float(acc), f1=float(np.mean(f1s)), bal=float(np.mean(bals)), tvd=float(np.mean(tvds)), acc_var=accv)


def design(X, K, outcome, positive):
    """Dummy coding (first category dropped) of all non-outcome variables; binary outcome."""
    cols = []
    for j, k in enumerate(K):
        if j == outcome:
            continue
        cols.append((X[:, j][:, None] == np.arange(1, k)[None]).astype(float))
    return np.concatenate(cols, 1), (X[:, outcome] == positive).astype(int)


def coefs(X, K, outcome, positive):
    F, y = design(X, K, outcome, positive)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(F, y)
    return lr.coef_.ravel()


def coef_rmse(beta_true, Ximp, K, outcome, positive):
    b = coefs(Ximp, K, outcome, positive)
    return float(np.sqrt(np.mean((b - beta_true) ** 2)))
