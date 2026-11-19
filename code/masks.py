"""Missingness generators (Muzellec et al., 2020, adapted to categorical data).

The mechanism parameters (driver variables, weights, intercepts) are drawn once per setting
from the full 10,000-record sample and then applied independently (fresh Bernoulli draws)
to the training and test parts.
"""
import numpy as np
from nn import Blocks


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _calibrate(logit, target, lo=-50, hi=50, it=60):
    """Intercept b so that mean(sigmoid(logit + b)) == target."""
    for _ in range(it):
        mid = (lo + hi) / 2
        if _sig(logit + mid).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def make_mechanism(X, K, mech, p, rng):
    """Returns a function f(Xpart, rng) -> boolean missing mask."""
    n, d = X.shape
    if mech == "MCAR":
        return lambda Z, r: r.random(Z.shape) < p
    b = Blocks(K)
    if mech == "MAR":
        n_obs = max(1, int(round(0.3 * d)))
        always = rng.choice(d, n_obs, replace=False)
        rest = np.setdiff1d(np.arange(d), always)
        cols = np.concatenate([np.arange(b.off[j], b.off[j + 1]) for j in always])
        O = b.onehot(X)[:, cols]
        mu, sd = O.mean(0), O.std(0) + 1e-8
        dz = O.shape[1]
        Wt = rng.normal(0, np.sqrt(9.0 / dz), (dz, len(rest)))
        p_rest = min(p * d / len(rest), 0.99)          # overall rate equals p
        lg = ((O - mu) / sd) @ Wt
        bias = np.array([_calibrate(lg[:, i], p_rest) for i in range(len(rest))])

        def f(Z, r):
            Oz = (b.onehot(Z)[:, cols] - mu) / sd
            P = _sig(Oz @ Wt + bias)
            M = np.zeros(Z.shape, bool)
            M[:, rest] = r.random(P.shape) < P
            return M
        return f
    if mech == "MNAR":                                   # self-masking
        A = [rng.normal(0, 2.0, k) for k in K]           # a_jk ~ N(0, 4)
        bias = np.array([_calibrate(A[j][X[:, j]], p) for j in range(d)])

        def f(Z, r):
            P = np.stack([_sig(A[j][Z[:, j]] + bias[j]) for j in range(d)], 1)
            return r.random(P.shape) < P
        return f
    raise ValueError(mech)
