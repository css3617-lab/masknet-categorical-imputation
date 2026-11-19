"""Imputation methods for fully categorical tables.

Every imputer has the signature
    fit_impute(Xtr, Mtr, Xte, Mte, K, rng, **opts) -> dict(name -> imputed test codes or probabilities)
Xtr/Xte: integer codes (values at missing positions are ignored); Mtr/Mte: True = missing.
Training uses only observed training entries, except DAE-complete, which receives Xtr_complete.
"""
import numpy as np
try:                                     # scikit-learn is needed only by the classical imputers
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import KNNImputer
except ImportError:
    pass
from nn import MLP, Adam, Blocks, class_weights, entry_weights, Transformer, ln_fwd, ln_bwd


def mode_codes(X, M, K):
    out = []
    for j, k in enumerate(K):
        c = X[~M[:, j], j]
        out.append(np.bincount(c, minlength=k).argmax() if len(c) else 0)
    return np.array(out)


def fill(X, M, values):
    Y = X.copy()
    Y[M] = np.broadcast_to(values, X.shape)[M]
    return Y


# ----------------------------------------------------------------------------- classical
def impute_mode(Xtr, Mtr, Xte, Mte, K, rng):
    return fill(Xte, Mte, mode_codes(Xtr, Mtr, K))


def impute_knn(Xtr, Mtr, Xte, Mte, K, rng, k=10):
    b = Blocks(K)
    Otr = b.onehot(Xtr); Otr[b.expand(Mtr.astype(float)) > 0] = np.nan
    Ote = b.onehot(Xte); Ote[b.expand(Mte.astype(float)) > 0] = np.nan
    imp = KNNImputer(n_neighbors=k, keep_empty_features=True).fit(Otr)
    P = imp.transform(Ote)
    return fill(Xte, Mte, b.argmax(P))


class _Chained:
    """Chained-equation imputer (MICE / MissForest) fitted inductively.

    Training: initialise missing training entries with the mode, then for `sweeps` sweeps,
    visit variables in order of increasing missingness, fit a classifier of variable j on
    the current completed values of all other variables (rows where j is observed), and
    update the missing entries of j. The fitted classifier of every (sweep, j) is stored;
    at test time the same sequence is replayed on the test table.
    Imputation is either deterministic (argmax) or a draw from the predicted distribution.
    """

    def __init__(self, K, make_clf, sweeps, features, stochastic, rng):
        self.K, self.make, self.sweeps, self.features, self.stoch, self.rng = K, make_clf, sweeps, features, stochastic, rng
        self.b = Blocks(K)

    def _feat(self, Xc, j):
        if self.features == "onehot":
            O = self.b.onehot(Xc)
            keep = self.b.var_of_col != j
            return O[:, keep]
        return np.delete(Xc, j, axis=1).astype(float)

    def _predict(self, clf, F):
        if isinstance(clf, (int, np.integer)):
            return np.full(F.shape[0], clf)
        if not self.stoch:
            return clf.predict(F)
        P = clf.predict_proba(F)
        u = self.rng.random((P.shape[0], 1))
        idx = np.minimum((u > P.cumsum(1)).sum(1), P.shape[1] - 1)
        return clf.classes_[idx]

    def fit_transform(self, X, M):
        self.mode = mode_codes(X, M, self.K)
        Xc = fill(X, M, self.mode)
        self.order = np.argsort(M.sum(0), kind="stable")
        self.models = []
        for s in range(self.sweeps):
            ms = {}
            for j in self.order:
                if not M[:, j].any():
                    continue
                obs = ~M[:, j]
                y = X[obs, j]
                if len(np.unique(y)) < 2:
                    clf = int(y[0]) if len(y) else int(self.mode[j])
                else:
                    clf = self.make().fit(self._feat(Xc[obs], j), y)
                ms[j] = clf
                Xc[M[:, j], j] = self._predict(clf, self._feat(Xc[M[:, j]], j))
            self.models.append(ms)
        return Xc

    def transform(self, X, M):
        Xc = fill(X, M, self.mode)
        for ms in self.models:
            for j in self.order:
                if j in ms and M[:, j].any():
                    Xc[M[:, j], j] = self._predict(ms[j], self._feat(Xc[M[:, j]], j))
        # variables never missing in training but missing at test: one pass with a fresh model
        return Xc


def impute_mice(Xtr, Mtr, Xte, Mte, K, rng, sweeps=3, stochastic=False):
    make = lambda: LogisticRegression(max_iter=300)          # multinomial (lbfgs) for K > 2
    ch = _Chained(K, make, sweeps, "onehot", stochastic, rng)
    ch.fit_transform(Xtr, Mtr)
    return ch.transform(Xte, Mte)


def impute_missforest(Xtr, Mtr, Xte, Mte, K, rng, sweeps=3, trees=50):
    seed = int(rng.integers(1 << 30))
    make = lambda: RandomForestClassifier(n_estimators=trees, n_jobs=1, random_state=seed)
    ch = _Chained(K, make, sweeps, "codes", False, rng)
    ch.fit_transform(Xtr, Mtr)
    return ch.transform(Xte, Mte)


# ----------------------------------------------------------------------------- helpers for neural models
def batches(n, bs, rng):
    idx = rng.permutation(n)
    for s in range(0, n, bs):
        yield idx[s:s + bs]


# ----------------------------------------------------------------------------- DAE family
def train_dae(Xtr, Mtr, K, rng, weighted, epochs=40, bs=256, drop=0.5, hidden=(256, 128, 256), lr=1e-3):
    """MIDAS-style DAE: observed variables dropped with prob `drop`; CE on all observed entries."""
    b = Blocks(K)
    net = MLP([b.D, *hidden, b.D], "softplus", rng)
    opt = Adam(net.params, lr)
    Y = b.onehot(Xtr)
    obs = ~Mtr
    W = entry_weights(Xtr, class_weights(Xtr, obs, K)) if weighted else np.ones(Xtr.shape)
    W = W * obs
    for ep in range(epochs):
        for idx in batches(len(Xtr), bs, rng):
            o = obs[idx]
            dropm = (rng.random(o.shape) < drop) & o
            x = Y[idx] * (1.0 - b.expand((~o | dropm).astype(float)))
            Z = net.forward(x)
            _, dZ, _ = b.ce_grad(Z, Y[idx], W[idx], len(idx))
            g = {}
            net.backward(dZ, g)
            opt.step(g)
    return b, net


def dae_impute(b, net, Xte, Mte):
    Z = net.forward(b.onehot(Xte, Mte))
    return b.softmax(Z)


# ----------------------------------------------------------------------------- Noise DNN (per-variable)
def impute_noise_dnn(Xtr, Mtr, Xte, Mte, K, rng, weighted=True, epochs=20, bs=256, hidden=64, lr=1e-3):
    """One sub-network per variable j: input = one-hot of all variables except j (missing = 0),
    two Softplus hidden layers, softmax over K_j; trained on rows where j is observed."""
    b = Blocks(K)
    Otr = b.onehot(Xtr, Mtr)
    Ote = b.onehot(Xte, Mte)
    cw = class_weights(Xtr, ~Mtr, K)
    out = Xte.copy()
    P_all = np.zeros((len(Xte), b.D))
    for j in range(b.d):
        s, e = b.off[j], b.off[j + 1]
        keep = np.ones(b.D, bool); keep[s:e] = False
        net = MLP([int(keep.sum()), hidden, hidden, int(K[j])], "softplus", rng)
        opt = Adam(net.params, lr)
        rows = np.where(~Mtr[:, j])[0]
        F, y = Otr[rows][:, keep], Xtr[rows, j]
        Yj = np.eye(K[j])[y]
        w = cw[j][y] if weighted else np.ones(len(y))
        for ep in range(epochs):
            for idx in batches(len(rows), bs, rng):
                Z = net.forward(F[idx])
                Z = Z - Z.max(1, keepdims=True)
                P = np.exp(Z); P /= P.sum(1, keepdims=True)
                dZ = (P - Yj[idx]) * w[idx, None] / len(idx)
                g = {}
                net.backward(dZ, g)
                opt.step(g)
        Z = net.forward(Ote[:, keep])
        Z = Z - Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        P_all[:, s:e] = P
    return P_all


# ----------------------------------------------------------------------------- MaskNet (proposed)
def train_masknet(Xtr, Mtr, K, rng, epochs=40, bs=256, hidden=256, lr=1e-3, r_range=(0.2, 0.6),
                  indicators=True, remask=True, weighted=False, loss_on="remasked", act="silu"):
    """Single shared network with missingness indicators, trained by stochastic re-masking.

    remask=False   -> no extra hiding; loss on visible (observed) entries.
    loss_on        -> 'remasked' (MaskNet) or 'observed' (re-masked + visible, ReMasker-style loss).
    r_range        -> (lo, hi) for r ~ U(lo, hi); use (0.5, 0.5) for a fixed ratio.
    """
    b = Blocks(K)
    d = b.d
    nin = b.D + (d if indicators else 0)
    net = MLP([nin, hidden, hidden, b.D], act, rng)
    opt = Adam(net.params, lr)
    Y = b.onehot(Xtr)
    obs = ~Mtr
    W0 = entry_weights(Xtr, class_weights(Xtr, obs, K)) if weighted else np.ones(Xtr.shape)
    for ep in range(epochs):
        for idx in batches(len(Xtr), bs, rng):
            o = obs[idx]
            if remask:
                r = rng.uniform(*r_range)
                R = (rng.random(o.shape) < r) & o
            else:
                R = np.zeros_like(o)
            hidden_m = ~o | R
            x = Y[idx] * (1.0 - b.expand(hidden_m.astype(float)))
            if indicators:
                x = np.concatenate([x, hidden_m.astype(float)], 1)
            if remask and loss_on == "remasked":
                L = R
            else:
                L = o
            if not L.any():
                continue
            Z = net.forward(x)
            _, dZ, _ = b.ce_grad(Z, Y[idx], W0[idx] * L, len(idx))
            g = {}
            net.backward(dZ, g)
            opt.step(g)
    return b, net, indicators


def masknet_proba(model, Xte, Mte):
    b, net, ind = model
    x = b.onehot(Xte, Mte)
    if ind:
        x = np.concatenate([x, Mte.astype(float)], 1)
    return b.softmax(net.forward(x))


# ----------------------------------------------------------------------------- GAIN
def impute_gain(Xtr, Mtr, Xte, Mte, K, rng, iters=3000, bs=256, hint=0.9, alpha=10.0, lr=1e-3):
    """GAIN (Yoon et al., 2018) adapted to one-hot data: generator emits one softmax per variable,
    discriminator predicts the per-variable mask given a hint; reconstruction = CE on observed."""
    b = Blocks(K)
    D, d = b.D, b.d
    G = MLP([D + d, D, D, D], "relu", rng, prefix="G")
    Dn = MLP([D + d, D, D, d], "relu", rng, prefix="D")
    optG, optD = Adam(G.params, lr), Adam(Dn.params, lr)
    Y = b.onehot(Xtr)
    Obs = (~Mtr).astype(float)
    n = len(Xtr)

    def gen(x, mo):
        mb = b.expand(mo)
        noise = rng.uniform(0, 0.01, x.shape)
        inp = np.concatenate([x * mb + noise * (1 - mb), mo], 1)
        P = b.softmax(G.forward(inp))
        return P, mb

    for it in range(iters):
        idx = rng.integers(0, n, bs)
        x, mo = Y[idx], Obs[idx]
        Bh = (rng.random(mo.shape) < hint).astype(float)
        H = mo * Bh + 0.5 * (1 - Bh)
        # --- discriminator step
        P, mb = gen(x, mo)
        xh = mb * x + (1 - mb) * P
        logit = Dn.forward(np.concatenate([xh, H], 1))
        Dp = 1 / (1 + np.exp(-logit))
        dlogit = (Dp - mo) / mo.size                       # BCE gradient
        gD = {}
        Dn.backward(dlogit, gD)
        optD.step(gD)
        # --- generator step
        P, mb = gen(x, mo)
        Zcache = G._cache
        xh = mb * x + (1 - mb) * P
        logit = Dn.forward(np.concatenate([xh, H], 1))
        Dp = 1 / (1 + np.exp(-logit))
        # adversarial: -mean((1-m) log D)
        dlogit = -(1 - mo) * (1 - Dp) / mo.size
        gtmp = {}
        dinp = Dn.backward(dlogit, gtmp, need_dx=True)
        dxh = dinp[:, :D]
        dP = dxh * (1 - mb)
        # reconstruction: alpha * CE on observed entries
        nobs = max(mo.sum(), 1.0)
        dP = dP + alpha * (-(mb * x) / (P + 1e-12)) / nobs
        # softmax backward per block
        s = b.expand((dP * P) @ b.E.T)
        dZ = P * (dP - s)
        G._cache = Zcache
        gG = {}
        G.backward(dZ, gG)
        optG.step(gG)
    Ote = b.onehot(Xte)
    P, mb = gen(Ote, (~Mte).astype(float))
    return P


# ----------------------------------------------------------------------------- ReMasker (categorical adaptation)
class ReMasker:
    """ReMasker (Du et al., ICLR 2024) adapted to categorical data.

    Each variable is a token: category embedding + positional embedding. The encoder attends only
    to visible tokens (key mask); the decoder replaces hidden tokens by a learnable mask token,
    adds its own positional embedding, and predicts every variable with a per-variable softmax.
    Training: re-mask a fixed ratio of observed entries; loss on re-masked AND visible entries
    (the best configuration in the original paper), cross-entropy instead of MSE.
    """

    def __init__(self, K, rng, w=32, heads=4, enc_depth=4, dec_depth=2, mlp=64, lr=1e-3):
        self.b = Blocks(K)
        d, D = self.b.d, self.b.D
        p = {}
        p["emb"] = rng.normal(0, 0.02, (D, w))
        p["pos_e"] = rng.normal(0, 0.02, (d, w))
        p["pos_d"] = rng.normal(0, 0.02, (d, w))
        p["mtok"] = rng.normal(0, 0.02, (w,))
        p["lneg"] = np.ones(w); p["lneb"] = np.zeros(w)
        p["lndg"] = np.ones(w); p["lndb"] = np.zeros(w)
        p["Wout"] = rng.normal(0, 0.02, (w, D)); p["bout"] = np.zeros(D)
        self.p = p
        self.enc = Transformer(p, "enc", enc_depth, w, heads, mlp, rng)
        self.dec = Transformer(p, "dec", dec_depth, w, heads, mlp, rng)
        for k in p:                                   # float32 for CPU speed
            p[k] = p[k].astype(np.float32)
        self.opt = Adam(p, lr)
        self.w = w

    def forward(self, codes, vis):
        b, p = self.b, self.p
        B, d = codes.shape
        cols = codes + b.off[:-1]
        x = p["emb"][cols] + p["pos_e"][None]
        x = x * vis[..., None]
        h = self.enc.forward(x, vis.astype(np.float32))
        he, c_e = ln_fwd(h, p["lneg"], p["lneb"])
        v = vis[..., None].astype(np.float32)
        z = he * v + p["mtok"][None, None] * (1 - v) + p["pos_d"][None]
        u = self.dec.forward(z, np.ones((B, d), np.float32))
        ud, c_d = ln_fwd(u, p["lndg"], p["lndb"])
        L3 = ud @ p["Wout"]                                   # B x d x D
        Z = L3[:, b.var_of_col, np.arange(b.D)] + p["bout"]   # B x D
        self.cache = (cols, vis, v, c_e, c_d, ud)
        return Z

    def backward(self, dZ):
        b, p = self.b, self.p
        cols, vis, v, c_e, c_d, ud = self.cache
        B, d, w = ud.shape
        g = {}
        g["bout"] = dZ.sum(0)
        dL3 = np.zeros((B, d, b.D), np.float32)
        dL3[:, b.var_of_col, np.arange(b.D)] = dZ
        g["Wout"] = ud.reshape(-1, w).T @ dL3.reshape(-1, b.D)
        dud = dL3 @ p["Wout"].T
        du, g["lndg"], g["lndb"] = ln_bwd(dud, c_d)
        dz = self.dec.backward(du, g)
        g["pos_d"] = dz.sum(0)
        g["mtok"] = (dz * (1 - v)).sum((0, 1))
        dhe = dz * v
        dh, g["lneg"], g["lneb"] = ln_bwd(dhe, c_e)
        dx = self.enc.backward(dh, g)
        dx = dx * vis[..., None]
        g["pos_e"] = dx.sum(0)
        gemb = np.zeros_like(p["emb"])
        np.add.at(gemb, cols.ravel(), dx.reshape(-1, w))
        g["emb"] = gemb
        return g

    def fit(self, X, M, rng, epochs=100, bs=256, ratio=0.5):
        b = self.b
        Y = b.onehot(X).astype(np.float32)
        obs = ~M
        for ep in range(epochs):
            for idx in batches(len(X), bs, rng):
                o = obs[idx]
                R = (rng.random(o.shape) < ratio) & o
                vis = o & ~R
                Z = self.forward(X[idx], vis)
                _, dZ, _ = b.ce_grad(Z, Y[idx], o.astype(np.float32), len(idx))
                self.opt.step(self.backward(dZ))
        return self

    def proba(self, X, M, bs=1000):
        out = []
        for s in range(0, len(X), bs):
            out.append(self.b.softmax(self.forward(X[s:s + bs], ~M[s:s + bs])))
        return np.concatenate(out)
