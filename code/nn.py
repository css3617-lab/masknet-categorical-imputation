"""Minimal numpy neural-network toolkit with hand-written backward passes.

Used for all neural imputers (MIDAS-DAE, DAE-complete, Noise DNN, MaskNet, GAIN, ReMasker),
so that every neural model shares the same numerical substrate and optimiser.
"""
import numpy as np


# ----------------------------------------------------------------------------- activations
def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


ACT = {
    "softplus": (lambda x: np.logaddexp(0.0, x), lambda x, y: sigmoid(x)),
    "silu": (lambda x: x * sigmoid(x), lambda x, y: (lambda s: s + x * s * (1 - s))(sigmoid(x))),
    "relu": (lambda x: np.maximum(x, 0.0), lambda x, y: (x > 0).astype(x.dtype)),
}
_C = np.sqrt(2.0 / np.pi)


def gelu(x):
    t = np.tanh(_C * x * (1.0 + 0.044715 * x * x))
    return 0.5 * x * (1 + t), t


def gelu_grad(x, t):
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * _C * (1 + 3 * 0.044715 * x * x)


# ----------------------------------------------------------------------------- optimiser
class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        a = self.lr * np.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            self.p[k] -= a * self.m[k] / (np.sqrt(self.v[k]) + self.eps)


def glorot(rng, fan_in, fan_out):
    s = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0.0, s, (fan_in, fan_out))


# ----------------------------------------------------------------------------- MLP
class MLP:
    """Dense layers with a shared hidden activation and a linear output layer."""

    def __init__(self, sizes, act, rng, prefix="", params=None):
        self.sizes, self.act, self.prefix = sizes, act, prefix
        self.params = params if params is not None else {}
        for i in range(len(sizes) - 1):
            self.params[f"{prefix}W{i}"] = glorot(rng, sizes[i], sizes[i + 1])
            self.params[f"{prefix}b{i}"] = np.zeros(sizes[i + 1])
        self.L = len(sizes) - 1

    def forward(self, x):
        f, _ = ACT[self.act]
        cache = [x]
        h = x
        for i in range(self.L):
            z = h @ self.params[f"{self.prefix}W{i}"] + self.params[f"{self.prefix}b{i}"]
            if i < self.L - 1:
                cache.append(z)
                h = f(z)
                cache.append(h)
            else:
                h = z
        self._cache = cache
        return h

    def backward(self, g, grads, need_dx=False):
        _, df = ACT[self.act]
        c = self._cache
        for i in reversed(range(self.L)):
            hin = c[2 * i] if i > 0 else c[0]
            if i > 0:
                hin = c[2 * i]
            grads[f"{self.prefix}W{i}"] = grads.get(f"{self.prefix}W{i}", 0) + hin.T @ g
            grads[f"{self.prefix}b{i}"] = grads.get(f"{self.prefix}b{i}", 0) + g.sum(0)
            if i > 0 or need_dx:
                g = g @ self.params[f"{self.prefix}W{i}"].T
                if i > 0:
                    z = c[2 * i - 1]
                    g = g * df(z, None)
        return g if need_dx else None


# ----------------------------------------------------------------------------- categorical heads
class Blocks:
    """Bookkeeping for one-hot blocks of a categorical table."""

    def __init__(self, K):
        self.K = np.asarray(K)
        self.d = len(K)
        self.off = np.concatenate([[0], np.cumsum(K)])
        self.D = int(self.off[-1])
        self.var_of_col = np.repeat(np.arange(self.d), K)
        self.E = np.zeros((self.d, self.D))
        for j in range(self.d):
            self.E[j, self.off[j]:self.off[j + 1]] = 1.0

    def onehot(self, codes, miss=None):
        """codes: n x d int; miss: n x d bool (True = missing -> zero block)."""
        n = codes.shape[0]
        X = np.zeros((n, self.D))
        cols = codes + self.off[:-1]
        rows = np.repeat(np.arange(n), self.d)
        X[rows, cols.ravel()] = 1.0
        if miss is not None:
            X *= 1.0 - (miss.astype(float) @ self.E)
        return X

    def softmax(self, Z):
        P = np.empty_like(Z)
        for j in range(self.d):
            s, e = self.off[j], self.off[j + 1]
            z = Z[:, s:e] - Z[:, s:e].max(1, keepdims=True)
            ez = np.exp(z)
            P[:, s:e] = ez / ez.sum(1, keepdims=True)
        return P

    def expand(self, A):
        """n x d -> n x D (repeat per-variable values over their block)."""
        return A @ self.E

    def ce_grad(self, Z, Y, W, norm):
        """Weighted CE over blocks. W: n x d entry weights (0 = not in loss).
        Returns (loss, dZ)."""
        P = self.softmax(Z)
        Wx = self.expand(W).astype(Z.dtype)
        loss = -(Wx * Y * np.log(P + 1e-12)).sum() / norm
        dZ = ((P - Y) * Wx / norm).astype(Z.dtype)
        return loss, dZ, P

    def argmax(self, P):
        return np.stack([P[:, self.off[j]:self.off[j + 1]].argmax(1) for j in range(self.d)], 1)

    def sample(self, P, rng):
        out = np.empty((P.shape[0], self.d), dtype=np.int64)
        for j in range(self.d):
            p = P[:, self.off[j]:self.off[j + 1]]
            c = p.cumsum(1)
            u = rng.random((p.shape[0], 1)) * c[:, -1:]
            out[:, j] = np.minimum((u > c).sum(1), p.shape[1] - 1)
        return out


def class_weights(codes, obs, K):
    """w_jk = n_j / (K_j n_jk) from observed training entries (He & Garcia)."""
    W = []
    for j, k in enumerate(K):
        c = codes[obs[:, j], j]
        n = np.bincount(c, minlength=k).astype(float)
        w = np.where(n > 0, n.sum() / (k * np.maximum(n, 1)), 0.0)
        W.append(w)
    return W


def entry_weights(codes, W):
    """n x d matrix of class weights of the true category of each entry."""
    return np.stack([W[j][codes[:, j]] for j in range(codes.shape[1])], 1)


# ----------------------------------------------------------------------------- transformer pieces
def ln_fwd(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xh = (x - mu) * inv
    return g * xh + b, (xh, inv, g)


def ln_bwd(dy, cache):
    xh, inv, g = cache
    N = xh.shape[-1]
    dg = (dy * xh).reshape(-1, N).sum(0)
    db = dy.reshape(-1, N).sum(0)
    dxh = dy * g
    dx = inv / N * (N * dxh - dxh.sum(-1, keepdims=True) - xh * (dxh * xh).sum(-1, keepdims=True))
    return dx, dg, db


class Transformer:
    """Pre-LN transformer blocks with key masking; parameters live in a shared dict."""

    def __init__(self, params, prefix, depth, w, heads, hid, rng):
        self.p, self.pre, self.depth, self.w, self.h, self.hid = params, prefix, depth, w, heads, hid
        for l in range(depth):
            q = f"{prefix}{l}_"
            params[q + "ln1g"] = np.ones(w); params[q + "ln1b"] = np.zeros(w)
            params[q + "Wqkv"] = glorot(rng, w, 3 * w); params[q + "bqkv"] = np.zeros(3 * w)
            params[q + "Wo"] = glorot(rng, w, w); params[q + "bo"] = np.zeros(w)
            params[q + "ln2g"] = np.ones(w); params[q + "ln2b"] = np.zeros(w)
            params[q + "W1"] = glorot(rng, w, hid); params[q + "b1"] = np.zeros(hid)
            params[q + "W2"] = glorot(rng, hid, w); params[q + "b2"] = np.zeros(w)

    def forward(self, x, keymask):
        B, T, w = x.shape
        h, dh = self.h, w // self.h
        bias = ((1.0 - keymask)[:, None, None, :] * -1e9).astype(x.dtype)
        self.caches = []
        for l in range(self.depth):
            q = f"{self.pre}{l}_"
            P = self.p
            a_in, c_ln1 = ln_fwd(x, P[q + "ln1g"], P[q + "ln1b"])
            qkv = a_in @ P[q + "Wqkv"] + P[q + "bqkv"]
            Q, K_, V = [t.reshape(B, T, h, dh).transpose(0, 2, 1, 3) for t in np.split(qkv, 3, -1)]
            S = Q @ K_.transpose(0, 1, 3, 2) / np.sqrt(dh) + bias
            S = S - S.max(-1, keepdims=True)
            A = np.exp(S); A /= A.sum(-1, keepdims=True)
            O = (A @ V).transpose(0, 2, 1, 3).reshape(B, T, w)
            x1 = x + O @ P[q + "Wo"] + P[q + "bo"]
            m_in, c_ln2 = ln_fwd(x1, P[q + "ln2g"], P[q + "ln2b"])
            z1 = m_in @ P[q + "W1"] + P[q + "b1"]
            g1, t1 = gelu(z1)
            x2 = x1 + g1 @ P[q + "W2"] + P[q + "b2"]
            self.caches.append((a_in, c_ln1, Q, K_, V, A, O, m_in, c_ln2, z1, g1, t1))
            x = x2
        return x

    def backward(self, dx, grads):
        P = self.p
        for l in reversed(range(self.depth)):
            q = f"{self.pre}{l}_"
            a_in, c_ln1, Q, K_, V, A, O, m_in, c_ln2, z1, g1, t1 = self.caches[l]
            B, T, w = dx.shape
            h, dh = self.h, w // self.h
            # MLP
            grads[q + "W2"] = (g1.reshape(-1, self.hid).T @ dx.reshape(-1, w))
            grads[q + "b2"] = dx.reshape(-1, w).sum(0)
            dg1 = dx @ P[q + "W2"].T
            dz1 = dg1 * gelu_grad(z1, t1)
            grads[q + "W1"] = m_in.reshape(-1, w).T @ dz1.reshape(-1, self.hid)
            grads[q + "b1"] = dz1.reshape(-1, self.hid).sum(0)
            dm = dz1 @ P[q + "W1"].T
            dx1_ln, grads[q + "ln2g"], grads[q + "ln2b"] = ln_bwd(dm, c_ln2)
            dx1 = dx + dx1_ln
            # attention
            grads[q + "Wo"] = O.reshape(-1, w).T @ dx1.reshape(-1, w)
            grads[q + "bo"] = dx1.reshape(-1, w).sum(0)
            dO = (dx1 @ P[q + "Wo"].T).reshape(B, T, h, dh).transpose(0, 2, 1, 3)
            dA = dO @ V.transpose(0, 1, 3, 2)
            dV = A.transpose(0, 1, 3, 2) @ dO
            dS = A * (dA - (dA * A).sum(-1, keepdims=True)) / np.sqrt(dh)
            dQ = dS @ K_
            dK = dS.transpose(0, 1, 3, 2) @ Q
            dqkv = np.concatenate([t.transpose(0, 2, 1, 3).reshape(B, T, w) for t in (dQ, dK, dV)], -1)
            grads[q + "Wqkv"] = a_in.reshape(-1, w).T @ dqkv.reshape(-1, 3 * w)
            grads[q + "bqkv"] = dqkv.reshape(-1, 3 * w).sum(0)
            da = dqkv @ P[q + "Wqkv"].T
            dx0_ln, grads[q + "ln1g"], grads[q + "ln1b"] = ln_bwd(da, c_ln1)
            dx = dx1 + dx0_ln
        return dx
