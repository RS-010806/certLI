"""CertLI codes and interval Chamfer bounds.

LRS  : low-rank-plus-sparse code (the proposal). Per document: corpus-mean shift, top-k_D
       eigenbasis U_D, tokens with residual > tau kept verbatim, the rest pooled by
       farthest-point (k-center) clustering to radius eps inside the k_D-dim subspace.
POOL : k-center pooling in the full space (no subspace, no verbatim set).
RQ   : PLAID-style per-token code (global centroid + b-bit residual) with a stored
       per-token error radius.
All codes expose `bounds(Qtok, W)` returning, per query and document, the interval [L, U]
on the exact weighted Chamfer score of the (shifted) document and a point estimate Shat.
"""
import numpy as np
import torch

NEG = -1e9


# ------------------------------------------------------------------ k-center
def kcenter(A, eps):
    """Gonzalez farthest-point clustering until every point is within eps of a centre.
    Returns centre indices, assignment, distance of each point to its centre."""
    n = A.shape[0]
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    c0 = int(np.argmax(np.linalg.norm(A - A.mean(0), axis=1)))
    centers = [c0]
    dist = np.linalg.norm(A - A[c0], axis=1)
    assign = np.zeros(n, np.int64)
    while True:
        j = int(np.argmax(dist))
        if dist[j] <= eps:
            break
        centers.append(j)
        dj = np.linalg.norm(A - A[j], axis=1)
        upd = dj < dist
        assign[upd] = len(centers) - 1
        dist = np.minimum(dist, dj)
    return np.asarray(centers), assign, dist


def kcenter_fixed(A, n_c):
    """Gonzalez clustering to exactly min(n_c, n) centres (used for budgeted pooling)."""
    n = A.shape[0]
    n_c = min(n_c, n)
    c0 = int(np.argmax(np.linalg.norm(A - A.mean(0), axis=1)))
    centers = [c0]
    dist = np.linalg.norm(A - A[c0], axis=1)
    assign = np.zeros(n, np.int64)
    while len(centers) < n_c:
        j = int(np.argmax(dist))
        centers.append(j)
        dj = np.linalg.norm(A - A[j], axis=1)
        upd = dj < dist
        assign[upd] = len(centers) - 1
        dist = np.minimum(dist, dj)
    return np.asarray(centers), assign, dist


# ------------------------------------------------------------------ generic container
class IntervalCode:
    """Stacked representation used for vectorised bounds.

    Clusters c (rows of Z) belong to doc cdoc[c] and give, for a query token q,
        ub_c = <q, Z_c> + min(eps_c*|U^T q| + tau_c*|q_perp|, full_c*|q|) + bias_c
        lb_c = <q, Z_c> - rho_c*|q_perp|                                     + bias_c
    where U is the doc basis (rows of Ustack, segment U_off). Verbatim rows give exact
    scores. If a doc has no basis (POOL / RQ), gn = |q| and q_perp = 0 by convention
    (handled through `has_basis`)."""

    def __init__(self):
        self.name = ""
        self.params = {}

    def finalize(self):
        # stacks are set by builders
        if not hasattr(self, "lbr"):
            self.lbr = np.zeros(len(self.eps))
        self.N = len(self.C_off) - 1
        self.cdoc = np.repeat(np.arange(self.N), np.diff(self.C_off))
        self.c_starts = self.C_off[:-1]
        self.v_starts = self.V_off[:-1]
        if self.Ustack is not None:
            self.u_starts = self.U_off[:-1]
        return self

    def _chunks(self, max_rows=250000):
        if getattr(self, "_chunk_cache", None) is not None:
            return self._chunk_cache
        chunks, c0 = [], 0
        while c0 < self.N:
            c1 = c0 + 1
            while c1 < self.N and (self.C_off[c1 + 1] - self.C_off[c0]) + (self.V_off[c1 + 1] - self.V_off[c0]) <= max_rows:
                c1 += 1
            ch = dict(c0=c0, c1=c1)
            z0, z1 = self.C_off[c0], self.C_off[c1]
            v0, v1 = self.V_off[c0], self.V_off[c1]
            ch["Z"] = torch.from_numpy(np.ascontiguousarray(self.Z[z0:z1], dtype=np.float32))
            ch["V"] = torch.from_numpy(np.ascontiguousarray(self.V[v0:v1], dtype=np.float32))
            ch["cst"] = self.C_off[c0:c1] - z0
            ch["vst"] = self.V_off[c0:c1] - v0
            ch["cdoc"] = (self.cdoc[z0:z1] - c0)
            for name in ("eps", "tau", "rho", "full", "bias", "lbr"):
                ch[name] = getattr(self, name)[z0:z1].astype(np.float32)[None, :]
            ch["vbias"] = self.vbias[v0:v1].astype(np.float32)[None, :]
            if self.Ustack is not None:
                u0, u1 = self.U_off[c0], self.U_off[c1]
                ch["U"] = torch.from_numpy(np.ascontiguousarray(self.Ustack[u0:u1], dtype=np.float32))
                ch["ust"] = self.U_off[c0:c1] - u0
            chunks.append(ch)
            c0 = c1
        self._chunk_cache = chunks
        return chunks

    def bounds(self, Qtok, W, q_batch=1, slack=1e-5):
        """Qtok [nq, Lq, d], W [nq, Lq] -> U, L, Shat each [nq, N] (float64).
        Computed in float32 per document chunk; `slack` per query token absorbs float32 rounding."""
        nq, Lq, d = Qtok.shape
        U = np.zeros((nq, self.N)); L = np.zeros((nq, self.N)); S = np.zeros((nq, self.N))
        chunks = self._chunks()
        for ch in chunks:
            if "t_eps" not in ch:
                for name in ("eps", "tau", "rho", "full", "bias", "lbr"):
                    ch["t_" + name] = torch.from_numpy(ch[name])
                ch["t_cdoc"] = torch.from_numpy(ch["cdoc"].astype(np.int64))
                ch["t_vbias"] = torch.from_numpy(ch["vbias"])
        for b0 in range(0, nq, q_batch):
            qb = Qtok[b0 : b0 + q_batch]
            B = qb.shape[0]
            q = np.ascontiguousarray(qb.reshape(B * Lq, d), dtype=np.float32)
            qt = torch.from_numpy(q)
            qn2 = (qt * qt).sum(1, keepdim=True)
            qn = torch.sqrt(qn2)
            wb = W[b0 : b0 + B].astype(np.float64)
            for ch in chunks:
                c0, c1 = ch["c0"], ch["c1"]
                n = c1 - c0
                CS = torch.matmul(qt, ch["Z"].T)
                CS += ch["t_bias"]
                if self.Ustack is not None:
                    P = torch.matmul(qt, ch["U"].T)
                    P.mul_(P)
                    gn2 = torch.from_numpy(np.add.reduceat(P.numpy(), ch["ust"], axis=1))
                    gn2 = torch.minimum(gn2, qn2)
                    gn = torch.sqrt(gn2)
                    h = torch.sqrt(torch.clamp(qn2 - gn2, min=0.0))
                    gnc = gn.index_select(1, ch["t_cdoc"])
                    hc = h.index_select(1, ch["t_cdoc"])
                    width = gnc.mul_(ch["t_eps"]).add_(hc * ch["t_tau"])
                    width = torch.minimum(width, ch["t_full"] * qn)
                    low = hc.mul_(ch["t_rho"]).add_(ch["t_lbr"] * qn)
                else:
                    width = torch.minimum(ch["t_eps"] * qn, ch["t_full"] * qn)
                    low = ch["t_lbr"] * qn
                cs_np = CS.numpy()
                cs_d = np.maximum.reduceat(cs_np, ch["cst"], axis=1)
                ub_d = np.maximum.reduceat((CS + width).numpy(), ch["cst"], axis=1)
                lb_d = np.maximum.reduceat((CS - low).numpy(), ch["cst"], axis=1)
                VS = torch.matmul(qt, ch["V"].T)
                VS += ch["t_vbias"]
                v_d = np.maximum.reduceat(VS.numpy(), ch["vst"], axis=1)
                u_tok = (np.maximum(ub_d, v_d) + slack).astype(np.float64).reshape(B, Lq, n)
                l_tok = (np.maximum(lb_d, v_d) - slack).astype(np.float64).reshape(B, Lq, n)
                s_tok = np.maximum(cs_d, v_d).astype(np.float64).reshape(B, Lq, n)
                U[b0 : b0 + B, c0:c1] = np.einsum("bln,bl->bn", u_tok, wb)
                L[b0 : b0 + B, c0:c1] = np.einsum("bln,bl->bn", l_tok, wb)
                S[b0 : b0 + B, c0:c1] = np.einsum("bln,bl->bn", s_tok, wb)
        return U, L, S


def _stack(rows_per_doc, d):
    off = np.zeros(len(rows_per_doc) + 1, np.int64)
    off[1:] = np.cumsum([len(r) for r in rows_per_doc])
    M = np.concatenate([r for r in rows_per_doc if len(r)], 0) if off[-1] else np.zeros((0, d))
    return M, off


# ------------------------------------------------------------------ LRS (proposal)
def build_lrs(E, off, tau, eps, k_grid=(4, 8, 16, 32), shift=True, mu=None,
              fixed_k=None, local_radii=True, eig_cache=None):
    """Low-rank-plus-sparse CertLI code. Returns IntervalCode with byte accounting."""
    N = len(off) - 1
    d = E.shape[1]
    if mu is None:
        mu = E.mean(0) if shift else np.zeros(d, E.dtype)
    Ulist, Zlist, Vlist = [], [], []
    eps_l, tau_l, rho_l, full_l, bias_l, vbias_l = [], [], [], [], [], []
    k_of, nc_of, nv_of, bytes_of, m_of = [], [], [], [], []
    grid = (fixed_k,) if fixed_k else k_grid
    for i in range(N):
        X = (E[off[i] : off[i + 1]] - mu).astype(np.float64)
        m = X.shape[0]
        if eig_cache is not None and i in eig_cache:
            Vec = eig_cache[i]
        else:
            w, Vec = np.linalg.eigh(X.T @ X)
            Vec = Vec[:, ::-1][:, : max(grid)].copy()
            if eig_cache is not None:
                eig_cache[i] = Vec
        best = None
        for k in grid:
            k = min(k, d)
            Ub = Vec[:, :k]
            A = X @ Ub
            r = np.linalg.norm(X - A @ Ub.T, axis=1)
            verb = r > tau
            inl = np.where(~verb)[0]
            if len(inl):
                cidx, assign, dist = kcenter(A[inl], eps)
                nc = len(cidx)
            else:
                cidx, assign, dist, nc = None, None, None, 0
            nv = int(verb.sum())
            nbytes = 2 * (d * k + nc * (k + 4) + nv * d)
            if best is None or nbytes < best[0]:
                best = (nbytes, k, Ub, A, r, verb, inl, cidx, assign, dist)
        nbytes, k, Ub, A, r, verb, inl, cidx, assign, dist = best
        Ulist.append(Ub.T.copy())
        Vlist.append(np.vstack([X[verb], np.zeros((1, d))]))
        vbias_l.append(np.r_[np.zeros(int(verb.sum())), NEG])
        if cidx is not None and len(cidx):
            Ai, ri = A[inl], r[inl]
            Zc = Ai[cidx] @ Ub.T
            nc = len(cidx)
            e_c = np.zeros(nc); t_c = np.zeros(nc); f_c = np.zeros(nc)
            np.maximum.at(e_c, assign, dist)
            np.maximum.at(t_c, assign, ri)
            np.maximum.at(f_c, assign, np.sqrt(dist ** 2 + ri ** 2))
            if not local_radii:  # global radii, exactly as stated in Lemma 2
                e_c[:] = eps; t_c[:] = tau; f_c[:] = np.inf
            Zlist.append(Zc); eps_l.append(e_c); tau_l.append(t_c); full_l.append(f_c)
            rho_l.append(ri[cidx] if local_radii else np.full(nc, tau)); bias_l.append(np.zeros(nc))
        else:
            nc = 0
            Zlist.append(np.zeros((1, d))); eps_l.append(np.zeros(1)); tau_l.append(np.zeros(1))
            full_l.append(np.zeros(1)); rho_l.append(np.zeros(1)); bias_l.append(np.array([NEG]))
        k_of.append(k); nc_of.append(nc); nv_of.append(int(verb.sum())); bytes_of.append(nbytes); m_of.append(m)
    c = IntervalCode()
    c.name = "LRS"
    c.params = dict(tau=tau, eps=eps, shift=shift, fixed_k=fixed_k, local_radii=local_radii)
    c.mu = mu
    c.Ustack, c.U_off = _stack(Ulist, d)
    c.Z, c.C_off = _stack(Zlist, d)
    c.V, c.V_off = _stack(Vlist, d)
    c.eps = np.concatenate(eps_l); c.tau = np.concatenate(tau_l); c.rho = np.concatenate(rho_l)
    c.full = np.concatenate(full_l); c.bias = np.concatenate(bias_l); c.vbias = np.concatenate(vbias_l)
    c.k_of = np.array(k_of); c.nc_of = np.array(nc_of); c.nv_of = np.array(nv_of)
    c.bytes_of = np.array(bytes_of); c.m_of = np.array(m_of)
    c.full_bytes = 2 * d * np.array(m_of)
    return c.finalize()


# ------------------------------------------------------------------ POOL (full-space k-center)
def build_pool(E, off, eps=None, n_centers=None, shift=True, mu=None):
    N = len(off) - 1
    d = E.shape[1]
    if mu is None:
        mu = E.mean(0) if shift else np.zeros(d, E.dtype)
    Zlist, eps_l, nc_of, bytes_of, m_of = [], [], [], [], []
    for i in range(N):
        X = (E[off[i] : off[i + 1]] - mu).astype(np.float64)
        if n_centers is not None:
            cidx, assign, dist = kcenter_fixed(X, n_centers)
        else:
            cidx, assign, dist = kcenter(X, eps)
        e_c = np.zeros(len(cidx))
        np.maximum.at(e_c, assign, dist)
        Zlist.append(X[cidx]); eps_l.append(e_c)
        nc_of.append(len(cidx)); bytes_of.append(2 * len(cidx) * (d + 1)); m_of.append(X.shape[0])
    c = IntervalCode()
    c.name = "POOL"
    c.params = dict(eps=eps, n_centers=n_centers, shift=shift)
    c.mu = mu
    c.Ustack = None
    c.Z, c.C_off = _stack(Zlist, d)
    c.eps = np.concatenate(eps_l)
    C = len(c.eps)
    c.tau = np.zeros(C); c.rho = np.zeros(C); c.full = c.eps.copy(); c.bias = np.zeros(C)
    c.V = np.zeros((N, d)); c.V_off = np.arange(N + 1); c.vbias = np.full(N, NEG)
    c.nc_of = np.array(nc_of); c.nv_of = np.zeros(N, int); c.k_of = np.zeros(N, int)
    c.bytes_of = np.array(bytes_of); c.m_of = np.array(m_of); c.full_bytes = 2 * d * np.array(m_of)
    return c.finalize()


# ------------------------------------------------------------------ RQ (PLAID-style with radii)
def build_rq(E, off, nbits=2, n_centroids=4096, shift=True, mu=None, seed=0, sample=200000, chunk=20000):
    """Per-token code: centroid id + nbits-per-dim residual (ColBERTv2-style quantile buckets)
    + one byte error radius (rounded up). Every token is kept; bound per token is
    <q, xhat_j> +/- e_j |q|. Built in chunks to bound memory."""
    from sklearn.cluster import MiniBatchKMeans

    N = len(off) - 1
    T, d = E.shape
    if mu is None:
        mu = E.mean(0) if shift else np.zeros(d, E.dtype)
    mu32 = mu.astype(np.float32)
    rng = np.random.default_rng(seed)
    sidx = np.sort(rng.choice(T, size=min(sample, T), replace=False))
    Xs = E[sidx].astype(np.float32) - mu32
    km = MiniBatchKMeans(n_clusters=n_centroids, random_state=seed, batch_size=8192, n_init=1,
                         max_iter=50).fit(Xs)
    cent = km.cluster_centers_.astype(np.float32)
    Ct = torch.from_numpy(cent)
    cn = (cent ** 2).sum(1)
    # bucket cutoffs from the sample residuals (assignments of the fitted sample)
    Rs = Xs - cent[km.labels_]
    nb = 2 ** nbits
    flat = Rs.ravel()
    cuts = np.quantile(flat, np.linspace(0, 1, nb + 1)[1:-1]).astype(np.float32)
    fb = np.searchsorted(cuts, flat)
    vals = np.array([flat[fb == b].mean() for b in range(nb)], np.float32)
    del Rs, flat, fb, Xs
    Xhat = np.empty((T, d), np.float32)
    err = np.empty(T, np.float32)
    for s0 in range(0, T, chunk):
        X = E[s0 : s0 + chunk].astype(np.float32) - mu32
        dd = cn[None, :] - 2 * torch.matmul(torch.from_numpy(X), Ct.T).numpy()
        ids = dd.argmin(1)
        del dd
        R = X - cent[ids]
        codes = np.searchsorted(cuts, R).astype(np.uint8)
        xh = cent[ids] + vals[codes]
        Xhat[s0 : s0 + chunk] = xh
        err[s0 : s0 + chunk] = np.linalg.norm(X - xh, axis=1)
    emax = float(err.max())
    err_q = np.ceil(err.astype(np.float64) / emax * 255.0) / 255.0 * emax  # one byte, rounded up
    c = IntervalCode()
    c.name = "RQ"
    c.params = dict(nbits=nbits, n_centroids=n_centroids, shift=shift)
    c.mu = mu
    c.Ustack = None
    c.Z = Xhat
    c.C_off = off.copy()
    c.eps = err_q
    c.tau = np.zeros(T); c.rho = np.zeros(T); c.full = err_q.copy(); c.bias = np.zeros(T)
    c.lbr = err_q.copy()  # lower bound <q, xhat> - e|q|
    c.V = np.zeros((N, d), np.float32); c.V_off = np.arange(N + 1); c.vbias = np.full(N, NEG)
    lens = np.diff(off)
    c.m_of = lens; c.nc_of = lens; c.nv_of = np.zeros(N, int); c.k_of = np.zeros(N, int)
    c.bytes_of = lens * (2 + nbits * d // 8 + 1)
    c.full_bytes = 2 * d * lens
    c.centroid_bytes = n_centroids * d * 2
    c.mean_err = float(err.mean())
    return c.finalize()
