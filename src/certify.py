"""Certification statistics, bound-ordered refinement and Learn-then-Test calibration."""
import heapq
import math

import numpy as np
from scipy import stats

TOL = 1e-6


def _mask_self(arrs, j):
    if j >= 0:
        for a in arrs:
            a[j] = -np.inf


def cert_stats(S, U, L, Shat, k, self_doc):
    """Per-query statistics of a certificate. S, U, L, Shat: [nq, N] in the same (shifted) space."""
    nq, N = S.shape
    out = dict(n_read=[], viol_lo=0, viol_hi=0, pairs=0, width=[], slack_up=[], err_hat=[],
               approx_recall=[], gap_k=[], s_k=[], frac_read=[])
    for qi in range(nq):
        s, u, l, sh = S[qi].copy(), U[qi].copy(), L[qi].copy(), Shat[qi].copy()
        _mask_self((s, u, l, sh), self_doc[qi])
        ok = np.isfinite(s)
        out["viol_lo"] += int((l[ok] > s[ok] + 1e-4).sum())
        out["viol_hi"] += int((u[ok] < s[ok] - 1e-4).sum())
        out["pairs"] += int(ok.sum())
        topk = np.argpartition(-s, k)[:k]
        s_k = s[topk].min()
        nread = int((u >= s_k - TOL).sum())
        out["n_read"].append(nread)
        out["frac_read"].append(nread / ok.sum())
        out["width"].append(float(np.mean(u[ok] - l[ok])))
        out["slack_up"].append(float(np.mean(u[ok] - s[ok])))
        out["err_hat"].append(float(np.mean(np.abs(sh[ok] - s[ok]))))
        th = np.argpartition(-sh, k)[:k]
        out["approx_recall"].append(len(set(th) & set(topk)) / k)
        srt = np.sort(s[ok])[::-1]
        out["gap_k"].append(float(srt[k - 1] - srt[k]))
        out["s_k"].append(float(s_k))
    return out


def refine(s, ub, k, order_key=None, max_reads=None):
    """Bound-ordered refinement. Reads docs in decreasing order_key (default: ub) and stops when the
    next doc's ub is below the current k-th best exact score. Returns (reads, topk_indices)."""
    key = ub if order_key is None else order_key
    order = np.argsort(-key, kind="stable")
    heap = []  # min-heap of (score, idx)
    reads = 0
    for idx in order:
        if not np.isfinite(s[idx]):
            continue
        if len(heap) == k and ub[idx] < heap[0][0] - TOL:
            break
        reads += 1
        if len(heap) < k:
            heapq.heappush(heap, (s[idx], idx))
        elif s[idx] > heap[0][0]:
            heapq.heapreplace(heap, (s[idx], idx))
        if max_reads is not None and reads >= max_reads:
            break
    top = [i for _, i in sorted(heap, reverse=True)]
    return reads, top


def shrink_sweep(S, U, Shat, k, self_doc, lambdas):
    """For each lambda, use U_lam = Shat + lam*(U - Shat) as (possibly invalid) upper bounds.
    Returns per-lambda arrays of reads and exact-top-k recall / miss indicator per query."""
    nq, N = S.shape
    reads = np.zeros((len(lambdas), nq), np.int64)
    rec = np.zeros((len(lambdas), nq))
    for qi in range(nq):
        s, u, sh = S[qi].copy(), U[qi].copy(), Shat[qi].copy()
        _mask_self((s, u, sh), self_doc[qi])
        true = set(np.argpartition(-s, k)[:k].tolist())
        for li, lam in enumerate(lambdas):
            ul = sh + lam * (u - sh)
            r, top = refine(s, ul, k)
            reads[li, qi] = r
            rec[li, qi] = len(true & set(top)) / k
    return reads, rec


def depth_sweep(S, Shat, k, self_doc, depths):
    """Fixed-depth rerank: read the top-R documents by Shat, return the exact top-k among them."""
    nq, N = S.shape
    rec = np.zeros((len(depths), nq))
    for qi in range(nq):
        s, sh = S[qi].copy(), Shat[qi].copy()
        _mask_self((s, sh), self_doc[qi])
        true = set(np.argpartition(-s, k)[:k].tolist())
        order = np.argsort(-sh, kind="stable")
        for di, R in enumerate(depths):
            cand = order[:R]
            cand = cand[np.isfinite(s[cand])]
            top = cand[np.argsort(-s[cand])[:k]]
            rec[di, qi] = len(true & set(top.tolist())) / k
    return rec


# ------------------------------------------------------------------ Learn then Test
def binom_pvalue(n_miss, n, alpha):
    """p-value for H0: risk > alpha with binary losses (exact binomial tail)."""
    return float(stats.binom.cdf(n_miss, n, alpha))


def hb_pvalue(rhat, n, alpha):
    """Hoeffding-Bentkus p-value (Angelopoulos et al., Learn then Test) for bounded losses."""
    def h1(a, b):
        a = min(max(a, 1e-12), 1 - 1e-12)
        return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))
    hoeff = math.exp(-n * h1(min(rhat, alpha), alpha)) if rhat < alpha else 1.0
    bent = math.e * stats.binom.cdf(math.ceil(n * rhat), n, alpha)
    return min(hoeff, bent, 1.0)


def ltt_select(loss_grid, alpha, delta, binary=True):
    """loss_grid: [n_params, n_cal] ordered from most conservative to most aggressive.
    Fixed-sequence testing; returns index of the last parameter whose null was rejected
    (or -1 if even the first one fails)."""
    n = loss_grid.shape[1]
    sel = -1
    for j in range(loss_grid.shape[0]):
        if binary:
            p = binom_pvalue(int(round(loss_grid[j].sum())), n, alpha)
        else:
            p = hb_pvalue(float(loss_grid[j].mean()), n, alpha)
        if p <= delta:
            sel = j
        else:
            break
    return sel
