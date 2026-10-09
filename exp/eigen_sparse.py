"""EigenLI with a sparse part ("low-rank + sparse EigenLI"), at equal memory.

EigenLI [Archish S et al., 2026] stores the top-k eigenvectors w_1..w_k of each document's second-moment matrix
sum_j d_j d_j^T and scores s(Q, D) = sum_i ||Pi q_i||^2 = sum_i max_{v in span(w), |v|=1} <q_i, v>^2.

CertLI's structure result is that what a document subspace misses is concentrated on a few (mostly rare) tokens.
Low-rank + sparse EigenLI therefore spends part of the same budget of `budget` vectors per document on those tokens:
  stored: top-k eigenvectors (k = budget - r) and the r tokens with the largest residual off that subspace;
  score : s(Q, D) = sum_i max( ||Pi_k q_i||^2 , max_t max(<q_i, t>, 0)^2 ),
which is EigenLI's own max-over-unit-vectors form with the r stored tokens added to the candidate set.
Memory per document is budget * d floats, identical to EigenLI with k = budget.
Pre-registered split: r = budget / 4. Other splits are reported for transparency.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import ROOT, Corpus  # noqa: E402
from data import evaluate_scores, load_beir  # noqa: E402
from run_approx import overlap_at_k  # noqa: E402


def doc_eigs(C):
    vecs = []
    for i in range(C.N):
        X = C.E[C.off[i]:C.off[i + 1]].astype(np.float64)
        _, vec = np.linalg.eigh(X.T @ X)
        vecs.append(vec[:, ::-1].astype(np.float32))  # columns sorted by decreasing eigenvalue
    return vecs


def lrs_eigen_scores(C, W, vecs, k, r):
    N, d = C.N, C.dim
    B = np.zeros((N, d, k), np.float32)
    V = np.zeros((N, max(r, 1), d), np.float32)
    for i in range(N):
        X = C.E[C.off[i]:C.off[i + 1]].astype(np.float32)
        U = vecs[i][:, :k]
        B[i] = U
        if r > 0:
            res = np.linalg.norm(X - (X @ U) @ U.T, axis=1)
            idx = np.argsort(-res)[:r]
            V[i, :len(idx)] = X[idx]
    Bt, Vt = torch.from_numpy(B), torch.from_numpy(V)
    S = np.zeros((len(C.qids), N))
    for qi in range(len(C.qids)):
        q = torch.from_numpy(C.Q[qi].astype(np.float32))
        e = torch.einsum("ld,ndk->nlk", q, Bt).pow(2).sum(2)  # [N, Lq]
        if r > 0:
            t = torch.einsum("ld,nrd->nlr", q, Vt).clamp(min=0).pow(2).amax(2)
            e = torch.maximum(e, t)
        S[qi] = (e.numpy() * W[qi][None, :]).sum(1)
    return S


def main(m, d):
    t0 = time.time()
    C = Corpus(m, d); _, _, qrels = load_beir(d)
    base = f"{ROOT}/emb/{m}/{d}"
    S_ref = np.load(f"{base}/S_w1.npy").astype(np.float64); W = np.load(f"{base}/W_w1.npy")
    full = float(2 * C.dim * C.lens.mean())
    ev = evaluate_scores(S_ref, C.qids, C.dids, qrels)
    rows = [dict(method="exact MaxSim", budget=None, k=None, r=None, memory_frac=1.0, ndcg10=ev["ndcg@10"],
                 recall100=ev["recall@100"], overlap10=1.0)]
    vecs = doc_eigs(C)
    print(m, d, "eigs", f"{time.time()-t0:.0f}s", flush=True)
    for budget in (16, 32):
        for r in sorted({0, budget // 4, budget // 8, budget // 2}):
            k = budget - r
            S = lrs_eigen_scores(C, W, vecs, k, r)
            ev = evaluate_scores(S, C.qids, C.dids, qrels)
            row = dict(method="EigenLI" if r == 0 else "low-rank + sparse EigenLI", budget=budget, k=k, r=r,
                       preregistered=(r == budget // 4), memory_frac=float(2 * C.dim * budget / full),
                       ndcg10=ev["ndcg@10"], recall100=ev["recall@100"], overlap10=overlap_at_k(S_ref, S, C.self_doc))
            rows.append(row)
            print(m, d, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items()},
                  f"{time.time()-t0:.0f}s", flush=True)
    os.makedirs(f"{ROOT}/results/eigen_sparse", exist_ok=True)
    json.dump(dict(model=m, dataset=d, rows=rows), open(f"{ROOT}/results/eigen_sparse/{m}__{d}.json", "w"), indent=1)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        main(*a.split("/"))
