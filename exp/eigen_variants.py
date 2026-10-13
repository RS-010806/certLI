"""EigenLI scoring variants at equal memory (same stored eigenvectors, same cost).

EigenLI [Archish S et al., 2026]: s(Q, D) = sum_i ||Pi_D q_i||^2, Pi_D = projection on the top-k eigenvectors of
the document's token second-moment matrix. By the paper's Lemma 1, ||Pi_D q|| = max_{v in span, |v| = 1} <q, v>,
so the *unsquared* sum

    s_1(Q, D) = sum_i ||Pi_D q_i|| = sum_i max_{v in S_D, |v| = 1} <q_i, v>

is exactly MaxSim with the document's token set replaced by its k-dimensional subspace ("subspace MaxSim").
MaxSim adds per-token maxima linearly; EigenLI adds their squares. This script compares, at k = 16 and 32:
  p2        EigenLI as published (squared)
  p1        subspace MaxSim (unsquared)
  p2_shift  EigenLI after removing the corpus mean from documents and queries
  p1_shift  subspace MaxSim after removing the corpus mean
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

KS = (16, 32)


def bases(C, mu=None, kmax=32):
    B = np.zeros((C.N, C.dim, kmax), np.float32)
    for i in range(C.N):
        X = C.E[C.off[i]:C.off[i + 1]].astype(np.float64)
        if mu is not None:
            X = X - mu
        _, vec = np.linalg.eigh(X.T @ X)
        B[i] = vec[:, ::-1][:, :kmax]
    return torch.from_numpy(B)


def scores(C, W, B, mu=None):
    """Returns {(k, p): S} for k in KS and p in (1, 2)."""
    nq = len(C.qids)
    out = {(k, p): np.zeros((nq, C.N)) for k in KS for p in (1, 2)}
    Q = C.Q.astype(np.float64)
    if mu is not None:
        Q = Q - mu
    for qi in range(nq):
        q = torch.from_numpy(Q[qi].astype(np.float32))
        proj2 = torch.einsum("ld,ndk->nlk", q, B).pow(2)  # [N, Lq, kmax]
        cum = torch.cumsum(proj2, dim=2)
        for k in KS:
            e = cum[:, :, k - 1]  # ||Pi_k q_i||^2, [N, Lq]
            w = torch.from_numpy(W[qi].astype(np.float32))[None, :]
            out[(k, 2)][qi] = (e * w).sum(1).numpy()
            out[(k, 1)][qi] = (e.clamp(min=0).sqrt() * w).sum(1).numpy()
    return out


def main(m, d):
    t0 = time.time()
    C = Corpus(m, d); _, _, qrels = load_beir(d)
    base = f"{ROOT}/emb/{m}/{d}"
    S_ref = np.load(f"{base}/S_w1.npy").astype(np.float64); W = np.load(f"{base}/W_w1.npy")
    mu = C.E.astype(np.float64).mean(0)
    full = float(2 * C.dim * C.lens.mean())
    ev = evaluate_scores(S_ref, C.qids, C.dids, qrels)
    res = dict(model=m, dataset=d, n_docs=C.N, n_queries=len(C.qids), exact=dict(ndcg10=ev["ndcg@10"], recall100=ev["recall@100"]),
               rows=[])
    for tag, mm in (("raw", None), ("shift", mu)):
        B = bases(C, mm)
        S = scores(C, W, B, mm)
        for (k, p), Sk in sorted(S.items()):
            ev = evaluate_scores(Sk, C.qids, C.dids, qrels)
            row = dict(variant=f"p{p}" + ("_shift" if tag == "shift" else ""), k=k, p=p, shift=(tag == "shift"),
                       memory_frac=float(2 * C.dim * k / full), ndcg10=ev["ndcg@10"], recall100=ev["recall@100"],
                       overlap10=overlap_at_k(S_ref, Sk, C.self_doc), per_query_ndcg10=ev["per_query_ndcg@10"])
            res["rows"].append(row)
            print(m, d, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items() if kk != "per_query_ndcg10"},
                  f"{time.time()-t0:.0f}s", flush=True)
        del B
    os.makedirs(f"{ROOT}/results/eigen_variants", exist_ok=True)
    json.dump(res, open(f"{ROOT}/results/eigen_variants/{m}__{d}.json", "w"))


if __name__ == "__main__":
    for a in sys.argv[1:]:
        main(*a.split("/"))
