"""E7: approximate (uncertified) training-free compressions at fixed vectors-per-document budgets:
k-means++ pooling, Ward pooling, EigenLI (raw), and EigenLI computed after the corpus-mean shift.
Reports nDCG@10, Recall@100 and agreement with the exact MaxSim top-10."""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from core import ROOT, Corpus, chamfer_all  # noqa: E402
from data import evaluate_scores, load_beir  # noqa: E402


def pooled(E, off, k, method):
    out, offs = [], [0]
    for i in range(len(off) - 1):
        X = E[off[i]:off[i + 1]].astype(np.float64)
        m = len(X)
        if m <= k:
            P = X
        elif method == "kmeans":
            km = KMeans(n_clusters=k, n_init=1, init="k-means++", random_state=0, max_iter=100).fit(X)
            lab = km.labels_
            P = np.stack([X[lab == c].mean(0) for c in range(k) if (lab == c).any()])
        else:
            Z = hierarchy.linkage(X, method="ward")
            lab = hierarchy.fcluster(Z, t=k, criterion="maxclust") - 1
            P = np.stack([X[lab == c].mean(0) for c in np.unique(lab)])
        out.append(P.astype(np.float32))
        offs.append(offs[-1] + len(P))
    return np.concatenate(out), np.array(offs)


def eigenli_scores(C, W, k, mu_doc=None, mu_query=None):
    """k-EigenLI: s(Q, D) = sum_i w_i ||Pi_D q_i||^2 with Pi_D the top-k eigenspace of the
    (optionally shifted) document second-moment matrix."""
    d = C.dim
    N = C.N
    bases = np.zeros((N, d, k), np.float32)
    for i in range(N):
        X = C.E[C.off[i]:C.off[i + 1]].astype(np.float64)
        if mu_doc is not None:
            X = X - mu_doc
        ev, vec = np.linalg.eigh(X.T @ X)
        kk = min(k, d)
        bases[i, :, :kk] = vec[:, ::-1][:, :kk]
    Bt = torch.from_numpy(bases.reshape(N, -1))  # [N, d*k]
    S = np.zeros((len(C.qids), N))
    Q = C.Q.astype(np.float64)
    if mu_query is not None:
        Q = Q - mu_query
    for qi in range(len(C.qids)):
        q = torch.from_numpy(Q[qi].astype(np.float32))  # [Lq, d]
        # ||U^T q||^2 for all docs: (q U_D) -> [N, Lq, k]
        proj = torch.einsum("ld,ndk->nlk", q, torch.from_numpy(bases))
        S[qi] = (proj.pow(2).sum(2).numpy() * W[qi][None, :]).sum(1)
    return S


def overlap_at_k(S_ref, S, self_doc, k=10):
    ov = []
    for qi in range(S.shape[0]):
        a, b = S_ref[qi].copy(), S[qi].copy()
        if self_doc[qi] >= 0:
            a[self_doc[qi]] = -np.inf; b[self_doc[qi]] = -np.inf
        ta = set(np.argpartition(-a, k)[:k]); tb = set(np.argpartition(-b, k)[:k])
        ov.append(len(ta & tb) / k)
    return float(np.mean(ov))


def main(model, dataset, ks=(16, 32)):
    C = Corpus(model, dataset)
    _, _, qrels = load_beir(dataset)
    d = f"{ROOT}/emb/{model}/{dataset}"
    S_ref = np.load(f"{d}/S_w1.npy").astype(np.float64)
    W = np.load(f"{d}/W_w1.npy")
    mu = C.E.astype(np.float64).mean(0)
    full_bytes = float(2 * C.dim * C.lens.mean())
    rows = []
    ev = evaluate_scores(S_ref, C.qids, C.dids, qrels)
    rows.append(dict(method="exact MaxSim", k=None, bytes_ratio=1.0, ndcg10=ev["ndcg@10"],
                     recall100=ev["recall@100"], overlap10=1.0))
    for k in ks:
        for method in ("kmeans", "ward"):
            t = time.time()
            P, poff = pooled(C.E, C.off, k, method)
            S = chamfer_all(C.Q, W, P, poff)
            ev = evaluate_scores(S, C.qids, C.dids, qrels)
            rows.append(dict(method=f"{method} pooling", k=k, bytes_ratio=float(2 * C.dim * np.diff(poff).mean() / full_bytes),
                             ndcg10=ev["ndcg@10"], recall100=ev["recall@100"],
                             overlap10=overlap_at_k(S_ref, S, C.self_doc), time_s=time.time() - t))
            print(model, dataset, rows[-1], flush=True)
        for tag, md, mq in (("EigenLI", None, None), ("EigenLI + doc shift", mu, None),
                            ("EigenLI + doc&query shift", mu, mu)):
            t = time.time()
            S = eigenli_scores(C, W, k, md, mq)
            ev = evaluate_scores(S, C.qids, C.dids, qrels)
            rows.append(dict(method=tag, k=k, bytes_ratio=float(2 * C.dim * k / full_bytes),
                             ndcg10=ev["ndcg@10"], recall100=ev["recall@100"],
                             overlap10=overlap_at_k(S_ref, S, C.self_doc), time_s=time.time() - t))
            print(model, dataset, rows[-1], flush=True)
    os.makedirs(f"{ROOT}/results/approx", exist_ok=True)
    json.dump(dict(model=model, dataset=dataset, rows=rows), open(f"{ROOT}/results/approx/{model}__{dataset}.json", "w"), indent=1)


if __name__ == "__main__":
    for ds in sys.argv[2:]:
        main(sys.argv[1], ds)
