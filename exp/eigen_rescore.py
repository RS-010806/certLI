"""EigenLI vs CertLI on nDCG@10, and EigenLI as the first stage of a calibrated exact rescoring.

For each (model, corpus):
  * nDCG@10 of exact MaxSim, of EigenLI-style scoring at k = 16, 32 (my re-implementation, see run_approx.py),
    and of EigenLI at k = 32 after removing the corpus mean from documents and queries;
  * EigenLI + calibrated rescoring: documents are read in EigenLI order and rescored with exact MaxSim; the
    depth is chosen with Learn-then-Test (alpha = 0.05, delta = 0.1, 200 random 50/50 splits, loss =
    1{top-10 != exact}); reported: share of corpus rescored, realised error and nDCG@10 on the test halves;
  * oracle depth in EigenLI order (deepest exact top-10 document);
  * calibrated CertLI (shrunk intervals of the LRS code, tau = 0.5, eps = 0.45) with the same protocol, from
    the saved bound matrices of run_cert.py.
"""
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certify import ltt_select, refine  # noqa: E402
from core import ROOT, Corpus  # noqa: E402
from data import evaluate_scores, load_beir, ndcg_at_k  # noqa: E402
from run_approx import eigenli_scores  # noqa: E402

K = 10
ALPHA, DELTA, SPLITS = 0.05, 0.1, 200
LAMS = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.275, 0.25, 0.225, 0.2, 0.175, 0.15, 0.125, 0.1, 0.075, 0.05, 0.025, 0.0]


def per_query_eval(S_exact, order_scores, kind, grid, dids, qids, qrels, sd, U=None):
    """Returns reads, miss, ndcg arrays [len(grid), nq] for depth (in order_scores order) or shrink families."""
    nq, N = S_exact.shape
    reads = np.zeros((len(grid), nq)); miss = np.zeros((len(grid), nq)); nd = np.zeros((len(grid), nq))
    for qi in range(nq):
        s = S_exact[qi].copy(); o = order_scores[qi].copy()
        u = U[qi].copy() if U is not None else None
        if sd[qi] >= 0:
            s[sd[qi]] = -np.inf; o[sd[qi]] = -np.inf
            if u is not None:
                u[sd[qi]] = -np.inf
        true = set(np.argpartition(-s, K)[:K].tolist())
        rels = qrels[qids[qi]]
        order = np.argsort(-o, kind="stable") if kind == "depth" else None
        for j, p in enumerate(grid):
            if kind == "depth":
                cand = order[: int(p)]
                cand = cand[np.isfinite(s[cand])]
                top = cand[np.argsort(-s[cand], kind="stable")[:K]]
                r = min(int(p), N)
            else:
                r, top = refine(s, o + p * (u - o), K)
                top = np.array(top)[np.argsort(-s[np.array(top)], kind="stable")]
            reads[j, qi] = r
            miss[j, qi] = float(len(true & set(top.tolist())) < K)
            nd[j, qi] = ndcg_at_k([dids[t] for t in top], rels, K)
    return reads, miss, nd


def ltt(reads, miss, nd, N, seed=0):
    nq = reads.shape[1]
    rng = np.random.default_rng(seed)
    rr, mm, nn, jj = [], [], [], []
    for _ in range(SPLITS):
        perm = rng.permutation(nq)
        cal, te = perm[: nq // 2], perm[nq // 2:]
        j = max(ltt_select(miss[:, cal], ALPHA, DELTA, binary=True), 0)
        rr.append(reads[j, te].mean()); mm.append(miss[j, te].mean()); nn.append(nd[j, te].mean()); jj.append(j)
    return dict(reads_frac=float(np.mean(rr)) / N, reads_docs=float(np.mean(rr)), miss=float(np.mean(mm)),
                ndcg10=float(np.mean(nn)), param_index_median=float(np.median(jj)))


def oracle_depth(S_exact, order_scores, sd):
    nq, N = S_exact.shape
    dep = []
    for qi in range(nq):
        s = S_exact[qi].copy(); o = order_scores[qi].copy()
        if sd[qi] >= 0:
            s[sd[qi]] = -np.inf; o[sd[qi]] = -np.inf
        top = np.argpartition(-s, K)[:K]
        order = np.argsort(-o, kind="stable")
        rank = np.empty(N, int); rank[order] = np.arange(N)
        dep.append(int(rank[top].max()) + 1)
    return float(np.mean(dep))


def main(pairs):
    for m, d in pairs:
        key = f"{m}__{d}"
        t0 = time.time()
        C = Corpus(m, d); _, _, qrels = load_beir(d)
        base = f"{ROOT}/emb/{m}/{d}"
        S = np.load(f"{base}/S_w1.npy").astype(np.float64); W = np.load(f"{base}/W_w1.npy")
        N = S.shape[1]; sd = C.self_doc; dids = list(C.dids); qids = list(C.qids)
        mu = C.E.astype(np.float64).mean(0)
        full_bytes = float(2 * C.dim * C.lens.mean())
        res = dict(N=N, n_queries=len(qids), exact_ndcg10=evaluate_scores(S, qids, dids, qrels)["ndcg@10"])
        depths = sorted({int(x) for x in np.geomspace(N, 10, 60)} | {N}, reverse=True)
        for k in (16, 32):
            for tag, mdoc, mq in (("raw", None, None), ("centred", mu, mu)):
                if tag == "centred" and k != 32:
                    continue
                E = eigenli_scores(C, W, k, mdoc, mq)
                name = f"eigenli{k}" + ("_centred" if tag == "centred" else "")
                ev = evaluate_scores(E, qids, dids, qrels)
                row = dict(k=k, memory_frac=float(2 * C.dim * k / full_bytes), ndcg10=ev["ndcg@10"])
                if tag == "raw":
                    reads, miss, nd = per_query_eval(S, E, "depth", depths, dids, qids, qrels, sd)
                    row["rescore"] = ltt(reads, miss, nd, N)
                    row["oracle_depth_frac"] = oracle_depth(S, E, sd) / N
                    row["depth_grid"] = depths
                    row["depth_curve"] = dict(reads_frac=(reads.mean(1) / N).tolist(), miss=miss.mean(1).tolist(),
                                              ndcg10=nd.mean(1).tolist())
                res[name] = row
                print(key, name, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items()
                                  if kk not in ("depth_curve", "depth_grid")}, f"{time.time()-t0:.0f}s", flush=True)
        # calibrated CertLI from saved bounds (LRS tau=0.5, eps=0.45)
        for p in sorted(glob.glob(f"{ROOT}/results/bounds/{m}__{d}__w1__lrs__*.npz")):
            z = np.load(p)
            if '"tau": 0.5' not in str(z["rel"]) or '"eps": 0.45' not in str(z["rel"]):
                continue
            Ssh, Sh, U = z["S_sh"].astype(np.float64), z["Sh"].astype(np.float64), z["U"].astype(np.float64)
            mem = None
            for suite in ("main", "bonly"):
                fp = f"{ROOT}/results/cert/{m}__{d}__w1__{suite}.json"
                if os.path.exists(fp):
                    for r in json.load(open(fp))["rows"]:
                        if r["kind"] == "lrs" and r["rel"].get("tau") == 0.5 and r["rel"].get("eps") == 0.45 \
                                and r["rel"].get("shift", True) and r["rel"].get("local_radii", True):
                            mem = r["bytes_ratio"]
            reads, miss, nd = per_query_eval(Ssh, Sh, "shrink", LAMS, dids, qids, qrels, z["self_doc"], U=U)
            res["certli"] = dict(memory_frac=mem, rescore=ltt(reads, miss, nd, N),
                                 deterministic=dict(reads_frac=float(reads[0].mean() / N), ndcg10=float(nd[0].mean())))
            print(key, "certli", res["certli"], flush=True)
            break
        os.makedirs(f"{ROOT}/results/eigen_rescore", exist_ok=True)
        json.dump(res, open(f"{ROOT}/results/eigen_rescore/{key}.json", "w"), indent=1)


if __name__ == "__main__":
    pairs = [(a.split("/")[0], a.split("/")[1]) for a in sys.argv[1:]] or \
        [(m, d) for m in ("colbertv2", "answerai-small") for d in ("scifact", "nfcorpus")]
    main(pairs)
