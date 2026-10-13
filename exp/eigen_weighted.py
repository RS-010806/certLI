"""EigenLI with query-token importance weights (Weighted Chamfer [Archish S, Garg, Shiragur, Kayal, AAAI 2026]).

Same stored eigenvectors and cost as EigenLI; only the query-side weights change:
    s(Q, D) = sum_i w_i ||Pi_D q_i||^2,   w_i = IDF weight of query token i (W_idf from run_exact.py)
Variants: weights in {uniform, idf} x corpus-mean shift in {off, on}, k in {16, 32}.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import ROOT, Corpus  # noqa: E402
from data import evaluate_scores, load_beir  # noqa: E402
from eigen_variants import bases, scores  # noqa: E402
from run_approx import overlap_at_k  # noqa: E402


def main(m, d):
    t0 = time.time()
    C = Corpus(m, d); _, _, qrels = load_beir(d)
    base = f"{ROOT}/emb/{m}/{d}"
    S1 = np.load(f"{base}/S_w1.npy").astype(np.float64)
    Widf = np.load(f"{base}/W_idf.npy") if os.path.exists(f"{base}/W_idf.npy") else C.idf_weights(special_weight=1.0)
    Ws = {"uniform": np.load(f"{base}/W_w1.npy"), "idf": Widf}
    mu = C.E.astype(np.float64).mean(0)
    exact = dict(uniform=evaluate_scores(S1, C.qids, C.dids, qrels)["ndcg@10"])
    if os.path.exists(f"{base}/S_idf.npy"):
        exact["idf"] = evaluate_scores(np.load(f"{base}/S_idf.npy").astype(np.float64), C.qids, C.dids, qrels)["ndcg@10"]
    res = dict(model=m, dataset=d, exact=exact, rows=[])
    variants = (("raw", None),) if os.environ.get("RAW_ONLY") else (("raw", None), ("shift", mu))
    for tag, mm in variants:
        B = bases(C, mm)
        for wname, W in Ws.items():
            S = scores(C, W, B, mm)
            for k in (16, 32):
                ev = evaluate_scores(S[(k, 2)], C.qids, C.dids, qrels)
                row = dict(weights=wname, shift=(tag == "shift"), k=k, ndcg10=ev["ndcg@10"], recall100=ev["recall@100"],
                           overlap10_vs_exact_uniform=overlap_at_k(S1, S[(k, 2)], C.self_doc),
                           per_query_ndcg10=ev["per_query_ndcg@10"])
                res["rows"].append(row)
                print(m, d, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items()
                             if kk != "per_query_ndcg10"}, f"{time.time()-t0:.0f}s", flush=True)
        del B
    os.makedirs(f"{ROOT}/results/eigen_weighted", exist_ok=True)
    json.dump(res, open(f"{ROOT}/results/eigen_weighted/{m}__{d}.json", "w"))


if __name__ == "__main__":
    for a in sys.argv[1:]:
        main(*a.split("/"))
