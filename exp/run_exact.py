"""E0/E1/E8: exact MaxSim effectiveness (uniform and IDF weights) and the Lemma 1 check."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from core import ROOT, Corpus, chamfer_all  # noqa: E402
from data import load_beir, evaluate_scores  # noqa: E402


def main(model, dataset):
    t0 = time.time()
    C = Corpus(model, dataset)
    _, _, qrels = load_beir(dataset)
    out_dir = f"{ROOT}/emb/{model}/{dataset}"
    W1 = np.ones(C.Q.shape[:2], np.float32)
    W1[C.QT < 0] = 0
    Widf = C.idf_weights(special_weight=1.0)
    res = dict(model=model, dataset=dataset, n_docs=C.N, n_queries=len(C.qids),
               n_tokens=int(C.off[-1]), mean_doc_tokens=float(C.lens.mean()), dim=C.dim)
    for tag, W in [("w1", W1), ("idf", Widf)]:
        t = time.time()
        S = chamfer_all(C.Q, W, C.E, C.off)
        np.save(f"{out_dir}/S_{tag}.npy", S.astype(np.float32))
        np.save(f"{out_dir}/W_{tag}.npy", W)
        ev = evaluate_scores(S, C.qids, C.dids, qrels)
        res[tag] = {k: v for k, v in ev.items() if not k.startswith("per_query")}
        res[tag]["time_s"] = time.time() - t
        res[tag + "_per_query_ndcg"] = ev["per_query_ndcg@10"]
        print(model, dataset, tag, res[tag], flush=True)
    # Lemma 1: shifting every document token by the corpus mean changes every score of a
    # query by the same constant, so rankings are unchanged.
    mu = C.E.mean(0)
    S1 = np.load(f"{out_dir}/S_w1.npy").astype(np.float64)
    Ssh = chamfer_all(C.Q, W1, (C.E - mu).astype(np.float32), C.off)
    const = np.einsum("qld,ql,d->q", C.Q.astype(np.float64), W1.astype(np.float64), mu.astype(np.float64))
    diff = Ssh + const[:, None] - S1
    same_top100 = []
    for qi in range(len(C.qids)):
        a = np.argsort(-S1[qi], kind="stable")[:100]
        b = np.argsort(-Ssh[qi], kind="stable")[:100]
        same_top100.append(bool(np.array_equal(a, b)) or bool(np.allclose(S1[qi][a], S1[qi][b], atol=1e-5)))
    res["lemma1"] = dict(max_abs_dev=float(np.abs(diff).max()),
                         mean_abs_dev=float(np.abs(diff).mean()),
                         frac_queries_identical_top100=float(np.mean(same_top100)),
                         mu_norm=float(np.linalg.norm(mu)))
    # per-document centering (NOT rank preserving) for contrast
    Ec = C.E.astype(np.float64).copy()
    for i in range(C.N):
        Ec[C.off[i]:C.off[i + 1]] -= Ec[C.off[i]:C.off[i + 1]].mean(0)
    Sdc = chamfer_all(C.Q, W1, Ec.astype(np.float32), C.off)
    ev = evaluate_scores(Sdc, C.qids, C.dids, qrels)
    res["perdoc_centering_ndcg@10"] = ev["ndcg@10"]
    res["time_total_s"] = time.time() - t0
    os.makedirs(f"{ROOT}/results/exact", exist_ok=True)
    json.dump(res, open(f"{ROOT}/results/exact/{model}__{dataset}.json", "w"), indent=1)
    print("lemma1", res["lemma1"], "perdoc-centering nDCG", res["perdoc_centering_ndcg@10"], flush=True)


if __name__ == "__main__":
    for ds in sys.argv[2:]:
        main(sys.argv[1], ds)
