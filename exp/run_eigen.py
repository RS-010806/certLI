"""EigenLI with and without the corpus-mean shift, plus Ward pooling, at k vectors per document (SciFact)."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import ROOT, Corpus, chamfer_all
from data import evaluate_scores, load_beir
from run_approx import eigenli_scores, overlap_at_k, pooled

def main(model, dataset, ks=(16, 32)):
    C = Corpus(model, dataset); _, _, qrels = load_beir(dataset)
    d = f"{ROOT}/emb/{model}/{dataset}"
    S_ref = np.load(f"{d}/S_w1.npy").astype(np.float64); W = np.load(f"{d}/W_w1.npy")
    mu = C.E.astype(np.float64).mean(0); full = float(2 * C.dim * C.lens.mean())
    ev = evaluate_scores(S_ref, C.qids, C.dids, qrels)
    rows = [dict(method="exact MaxSim", k=None, bytes_ratio=1.0, ndcg10=ev["ndcg@10"], recall100=ev["recall@100"], overlap10=1.0)]
    for k in ks:
        for tag, md, mq in (("EigenLI", None, None), ("EigenLI + shift", mu, None), ("EigenLI + shift (doc & query)", mu, mu)):
            t = time.time(); S = eigenli_scores(C, W, k, md, mq); ev = evaluate_scores(S, C.qids, C.dids, qrels)
            rows.append(dict(method=tag, k=k, bytes_ratio=float(2 * C.dim * k / full), ndcg10=ev["ndcg@10"],
                             recall100=ev["recall@100"], overlap10=overlap_at_k(S_ref, S, C.self_doc), time_s=time.time() - t))
            print(model, dataset, rows[-1], flush=True)
        t = time.time(); P, poff = pooled(C.E, C.off, k, "ward"); S = chamfer_all(C.Q, W, P, poff)
        ev = evaluate_scores(S, C.qids, C.dids, qrels)
        rows.append(dict(method="Ward pooling", k=k, bytes_ratio=float(2 * C.dim * np.diff(poff).mean() / full), ndcg10=ev["ndcg@10"],
                         recall100=ev["recall@100"], overlap10=overlap_at_k(S_ref, S, C.self_doc), time_s=time.time() - t))
        print(model, dataset, rows[-1], flush=True)
    os.makedirs(f"{ROOT}/results/approx", exist_ok=True)
    json.dump(dict(model=model, dataset=dataset, rows=rows), open(f"{ROOT}/results/approx/{model}__{dataset}.json", "w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
