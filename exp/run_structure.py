"""E2: geometry of late-interaction token sets.
- anisotropy (mean cosine between tokens of different documents, |mu|)
- per-document spectral energy captured by the top-k eigenvectors (raw / corpus-mean shift /
  per-document centering)
- residual structure after a rank-k projection: tail heaviness and correlation with IDF.
"""
import json
import math
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from core import ROOT, Corpus  # noqa: E402

KS = (2, 4, 8, 16, 32)


def main(model, dataset, max_docs=3000, seed=0):
    C = Corpus(model, dataset)
    rng = np.random.default_rng(seed)
    E = C.E.astype(np.float64)
    mu = E.mean(0)
    # anisotropy
    a = rng.integers(0, len(E), 200000)
    b = rng.integers(0, len(E), 200000)
    diffdoc = C.doc_of_tok[a] != C.doc_of_tok[b]
    cos = (E[a] * E[b]).sum(1)[diffdoc]
    Es = E - mu
    ns = np.linalg.norm(Es, axis=1)
    cos_s = ((Es[a] * Es[b]).sum(1) / (ns[a] * ns[b]))[diffdoc]
    res = dict(model=model, dataset=dataset, mu_norm=float(np.linalg.norm(mu)),
               mean_cos_between_docs=float(cos.mean()), mean_cos_between_docs_shifted=float(cos_s.mean()),
               mean_shifted_norm=float(ns.mean()))
    # global spectrum of all tokens (second moment) raw vs shifted
    sub = E[rng.choice(len(E), min(200000, len(E)), replace=False)]
    for tag, X in [("raw", sub), ("shift", sub - mu)]:
        ev = np.linalg.eigvalsh(X.T @ X)[::-1]
        res[f"global_energy_top1_{tag}"] = float(ev[0] / ev.sum())
        res[f"global_eff_rank_{tag}"] = float(math.exp(stats.entropy(ev / ev.sum())))
    # per-document spectra
    docs = rng.choice(C.N, min(max_docs, C.N), replace=False)
    energy = {t: {k: [] for k in KS} for t in ("raw", "shift", "center")}
    effrank = {t: [] for t in ("raw", "shift", "center")}
    top1_dir_cos_mu = []
    rows_resid, rows_idf = [], []
    # document frequency for IDF of document tokens
    pairs = np.unique(np.stack([C.doc_of_tok, C.tok], 1), axis=0)
    df = np.bincount(pairs[:, 1], minlength=int(C.tok.max()) + 1)
    idf_tok = np.log((C.N - df + 0.5) / (df + 0.5) + 1.0)
    resid_k = 16
    tail_share = []
    for i in docs:
        X = E[C.off[i]:C.off[i + 1]]
        m = len(X)
        for tag, Y in (("raw", X), ("shift", X - mu), ("center", X - X.mean(0))):
            ev, vec = np.linalg.eigh(Y.T @ Y)
            ev = np.clip(ev[::-1], 0, None)
            tot = ev.sum()
            for k in KS:
                energy[tag][k].append(ev[:k].sum() / tot)
            p = ev / tot
            effrank[tag].append(math.exp(stats.entropy(p[p > 0])))
            if tag == "raw":
                top1_dir_cos_mu.append(abs(vec[:, -1] @ mu) / np.linalg.norm(mu))
            if tag == "shift":
                U = vec[:, ::-1][:, :resid_k]
                r = np.linalg.norm(Y - (Y @ U) @ U.T, axis=1)
                rows_resid.append(r)
                rows_idf.append(idf_tok[C.tok[C.off[i]:C.off[i + 1]]])
                e2 = np.sort(r ** 2)[::-1]
                top = max(1, int(round(0.1 * m)))
                tail_share.append(e2[:top].sum() / max(e2.sum(), 1e-12))
    res["doc_energy"] = {t: {str(k): float(np.mean(v)) for k, v in energy[t].items()} for t in energy}
    res["doc_eff_rank"] = {t: float(np.mean(v)) for t, v in effrank.items()}
    res["top1_eigvec_abs_cos_with_mu"] = float(np.mean(top1_dir_cos_mu))
    r = np.concatenate(rows_resid)
    idf = np.concatenate(rows_idf)
    res["resid_k"] = resid_k
    res["resid_quantiles"] = {str(q): float(np.quantile(r, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)}
    res["resid_frac_above"] = {str(t): float((r > t).mean()) for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8)}
    res["resid_energy_share_top10pct_tokens"] = float(np.mean(tail_share))
    rho, p = stats.spearmanr(idf, r)
    res["spearman_idf_resid"] = float(rho)
    lo, hi = np.quantile(idf, [0.1, 0.9])
    res["mean_resid_low_idf_decile"] = float(r[idf <= lo].mean())
    res["mean_resid_high_idf_decile"] = float(r[idf >= hi].mean())
    os.makedirs(f"{ROOT}/results/structure", exist_ok=True)
    json.dump(res, open(f"{ROOT}/results/structure/{model}__{dataset}.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    for ds in sys.argv[2:]:
        main(sys.argv[1], ds)
