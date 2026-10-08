"""E5: why are deterministic certificates loose? Measures, for sampled (query token, document token)
pairs, how far the Cauchy-Schwarz terms of Lemma 2 are from the realised deviations."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from codec import kcenter  # noqa: E402
from core import ROOT, Corpus  # noqa: E402


def main(model, dataset, t_rel=0.55, e_rel=0.5, k=16, n_docs=400, n_q=60, seed=0):
    C = Corpus(model, dataset)
    rng = np.random.default_rng(seed)
    E = C.E.astype(np.float64)
    mu = E.mean(0)
    sigma = float(np.sqrt(((E - mu) ** 2).sum(1).mean()))
    tau, eps = t_rel * sigma, e_rel * sigma
    docs = rng.choice(C.N, min(n_docs, C.N), replace=False)
    qs = rng.choice(len(C.qids), min(n_q, len(C.qids)), replace=False)
    Qt = C.Q[qs].reshape(-1, C.dim).astype(np.float64)
    cos_perp, cos_sub, ratio_pt, ratio_cl, dev_all, b_cl_all = [], [], [], [], [], []
    for i in docs:
        X = E[C.off[i]:C.off[i + 1]] - mu
        ev, vec = np.linalg.eigh(X.T @ X)
        U = vec[:, ::-1][:, :k]
        A = X @ U
        R = X - A @ U.T
        r = np.linalg.norm(R, axis=1)
        inl = np.where(r <= tau)[0]
        if len(inl) < 2:
            continue
        cidx, assign, dist = kcenter(A[inl], eps)
        Ai, Ri, ri = A[inl], R[inl], r[inl]
        Ac = Ai[cidx][assign]                  # centre coords for each inlier
        e_c = np.zeros(len(cidx)); t_c = np.zeros(len(cidx))
        np.maximum.at(e_c, assign, dist); np.maximum.at(t_c, assign, ri)
        G = Qt @ U                              # [nq_tok, k]
        Qp = Qt - G @ U.T                       # q_perp
        gn = np.linalg.norm(G, axis=1); h = np.linalg.norm(Qp, axis=1)
        D_sub = Ai - Ac                         # a - a_c
        dev_sub = G @ D_sub.T                   # <g, a - a_c>
        dev_perp = Qp @ Ri.T                    # <q_perp, r>
        nsub = np.linalg.norm(D_sub, axis=1)
        mask_sub = nsub > 1e-9
        cs = dev_sub[:, mask_sub] / (gn[:, None] * nsub[None, mask_sub] + 1e-12)
        cp = dev_perp / (h[:, None] * ri[None, :] + 1e-12)
        sel = rng.random(cs.shape) < 0.05
        cos_sub.append(cs[sel])
        sel2 = rng.random(cp.shape) < 0.05
        cos_perp.append(cp[sel2])
        dev = dev_sub + dev_perp
        b_pt = gn[:, None] * nsub[None, :] + h[:, None] * ri[None, :]
        b_cl = gn[:, None] * e_c[assign][None, :] + h[:, None] * t_c[assign][None, :]
        sel3 = rng.random(dev.shape) < 0.05
        ratio_pt.append(np.abs(dev[sel3]) / (b_pt[sel3] + 1e-12))
        ratio_cl.append(np.abs(dev[sel3]) / (b_cl[sel3] + 1e-12))
        dev_all.append(dev[sel3]); b_cl_all.append(b_cl[sel3])
    cos_perp = np.concatenate(cos_perp); cos_sub = np.concatenate(cos_sub)
    ratio_pt = np.concatenate(ratio_pt); ratio_cl = np.concatenate(ratio_cl)
    q = lambda x, p: float(np.quantile(np.abs(x), p))
    res = dict(model=model, dataset=dataset, k=k, tau_rel=t_rel, eps_rel=e_rel, dim=C.dim,
               perp_dim=C.dim - k,
               abs_cos_perp_median=q(cos_perp, 0.5), abs_cos_perp_p99=q(cos_perp, 0.99),
               abs_cos_sub_median=q(cos_sub, 0.5), abs_cos_sub_p99=q(cos_sub, 0.99),
               rms_cos_perp=float(np.sqrt(np.mean(cos_perp ** 2))),
               rms_cos_sub=float(np.sqrt(np.mean(cos_sub ** 2))),
               iso_prediction_perp=float(1 / np.sqrt(C.dim - k)), iso_prediction_sub=float(1 / np.sqrt(k)),
               dev_over_pointbound_median=q(ratio_pt, 0.5), dev_over_pointbound_p99=q(ratio_pt, 0.99),
               dev_over_clusterbound_median=q(ratio_cl, 0.5), dev_over_clusterbound_p99=q(ratio_cl, 0.99),
               dev_over_clusterbound_p999=q(ratio_cl, 0.999), n_pairs=int(len(ratio_cl)))
    os.makedirs(f"{ROOT}/results/slack", exist_ok=True)
    json.dump(res, open(f"{ROOT}/results/slack/{model}__{dataset}.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    for ds in sys.argv[2:]:
        main(sys.argv[1], ds)
