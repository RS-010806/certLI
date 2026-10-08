"""Stopping rules with the same Learn-then-Test guarantee, evaluated on saved bounds.

All rules read documents in decreasing order of a score proxy and stop when the next proxy value is
below the current k-th best exact score:
  shrink  : proxy = Shat + lam * (U - Shat)       (shrunk certified interval, lam = 1 is exact)
  offset  : proxy = Shat + c                     (constant calibrated margin on the point estimate)
  depth   : read a fixed number R of documents in Shat order (no stopping rule)
Loss = 1{returned top-10 != exact top-10}; fixed-sequence testing from conservative to aggressive.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from certify import ltt_select, refine  # noqa: E402

ROOT = os.environ.get("CERTLI_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
K = 10


def family_eval(S, Sh, U, sd, kind, params):
    nq, N = S.shape
    reads = np.zeros((len(params), nq)); miss = np.zeros((len(params), nq)); rec = np.zeros((len(params), nq))
    for qi in range(nq):
        s, sh, u = S[qi].copy(), Sh[qi].copy(), U[qi].copy()
        if sd[qi] >= 0:
            s[sd[qi]] = -np.inf; sh[sd[qi]] = -np.inf; u[sd[qi]] = -np.inf
        true = set(np.argpartition(-s, K)[:K].tolist())
        order = np.argsort(-sh, kind="stable") if kind == "depth" else None
        for j, p in enumerate(params):
            if kind == "shrink":
                r, top = refine(s, sh + p * (u - sh), K)
            elif kind == "offset":
                r, top = refine(s, sh + p, K)
            else:
                cand = order[: int(p)]
                cand = cand[np.isfinite(s[cand])]
                top = cand[np.argsort(-s[cand])[:K]].tolist()
                r = min(int(p), N)
            reads[j, qi] = r
            got = len(true & set(top))
            rec[j, qi] = got / K
            miss[j, qi] = float(got < K)
    return reads, miss, rec


def ltt(reads, miss, rec, alpha, delta=0.1, n_splits=200, seed=0):
    nq = reads.shape[1]
    rng = np.random.default_rng(seed)
    rr, mm, cc, jj = [], [], [], []
    for _ in range(n_splits):
        perm = rng.permutation(nq)
        cal, te = perm[: nq // 2], perm[nq // 2:]
        j = max(ltt_select(miss[:, cal], alpha, delta, binary=True), 0)
        rr.append(reads[j, te].mean()); mm.append(miss[j, te].mean()); cc.append(rec[j, te].mean()); jj.append(j)
    return dict(reads=float(np.mean(rr)), miss=float(np.mean(mm)), recall=float(np.mean(cc)),
                frac_splits_miss_gt_alpha=float(np.mean(np.array(mm) > alpha)), param_index_median=float(np.median(jj)))


def main():
    prev = f"{ROOT}/results/stop/all.json"
    out = json.load(open(prev)) if os.path.exists(prev) else {}
    for p in sorted(glob.glob(f"{ROOT}/results/bounds/*.npz")):
        key = os.path.basename(p)[:-4]
        if key in out:
            continue
        z = np.load(p)
        S, Sh, U, sd = (z["S_sh"].astype(np.float64), z["Sh"].astype(np.float64), z["U"].astype(np.float64), z["self_doc"])
        N = S.shape[1]
        # parameter grids ordered from conservative to aggressive
        lam = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.275, 0.25, 0.225, 0.2, 0.175, 0.15, 0.125, 0.1, 0.075, 0.05, 0.025, 0.0]
        err = (S - Sh)[np.isfinite(S)]
        cmax = float(np.quantile(np.abs(err), 0.9999)) * 1.5
        offs = list(np.round(np.geomspace(cmax, cmax / 300, 40), 6)) + [0.0]
        depths = sorted({int(x) for x in np.geomspace(N, 10, 45)} | {N}, reverse=True)
        res = {"N": N, "rel": str(z["rel"])}
        for kind, grid in (("shrink", lam), ("offset", offs), ("depth", depths)):
            reads, miss, rec = family_eval(S, Sh, U, sd, kind, grid)
            res[kind] = dict(grid=[float(g) for g in grid], reads=reads.mean(1).tolist(), miss=miss.mean(1).tolist(),
                             recall=rec.mean(1).tolist(),
                             ltt05=ltt(reads, miss, rec, 0.05), ltt10=ltt(reads, miss, rec, 0.10))
        out[key] = res
        print(key, " | ".join(f"{k}: a=.05 reads {res[k]['ltt05']['reads']:.1f} miss {res[k]['ltt05']['miss']:.3f}; "
                              f"a=.10 reads {res[k]['ltt10']['reads']:.1f} miss {res[k]['ltt10']['miss']:.3f}"
                              for k in ("shrink", "offset", "depth")), flush=True)
        os.makedirs(f"{ROOT}/results/stop", exist_ok=True)
        json.dump(out, open(f"{ROOT}/results/stop/all.json", "w"))
    os.makedirs(f"{ROOT}/results/stop", exist_ok=True)
    json.dump(out, open(f"{ROOT}/results/stop/all.json", "w"))


if __name__ == "__main__":
    main()
