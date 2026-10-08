"""How much could adaptivity help at all? For each query, the oracle depth is the rank (in the code's
point-estimate order) of the deepest exact top-10 document: reading that many documents in Shat order
recovers the exact top-10. Mean oracle depth = cost of a perfect per-query stopping rule; a fixed depth
that covers 1-alpha of queries = cost of the best non-adaptive rule."""
import glob, json, os
import numpy as np
from scipy import stats
ROOT = os.environ.get("CERTLI_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = {}
for p in sorted(glob.glob(f"{ROOT}/results/bounds/*.npz")):
    z = np.load(p)
    S, Sh, U, sd = z["S_sh"].astype(np.float64), z["Sh"].astype(np.float64), z["U"].astype(np.float64), z["self_doc"]
    nq, N = S.shape
    depth, dens = [], []
    for qi in range(nq):
        s, sh = S[qi].copy(), Sh[qi].copy()
        if sd[qi] >= 0:
            s[sd[qi]] = -np.inf; sh[sd[qi]] = -np.inf
        top = np.argpartition(-s, 10)[:10]
        order = np.argsort(-sh, kind="stable")
        rank = np.empty(N, int); rank[order] = np.arange(N)
        depth.append(int(rank[top].max()) + 1)
        sk = np.sort(s)[::-1][9]
        dens.append(int((s >= sk - 0.02 * abs(sk)).sum()))
    depth = np.array(depth); dens = np.array(dens)
    rho = stats.spearmanr(depth, dens)[0]
    key = os.path.basename(p)[:-4]
    out[key] = dict(N=N, oracle_mean=float(depth.mean()), oracle_median=float(np.median(depth)),
                    fixed_p90=float(np.quantile(depth, 0.9)), fixed_p95=float(np.quantile(depth, 0.95)),
                    fixed_p99=float(np.quantile(depth, 0.99)), oracle_max=int(depth.max()),
                    spearman_depth_vs_density=float(rho), depth=depth.tolist(), dens=dens.tolist())
    print(key, {k: v for k, v in out[key].items() if k not in ("depth", "dens")})
os.makedirs(f"{ROOT}/results/oracle", exist_ok=True)
json.dump(out, open(f"{ROOT}/results/oracle/all.json", "w"))
